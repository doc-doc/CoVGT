import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
# from fvcore.nn import FlopCountAnalysis

def eval(model, data_loader, a2v, args, test=False, tokenizer="RoBERTa"):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            answer_id, answer, video_o, video_f, question, question_id, seg_feats, seg_num = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["question"].cuda(),
                batch['question_id'],
                batch['seg_feats'].cuda(),
                batch['seg_num']
            )
           
            video_len = batch["video_len"]
            seq_len = batch["seq_len"]
           
            question_mask = (question!=tokenizer.pad_token_id).float() #RobBERETa
            answer_mask = (answer!=tokenizer.pad_token_id).float() #RobBERETa

            video_mask = get_mask(video_len, video_o.size(1)).cuda()
            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(model, inputs)
                # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
                # break
                ###################################
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer,
                    seq_len = seq_len,
                    seg_feats = seg_feats,
                    seg_num = seg_num
                )
                # predicts = fusion_proj.squeeze() 
                
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}

    step = "val" if not test else "test"
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss, running_cl_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter()
    )
    for i, batch in enumerate(train_loader):
        answer_id, answer, video_o, video_f, question, seg_feats, seg_num, qsn_id, qsn_token_ids, qsn_seq_len = (
            batch["answer_id"],
            batch["answer"],
            batch["video_o"].cuda(),
            batch["video_f"].cuda(),
            batch["question"].cuda(),
            batch['seg_feats'].cuda(),
            batch['seg_num'],
            batch['qsn_id'],
            batch['qsn_token_ids'],
            batch['qsn_seq_len']
        )
        
        video_len = batch["video_len"]
        
        question_mask = (question != tokenizer.pad_token_id).float().cuda() #RobBERETa
        answer_mask = (answer!=tokenizer.pad_token_id).float().cuda() #RobBERETa
        video_mask = (
            get_mask(video_len, video_o.size(1)).cuda() if args.max_feats > 0 else None
        )
       
        qsn_mask = (qsn_token_ids != tokenizer.pad_token_id).float().cuda()
        
        video = (video_o, video_f)
        N = answer_id.size(0)
        seq_len = batch["seq_len"]
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                seq_len = seq_len
            )
        else:
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=answer_mask,
                video_mask=video_mask,
                answer=answer.cuda(),
                seq_len = seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num
            )
                    
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu() 
            running_acc.update((predicted == answer_id).sum().item() / N, N)
        if args.cl_loss:
            vt_proj, txt_proj = model(
                video,
                question,
                text_mask=qsn_mask,
                video_mask=video_mask,
                answer=qsn_token_ids,
                seq_len = qsn_seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num
            )
            vt_proj = vt_proj.unsqueeze(2)
            cl_predicts = torch.bmm(txt_proj, vt_proj).squeeze()
            cl_loss = criterion(cl_predicts, qsn_id.cuda())
            # cl_predicted = torch.max(cl_predicts, dim=1).indices.cpu()
            # running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            max_seq_len = args.qmax_words
            if args.mc > 0:
                tmp_id = [aid+(args.mc*i) for i, aid in enumerate(answer_id)]
                inputs = answer.view(N*args.mc, -1)[tmp_id,:]
                # question_mask = (inputs>0).float()
                question_mask = (inputs!=1).float()
                max_seq_len = args.amax_words
            else:
                inputs = batch["question"]
            
            inputs, labels = mask_tokens(inputs, tokenizer, mlm_probability=args.mlm_prob)
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                max_seq_len=max_seq_len,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        if args.cl_loss:
            loss = vqa_loss + args.cl_loss*cl_loss
        if args.cl_loss and args.mlm_prob:
            loss = vqa_loss + args.cl_loss*cl_loss + mlm_loss
        if not args.cl_loss and not args.mlm_prob:
            loss = vqa_loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()
        
        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if args.cl_loss:
            running_cl_loss.update(cl_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}, Lvq Loss: {running_cl_loss.avg:.4f}"
                )
            elif args.cl_loss:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}, Lvq Loss: {running_cl_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
            running_cl_loss.reset()
