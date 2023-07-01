
import sys
sys.path.insert(0, '../')
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
from util import tokenize, transform_bb, load_file, pkload, group, get_qsn_type
from tools.object_align import align
import os.path as osp
import h5py
import random as rd
import numpy as np

class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        amax_words=5,
        tokenizer=None,
        a2id=None,
        max_feats=20,
        mc=0,
        bnum=10,
        cl_loss=0
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param tokenizer: tokenizer
        :param a2id: answer to index mapping
        :param max_feats: maximum frames to sample from a video
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.dset = self.csv_path.split('/')[-2]
        
        self.video_feature_path = features
        self.bbox_num = bnum
        self.use_frame = True
        self.use_mot =  False
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.tokenizer = tokenizer
        self.max_feats = max_feats
        self.mc = mc
        self.lvq = cl_loss
        self.mode = osp.basename(csv_path).split('.')[0] #train, val or test
        
        # self.all_answers = set(self.data['answer'])
        
        if self.mode not in ['val', 'test']:
            self.all_answers = set(self.data['answer'])
            self.all_questions = set(self.data['question'])
            self.ans_group, self.qsn_group = group(self.data, gt=False)

        if self.dset == 'star':
            self.vid_clips = load_file(osp.dirname(csv_path)+f'/clips_{self.mode}.json')
        
        if self.dset == 'causalvid':
            data_dir = osp.dirname(csv_path)
            self.map_dir = load_file(osp.join(data_dir, 'map_dir_caul.json'))
            # vids = pkload(osp.join(data_dir, f'split/{self.mode}.pkl'))
            # self.txt_obj = {}
            # with h5py.File(osp.join(data_dir, 'ROI_text.h5'), 'r') as f:
            #     keys = [item for item in vids if item in f.keys()]
            #     for key in keys:
            #         tmp = {}
            #         labels = f[key].keys()
            #         for label in labels:
            #             new_label = label.replace('_', '')
            #             tmp[new_label] = f[key][label][...]
            #         self.txt_obj[key] = tmp

        if self.dset not in ['webvid', 'frameqa', 'star', 'causalvid']:
            filen = bnum
            if self.dset == 'nextqa': filen = 20
            bbox_feat_file = osp.join(self.video_feature_path, f'region_feat_n/acregion_8c{filen}b_{self.mode}.h5')
            print('Load {}...'.format(bbox_feat_file))          
            self.bbox_feats = {}
            with h5py.File(bbox_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                print(feats.shape) #v_num, clip_num, region_per_frame, feat_dim
                bboxes = fp['bbox']
                for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                    #(clip, frame, bbox, feat), (clip, frame, bbox, coord)
                    if self.dset == 'star': vid = vid.decode("utf-8")
                    self.bbox_feats[str(vid)] = (feat[:,:,:self.bbox_num, :], bbox[:,:,:self.bbox_num, :])

        if self.dset not in ['webvid', 'star']:
            if self.use_frame:
                app_feat_file = osp.join(self.video_feature_path, f'frame_feat/app_feat_{self.mode}.h5')
                print('Load {}...'.format(app_feat_file))
                self.frame_feats = {}
                with h5py.File(app_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['resnet_features']
                    print(feats.shape) #v_num, clip_num, feat_dim
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        if self.dset in ['star','causalvid']: vid = vid.decode("utf-8")
                        self.frame_feats[str(vid)] = feat


    def __len__(self):
        return len(self.data)
    
    def get_video_feature(self, video_name, width=320, height=240):
        """
        :param video_name:
        :param width:
        :param height:
        :return:
        """
        cnum = 8
        cids = list(range(cnum))
        pick_ids = cids
        
        if self.dset in ['frameqa', 'causalvid']:
            data_dir = '/raid/jbxiao/data/'
            if self.dset== 'causalvid':
                region_feat_file = osp.join(data_dir, 'causalvid/region_feat_aln/'+self.map_dir[video_name]+'.npz')
            elif self.dset == 'frameqa':
                region_feat_file = osp.join(data_dir, 'TGIF/region_feat_aln/'+video_name+'.npz')
            region_feat = np.load(region_feat_file)
            roi_feat, roi_bbox = region_feat['feat'], region_feat['bbox']
        else:
            roi_feat = self.bbox_feats[video_name][0][pick_ids]
            roi_bbox = self.bbox_feats[video_name][1][pick_ids]
        
        bbox_feat = transform_bb(roi_bbox, width, height)
        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
        
        if self.use_frame:
            temp_feat = self.frame_feats[video_name][pick_ids] #[:, pick_id,:] #.reshape(clip_num*fnum, -1)[pick_ids,:] #[:,pick_ids,:]
            app_feat = torch.from_numpy(temp_feat).type(torch.float32)
        else:
            app_feat = torch.tensor([0])
        
        # print('Sampled feat: {}'.format(region_feat.shape))
        return region_feat, app_feat

    def get_video_feat(self, video_name, width=320, height=240):
        # video_feature_path = f'../data/feats/{self.dset}/'
        video_feature_path = f'/raid/jbxiao/data/{self.dset}/'
        frame_feat_file = osp.join(video_feature_path, 'frame_feat/'+video_name+'.npy')
        if not osp.exists(frame_feat_file):
            video_feature_path = f'/raid/jbxiao/data/webvid80k/'
            frame_feat_file = osp.join(video_feature_path, 'frame_feat/'+video_name+'.npy')

        frame_feat = np.load(frame_feat_file)
        app_feat = torch.from_numpy(frame_feat).type(torch.float32)
        region_feat_file = osp.join(video_feature_path, 'bbox_feat_aln/'+video_name+'.npz')
        region_feat = np.load(region_feat_file)
        
        roi_feat, roi_bbox = region_feat['feat'], region_feat['bbox']

        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)

        bbox_feat = transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)

        return region_feat, app_feat

    def get_video_feat_star(self, video_name, qid, width=320, height=240):
        clips = self.vid_clips[qid]
        video_feature_path = f'/raid/jbxiao/data/star/'
        app_feats = []
        roi_feats, roi_bboxs = [], []
        for cid, clip in enumerate(clips):
            clip_feat, clip_rfeat, clip_rbbox = [], [], []
            for fid in clip:
                frame_feat_file = osp.join(video_feature_path, f'frame_feat/{video_name}/{fid:06d}.npy')
                frame_feat = np.load(frame_feat_file)
                clip_feat.append(frame_feat)

                region_feat_file = osp.join(video_feature_path, f'bbox/{video_name}/{fid:06d}.npz')
                region_feat = np.load(region_feat_file)
                clip_rfeat.append(region_feat['x'])
                clip_rbbox.append(region_feat['bbox'])
            app_feats.append(clip_feat)
            feats = np.asarray(clip_rfeat)
            bboxes = np.asarray(clip_rbbox)
            vid_feat_aln, vid_bbox_aln = align(feats, bboxes, video_name, cid)
            roi_feats.append(vid_feat_aln)
            roi_bboxs.append(vid_bbox_aln)
        frame_feat = np.asarray(app_feats)
        app_feats = torch.from_numpy(frame_feat).type(torch.float32)

        roi_feats = np.asarray(roi_feats)
        roi_feats = torch.from_numpy(roi_feats).type(torch.float32)

        roi_bboxs = np.asarray(roi_bboxs)
        bbox_feats = transform_bb(roi_bboxs, width, height)
        bbox_feats = torch.from_numpy(bbox_feats).type(torch.float32)

        region_feats = torch.cat((roi_feats, bbox_feats), dim=-1)

        # print(region_feats.shape, app_feats.shape)

        return region_feats, app_feats

    def __getitem__(self, index):
        
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid =  str(cur_sample['qid'])
        if 'width' not in cur_sample:
            #msrvtt
            width, height = 320, 240
        else:
            width, height = cur_sample['width'], cur_sample['height']
        if self.dset == 'webvid':
            video_o, video_f = self.get_video_feat(vid_id, width, height)
        elif self.dset == 'star':
            video_o, video_f = self.get_video_feat_star(vid_id, qid, width, height)
        else:
            video_o, video_f = self.get_video_feature(vid_id, width, height)
        
        vid_duration = video_o.shape[0]

        # video_o, video_f = torch.tensor([0]), torch.tensor([0])
        # vid_duration = 0
        
        question_txt = cur_sample['question']
            
        # print(question_txt)
        if self.mc == 0:
            #open-ended QA
            question_embd = torch.tensor(
                self.bert_tokenizer.encode(
                    question_txt,
                    add_special_tokens=True,
                    padding="longest",
                    max_length=self.qmax_words,
                    truncation=True,
                ),
                dtype=torch.long
            )
            seq_len = torch.tensor([len(question_embd)], dtype=torch.long)
        else:
            question_embd = torch.tensor([0], dtype=torch.long)
        
        qtype, ans_token_ids, answer_len = 0, 0, 0
        max_seg_num = self.amax_words
        seg_feats = torch.zeros(self.mc, max_seg_num, 2048)
        seg_num = torch.LongTensor(self.mc)

        qsn_id , qsn_token_ids, qsn_seq_len = 0, 0, 0
        qtype = 'null' if 'type' not in cur_sample  else cur_sample['type'] 
        if self.lvq and self.mode not in ['val','test']:
            
            qtype = get_qsn_type(question_txt, qtype)
            neg_num = 5
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < neg_num-1:
                valid_qsncans = self.all_questions #self.qsn_group[self.mtype]
            else:
                valid_qsncans = self.qsn_group[qtype]

            cand_qsn = valid_qsncans - set(question_txt)
            qchoices = rd.sample(list(cand_qsn), neg_num-1)
            qchoices.append(question_txt)
            rd.shuffle(qchoices)
            qsn_id = qchoices.index(question_txt)
            qsn_token_ids, qsn_tokens = tokenize(
                    qchoices,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.qmax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            qsn_seq_len = torch.tensor([len(qsn) for qsn in qsn_token_ids], dtype=torch.long)
        
        question_id = vid_id +'_'+str(cur_sample["qid"])
        if self.mc:
            if self.dset == 'causalvid':
                qtype = str(cur_sample["type"])   
                question_id = vid_id +'_'+qtype      
            
            if self.dset=='webvid': # and self.mode == 'train':
                ans = cur_sample["answer"]
                cand_answers = self.all_answers
                choices = rd.sample(cand_answers, self.mc-1)
                choices.append(ans)
                rd.shuffle(choices)
                answer_id = choices.index(ans)
                answer_txts = choices
            else:
                ans = cur_sample['answer']
                choices = [str(cur_sample["a" + str(i)]) for i in range(self.mc)]
                answer_id = choices.index(ans) if ans in choices else -1

                if self.mode not in ['val', 'test'] and rd.random() < 0.3:
                    #add randomness to negative answers
                    qtype = cur_sample['type']
                    if qtype == 'TP': qtype = 'TN'
                    qtype = get_qsn_type(question_txt, qtype) # argument 'qtype' is used to distinguish Question or Reason in CausalVid-QA
                    
                    if qtype not in self.ans_group or len(self.ans_group[qtype]) < self.mc-1:
                        valid_anscans = self.all_answers 
                    else:
                        valid_anscans = self.ans_group[qtype]
                   
                    # valid_anscans = self.all_answers
                    
                    cand_answers = valid_anscans - set(ans)
                    choices = rd.sample(list(cand_answers), self.mc-1)
                    choices.append(ans)

                    rd.shuffle(choices)
                    answer_id = choices.index(ans)
                   
                    # print(question_txt, choices, ans)
            
                answer_txts = [question_txt+f' {self.tokenizer.sep_token} '+opt for opt in choices]
                # print(answer_txts)
                # if self.dset == 'causalvid':
                    # if vid_id in self.txt_obj:
                    #     labels = set(self.txt_obj[vid_id])
                    #     for ai, qa in enumerate(answer_txts):
                    #         labs = list(labels.intersection(set(qa.split())))
                    #         cnt = 0
                    #         for i, lab in enumerate(labs):
                    #             seg_feats[ai][i] = torch.from_numpy(self.txt_obj[vid_id][lab])
                    #             cnt += 1
                    #         seg_num[ai] = cnt
        
            try:
                ans_token_ids, answer_tokens = tokenize(
                    answer_txts,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )
                # if self.dset == 'causalvid' and vid_id in self.txt_obj:
                #     for mcid, opt_tks in enumerate(answer_tks):
                #         for idx, tk in enumerate(opt_tks):
                #             if idx > 1 and tk.isdigit():
                #                 label = str(opt_tks[idx-1][1:])+tk #label should be like 'person1'
                #                 if label in self.txt_obj[vid_id]:
                #                     seg_feats[mcid][idx] = torch.from_numpy(self.txt_obj[vid_id][label])
            except:
                print('Fail to tokenize: '+answer_txts)
            seq_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long)
        else:
            answer_txts = cur_sample["answer"]
            answer_id = self.a2id.get(answer_txts, -1)  # answer_id -1 if not in top answers, that will be considered as wrong prediction during evaluation
           
        return {
            "video_id": vid_id,
            "video_o": video_o,
            "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_txt": question_txt,
            "type": qtype,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": ans_token_ids,
            "seq_len": seq_len,
            "question_id": question_id,
            "seg_feats": seg_feats,
            "seg_num": seg_num,
            "qsn_id": qsn_id,
            "qsn_token_ids": qsn_token_ids,
            "qsn_seq_len": qsn_seq_len
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, tokenizer, test_mode):
    
    if test_mode:
        test_dataset = VideoQADataset(
            csv_path=args.test_csv_path,
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:
        train_dataset = VideoQADataset(
        csv_path=args.train_csv_path,
        features=features,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        tokenizer=tokenizer,
        a2id=a2id,
        max_feats=args.max_feats,
        mc=args.mc,
        bnum =args.bnum,
        cl_loss=args.cl_loss
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
        )
        if args.dataset.split('/')[0] in ['tgifqa','tgifqa2']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            csv_path=args.val_csv_path,
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_loader = None

    return (train_loader, val_loader, test_loader)
