import argparse, os
import h5py
from scipy.misc import imresize
import skvideo.io as sio
from PIL import Image
import cv2
import json
import torch
from torch import nn
import torchvision
import random
import numpy as np
import shutil
import subprocess
from models import resnext
from datautils import utils
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
import os.path as osp
import sys
sys.path.insert(0, '../')
import time
from util import load_file, save_to


def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    
    model.cuda()
    model.eval()
    return model


def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('../../data/pretrained/resnext-101-kinetics.pth') 
    # download from https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M
    model_data = torch.load('../../data/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model

def extract_frame(video, dst):
    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(dst):
            # print(" cleanup: "+dst+"/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video2frame_cmd = [
            "ffmpeg",
            '-y',
            '-i', video,
            '-r', "10",
            # '-vf', "scale=400:300",
            '-vsync', '0',
            '-qscale:v', "2",
            '{0}/%06d.jpg'.format(dst)
        ]
        subprocess.call(video2frame_cmd, stdout = ffmpeg_log, stderr=ffmpeg_log)


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats


def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    
    clips = list()
    t1 = time.time()
    frame_list = sorted(os.listdir(path))
    video_data = [np.asarray(Image.open(osp.join(path, img))) for img in frame_list]
    
    valid = True
    video_data = np.asarray(video_data)
    t2 = time.time()
    print(t2-t1)

    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]

        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        

        # new_clip = clip #.transpose(0, 3, 1, 2)[None]
        # if clip.shape[0] < num_frames_per_clip:
        clip = clip[::4] #sample 4 frames per clip
        new_clip = []
        # for j in range(num_frames_per_clip):
            # if j >= len(clip):
            #     new_clip.append(new_clip[-1])
            # else:
                # new_clip.append(clip[j])
        for frame_data in clip:
            # frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = imresize(img, img_size, interp='bicubic')
            frame_data = np.array(img)
            frame_data = frame_data.transpose(2, 0, 1)[None]
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        # print(new_clip.shape)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    
    clips = clips[::4] # sample 8 clips per video
    t3 = time.time()
    
    return clips, valid

def extract_clip_frames(vpath, clips):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    # para_dict = {'r':'10', 'vsync':'0', 'qscale:v':'2'}
    # print(vpath)
    # rate = 10
    # meta = skvideo.io.ffprobe(vpath)
    # fp = meta['video']['@avg_frame_rate']
    # tstamp = int(fp.split('/')[0])//rate
    try:
        video_data = sio.vread(vpath) #ffmpeg as backend
    except:
        return None
    # video_data = video_data[::tstamp]
    total_frames, width, height, channel = video_data.shape
    # print(video_data.shape)
    img_size = (224, 224) #(args.image_height, args.image_width)
    img_clip = []
    num_clip = 8
    clips = clips[:8]
    for i, cids in enumerate(clips):
        # if i > 7: break
        fids = [int(r) for r in cids]
        # print(fids, video_data.shape)
        if fids[-1] >= total_frames:
            fids[-1] = total_frames -1
        clip = video_data[fids]
        new_clip = []
        for j in range(4):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            new_clip.append(frame_data)
        # new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        img_clip.extend(new_clip)
    
    return img_clip


def generate_npy(model, video_dir, clip_file, outfile):

    vclips = load_file(clip_file) 
    vclips = sorted(vclips.items(), key=lambda a:a[0])
    dataset_size = len(vclips)
    print(dataset_size)

    i0 = 0
    _t = {'misc': utils.Timer()}
    for i, (vname, clip) in enumerate(vclips):
        #if i <= 4000: continue
        #if i > 10000: break
        out_file = osp.join(outfile, vname+'.npy')
        if osp.exists(out_file): 
            continue
        video_path = osp.join(video_dir, vname+'.mp4')
        if not osp.exists(video_path):
            # print(video_path)
            continue
        clips = extract_clip_frames(video_path, clip)
        if clips == None: continue
        clips = np.asarray(clips)
        clip_feat = run_batch(clips, model)
        clip_feat = clip_feat.squeeze()#(32, 2048)
        
        feat = clip_feat.reshape(8, 4, 2048)
        dirname = osp.dirname(out_file)
        if not osp.exists(dirname):
            os.makedirs(dirname)
        np.save(out_file, feat)
        if i % 200 == 0:
            print(f'{i}/{dataset_size}')

def prepare_inputs(path, frame_list):
    video_data = [np.asarray(Image.open(osp.join(path, img))) for img in frame_list]
    video_data = np.asarray(video_data)
    total_frames = video_data.shape[0]
    img_size = (224, 224)
    video_inputs = []
    for j in range(total_frames):
        frame_data = video_data[j]
        img = Image.fromarray(frame_data)
        img = imresize(img, img_size, interp='bicubic')
        img = img.transpose(2, 0, 1)[None]
        frame_data = np.array(img)
        video_inputs.append(frame_data)
    video_inputs = np.asarray(video_inputs)
    # print(video_inputs.shape)
    return video_inputs

def generate_npy_byframe(model, video_list_file, video_dir, out_dir):
    videos = load_file(video_list_file)
    vnum = len(videos)
    for iv, vname in enumerate(videos):
        # if iv <= 2400: continue
        # if iv > 3000: break
        fpath = f'{video_dir}/{vname}'
        frames = sorted(os.listdir(fpath))
        out_path = osp.join(out_dir, vname)
        if osp.exists(out_path): continue
        videos = prepare_inputs(fpath, frames)
        fnum = videos.shape[0]
        if fnum > 100:
            it = fnum//100
            left = fnum % 100
            video_feats = []
            for i in range(it):
                data = run_batch(videos[i*100:100*(i+1)], model)
                video_feats.append(data)
            if left > 0:
                data = run_batch(videos[i*100:(i*100)+left], model)
                video_feats.append(data)
            # print(len(video_feats))
            video_feats = np.concatenate(video_feats, 0)
            assert video_feats.shape[0] == fnum, 'error'
        else:
            video_feats = run_batch(videos, model)
        video_feats = video_feats.squeeze()
        if not osp.exists(out_path): 
            os.makedirs(out_path)
        for iff, frame in enumerate(frames):
            fname = frame.split('.')[0]
            fpath_out = f'{out_path}/{fname}'
            # if osp.exists(fpath_out+'.npy'): continue
            np.save(fpath_out, video_feats[iff])
        if iv % 100 == 0:
            print(f'{iv}/{vnum}')
        

def generate_h5(model, v_path, v_file, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if args.dataset == "tgif-qa":
        if not os.path.exists('dataset/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('dataset/tgif-qa/{}'.format(args.question_type))
    else:
        if not os.path.exists(args.dataset):
            os.makedirs(args.dataset)
    
    vlist = load_file(v_file)
    dataset_size = len(vlist) 
    print(dataset_size)
    vnames = []
    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        for i in range(0, dataset_size):
            # if i < 20: continue
            _t['misc'].tic()
            
            video_path = osp.join(v_path, str(vlist[i]))
            
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=16)
            
            nclip, nframe = 8, 4
            if args.feature_type == 'appearance':
                clip_feat = []
                if valid:
                    # for clip_id, clip in enumerate(clips):
                    #     feats = run_batch(clip, model)  # (16, 2048)
                    #     feats = feats.squeeze()
                    #     clip_feat.append(feats)
                    # t4 = time.time()
                    clips = np.asarray(clips).squeeze()
                    clips = clips.reshape(clips.shape[0]*clips.shape[1], clips.shape[2],clips.shape[3],clips.shape[4])
                    
                    clips = torch.FloatTensor(clips).cuda().squeeze()
                    # print(clips.shape)
                    clip_feat = model(clips).squeeze()
                    # print(clip_feat.shape)
                    clip_feat = clip_feat.view(nclip, nframe, -1).detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(nclip, nframe, 2048))
                
                if feat_dset is None:
                    print(clip_feat.shape)
                    C, F, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
            
            elif args.feature_type == 'motion':
                if valid:
                    clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                    clip_feat = model(clip_torch)  # (8, 2048)
                    clip_feat = clip_feat.squeeze()
                    clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(nclip, 2048))
                if feat_dset is None:
                    print(clip_feat.shape)
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            
            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            video_ids_dset[i0:i1] = int(vlist[i])
            i0 = i1
            _t['misc'].toc()

            if (i % 100 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                        .format(i1, dataset_size, _t['misc'].average_time,
                                _t['misc'].average_time * (dataset_size - i1) / 3600))
        
        varry = np.array(vlist, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        fd.create_dataset('ids', data=varry, dtype=string_dt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='nextqa', choices=['tgif-qa', 'msvd', 'star', 'msrvtt', 'nextqa','webvid', 'causalvid'], type=str)
    parser.add_argument('--question_type', default='none', choices=['frameqa', 'count', 'transition', 'action', 'none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="../../data/nextqa/feat_{}.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=32, type=int)
    parser.add_argument('--image_height', default=112*2, type=int)
    parser.add_argument('--image_width', default=112*2, type=int)

    # network params
    parser.add_argument('--model', default='resnet101', choices=['resnet101', 'resnext101'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    if args.model == 'resnet101':
        args.feature_type = 'appearance'
    elif args.model == 'resnext101':
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')
    # set gpu
    if args.model != 'resnext101':
        torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # annotation files
    if args.dataset == 'tgifqa':
        args.annotation_file = '/storage_fast/jbxiao/workspace/VideoQA/data/{args.dataset}/videos.json'
        args.video_dir = '/raid/jbxiao/data/tgifqa/frames/'
        args.outfile = '../../data/{}/{}/{}_{}_{}_feat.h5'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.question_type, args.dataset, args.question_type, args.feature_type))
    
    elif args.dataset == 'webvid':
        args.video_dir = '/raid/jbxiao/data/WebVid/videos/'
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        clip_file = f'/storage_fast/jbxiao/workspace/VideoQA/data/datasets/webvid/val_clip.json'
        generate_npy(model, args.video_dir, clip_file, args.outfile)
        

    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'
        args.video_dir = '/ceph-g/lethao/datasets/msvd/MSVD-QA/video/'
        args.video_name_mapping = '/ceph-g/lethao/datasets/msvd/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

    elif args.dataset == 'nextqa':
        args.video_list_file = '../datasets/nextqa/vlist.json' #obtained from train/val/test csv files
        args.video_dir = '/storage/jbxiao/workspace/data/nextqa/frames/' #extacted video frames, refer to extract_video.py
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
            args.image_height = 112
            args.image_width = 112
        generate_h5(model, args.video_dir, args.video_list_file, args.num_clips, args.outfile.format(args.feature_type))
