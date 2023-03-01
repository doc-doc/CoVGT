import h5py
import os
import os.path as osp
import numpy as np
from bbox_visualizer import *
import sys
sys.path.insert(0, '../')
from util import load_file, save_to

bbox_colors = np.loadtxt('colors.txt')


def sample_clips(total_frames, num_clips, num_frames_per_clip):
    clips = []
    frames = [str(f+1).zfill(6) for f in range(total_frames)]
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1: num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        clip_start = 0 if clip_start < 0 else clip_start
        clip_end = total_frames if clip_end > total_frames else clip_end
        clip = frames[clip_start:clip_end] 
        if clip_start == 0 and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_start])
            if len(added_fids) > 0:
                clip = added_fids + clip
        if clip_end == total_frames and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_end-1])
            if len(added_fids) > 0:
                clip += added_fids
        clip = clip[::4]
        clips.append(clip)
    clips = clips[::2]
    return clips


def get_vbbox(feat_file, qvid, bbox_num):
    with h5py.File(feat_file, 'r') as fp:
        vids = fp['ids']
        bboxes = fp['bbox']
        for id, (vid, bbox) in enumerate(zip(vids, bboxes)):
            if str(vid) != qvid: continue
            vbbox = bbox[:,:,:bbox_num, :] 
    
    return vbbox


def vis_det(feat_file, vname):
    bbox_num = 5
    vid = vname.split('/')[-1]
    vbbox = get_vbbox(feat_file, vid, bbox_num)
    fids = os.listdir(vname)
    total_frames = len(fids)
    clips = sample_clips(total_frames, 16, 16)
    # clips = np.asarray(clips).reshape(-1)
    out_dir = '../demo/'
    
    for i, cids in enumerate(clips):
        for f, fid in enumerate(cids):
            img_path = osp.join(vname, fid+'.jpg')
            bboxes = vbbox[i][f]
            
            bboxes = [[int(np.round(b)) for b in bbox] for bbox in bboxes]
            # bbox = [int(np.round(b)) for b in bbox]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output = draw_multiple_rectangles(img, bboxes, bbox_colors)
            # output = draw_rectangle(img, bbox)
            
            out_file = osp.join(out_dir, str(vid))
            if not osp.exists(out_file):
                os.makedirs(out_file)
            img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(osp.join(out_file, fid+'.jpg'), img)
            # cv2.imshow('image', output)
            # cv2.waitKey(0)



def main():
    dataset = 'nextqa'
    feat_file = f'../../data/{dataset}/region_feat_n/acregion_8c20b_val.h5'
    frame_dir = '/storage/jbxiao/workspace/data/nextqa/frames/'
    vname = f'{frame_dir}/3376544720'
    vis_det(feat_file, vname)

if __name__ == "__main__":
    main()
