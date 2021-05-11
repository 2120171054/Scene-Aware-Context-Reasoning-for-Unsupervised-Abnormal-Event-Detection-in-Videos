"""
Get Detection Results of the Avenue Dataset.

"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect,im_detect_vg
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import caffe, os, sys, cv2,shutil
import argparse
import h5py
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

AVENUE_DIR='./' # your avenue dataset (frames)
OUTPUT_DIR='../SC_Graph/' # main project dir
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args
def filter_bbox(bbox):
    if (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) < 100: 
        return False
    else:
        return True
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    MODE=args.mode
    if MODE=='train':
        h5_file = OUTPUT_DIR+'data/avenue/avenue_video_train_proposals.h5'
        dataset_dirs = AVENUE_DI+'/training_frames'
        CONF_THRESH = 0.5

    else:
        h5_file = OUTPUT_DIR+'data/avenue/avenue_video_test_proposals.h5'
        dataset_dirs = AVENUE_DIR+'/testing_frames'
        gt_dirs =  AVENUE_DIR+'/testing_label_mask'
        CONF_THRESH = 0.4

    f = h5py.File(h5_file, 'w')
    print('-----'+MODE+'-------')
    time.sleep(5)
    im_scales=[]; im_to_roi_idx=[]; num_rois=[]; gts=[]; im_path=[];

    rpn_rois=np.ones((1,4)); rpn_scores=np.ones((1,1));
    
    current_num=0
    image_num=0
    video_num = len(os.listdir(dataset_dirs))
    for video_idx in range(video_num):
        video_path = os.path.join(dataset_dirs,"%02d.avi"%(video_idx+1))
        frame_num = len(os.listdir(video_path))
        if MODE=='train':
            frame_gts = np.zeros((frame_num,))
        else:
            gt_mats = scio.loadmat(os.path.join(gt_dirs,"%d_label.mat"%(video_idx+1)))['volLabel']
            frame_gts = []
            assert gt_mats.shape[1] == frame_num
            for i in range(gt_mats.shape[1]):
                gt_mat=gt_mats[0,i]
                frame_gts.append(np.max(gt_mat))
            frame_gts=np.array(frame_gts)
        for frame_idx in range(frame_num):

            frame_path=os.path.join(video_path,'%d.jpg' %(frame_idx+1))
            im_path.append(frame_path)
            im = cv2.imread(frame_path)
            scale, scores, boxes = im_detect_vg(net, im)
            counter=0
            NMS_THRESH = 0.3
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]

                dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                for ini in inds:
                  bbox = dets[ini, :4]
                  if filter_bbox(bbox)==True:
                    score = dets[ini, -1]
                    rpn_rois=np.vstack((rpn_rois,bbox))
                    rpn_scores=np.vstack((rpn_scores,score))
                    counter=counter+1
            rpn_rois=np.vstack((rpn_rois,np.array([0,0,im.shape[1],im.shape[0]])))
            rpn_scores=np.vstack((rpn_scores,np.array([1])))
            counter=counter+1
            im_scales.append(scale)
            num_rois.append(counter)
            im_to_roi_idx.append(current_num)

            gts.append(frame_gts[frame_idx])
            current_num=current_num+counter
            image_num=image_num+1
            print(image_num,frame_path,counter)

    im_scales=np.array(im_scales, dtype=np.float64)
    im_to_roi_idx=np.array(im_to_roi_idx, dtype=np.int64)
    num_rois=np.array(num_rois, dtype=np.int64)
    rpn_rois=np.array(rpn_rois, dtype=np.float32)
    rpn_rois=rpn_rois[1:rpn_rois.shape[0]+1,:]
    rpn_scores=np.array(rpn_scores, dtype=np.float32)
    rpn_scores=rpn_scores[1:rpn_scores.shape[0]+1,:]
    gts=np.array(gts,dtype=np.float64)
    
    f.create_dataset('im_scales', data=im_scales)
    f.create_dataset('im_to_roi_idx', data=im_to_roi_idx)
    f.create_dataset('num_rois', data=num_rois)
    f.create_dataset('rpn_rois', data=rpn_rois)
    f.create_dataset('rpn_scores', data=rpn_scores)
    f.create_dataset('gts', data=gts)
    f.create_dataset('im_paths', data=im_path)
    f.close()