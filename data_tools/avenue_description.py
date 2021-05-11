import sys
sys.path.append("./KLTTracking")
import cv2
import h5py
import numpy as np
import os,json
import math
from KLTTracking import KLTTracking
import argparse
from tqdm import tqdm

AVENUE_DIR='./' # your avenue dataset (frames)
MAIN_DIR='../' # main project dir
object_list=[] 
for line in  open('./VG/object_list.txt'):
    object_list.append(line[:-1])
rel_list=[]
for line in  open('./VG/predicate_list.txt'):
    rel_list.append(line[:-1])
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='images/objects/relations description(JSON)')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--save_img', default=False, type=bool, help='save processed images?')
    parser.add_argument('--im_height', default=360, type=int)
    parser.add_argument('--im_width', default=640, type=int)
    parser.add_argument('--valid_threshold', default=0.0002, type=float)
    args = parser.parse_args()

    return args

def record_obj(i,object_counter,box_):
    box=box_.copy()
    box[0],box[2]=box_[0]+im_width*(i%2),box_[2]+im_width*(i%2)
    box[1],box[3]=box_[1]+im_height*int(i/2),box_[3]+im_height*int(i/2)
    return {'object_id' : int(object_counter), 'merged_object_ids':[], 'synsets' : ['person.n.01'], 'names':[object_list[object_counter%10]],'x':float(box[0]), 'y':float(box[1]), 'w':float(box[2] - box[0]),'h': float(box[3] - box[1])}
def record_rel(rel_counter,obj,sub):
    return {'predicate':rel_list[rel_counter%50],'object':obj,'subject':sub,'relationship_id':int(rel_counter),'synsets':['snear_r.01']}
def isvaliad(box,threshold):
    if float(box[2] - box[0])*float(box[3] - box[1])>threshold:
        return True
    else:
        return False
if __name__ == '__main__':
    args = parse_args()
    MODE=args.mode

    im_height=args.im_height
    im_width=args.im_width

    if MODE=='train':
        dataset_dirs = '/mnt/sdb/mnt/sda/sunche/dataset/abnormal_event/videos/avenue/training_frames'
    else:
        dataset_dirs = '/mnt/sdb/mnt/sda/sunche/dataset/abnormal_event/videos/avenue/testing_frames'
    
    f=h5py.File(MAIN_DIR+'/data/avenue/avenue_video_'+MODE+'_proposals.h5','r') # object proposals' dir
    obj_data=[]
    im_to_roi_idx,num_rois,rois,gts,im_paths=f['im_to_roi_idx'][:],f['num_rois'][:],f['rpn_rois'],f['gts'],f['im_paths']
    tim_id,image_counter,object_counter,rel_counter=1,0,0,0
    timages_meta,rels_data,objs_data=[],[],[]
    curr_images= np.empty((10,),dtype=np.ndarray)
    stop_flag=0
    for i in tqdm(range(len(im_paths))):
        if stop_flag==1:
            break;
        im_path=im_paths[i]
        idx = i%10
        # im_path=str(im_path, encoding = "utf-8")
        im_path=str(im_path)
        curr_images[idx]=cv2.imread(im_path)
        if curr_images[idx].ndim==3:
            timage=np.zeros((im_height*5,im_width*2,curr_images[idx].shape[2]))
        else:
            timage=np.zeros((im_height*5,im_width*2))
        timage[int(idx/2)*im_height:int(idx/2+1)*im_height,int(idx%2)*im_width:int(idx%2+1)*im_width]=curr_images[idx]
        if i % 10 == 0:
            objects=[]
            relationships=[]
            obj_data={'image_id' : tim_id, 'objects':objects,'image_url' : 'www'} 
            rel_data={'image_id' : tim_id, 'relationships':relationships}
            timage_meta={'image_id' : tim_id, 'url':'www','height' : int(im_height),'width':int(im_width),'coco_id':None,'flickr_id':None}
            roi_idx=im_to_roi_idx[i]
            curr_num_rois = num_rois[i]
            act_num_rois = 0
            first_boxs =[]
            for j in range(curr_num_rois): # first object filtering processing!!!
                curr_box=rois[roi_idx+j,:]
                if isvaliad(curr_box,10*im_height*im_width*args.valid_threshold):
                    act_num_rois = act_num_rois +1
                    obj=record_obj(idx,object_counter,curr_box)
                    first_boxs.append(curr_box)
                    objects.append(obj)
                    object_counter=object_counter+1
                
            rel_objs=objects[idx*act_num_rois:(idx+1)*act_num_rois] # first rels
            for j, obj in enumerate(rel_objs):
                for k,sub in enumerate(rel_objs):
                    if j<k and math.sqrt((obj['x']-sub['x'])**2+(obj['y']-sub['y'])**2)<180 and j != act_num_rois-1 and k!= act_num_rois-1:
                        rel=record_rel(rel_counter,obj,sub)
                        relationships.append(rel)
                        rel_counter=rel_counter+1
            for res_i in range (i+1,i+10): # tracking 
                res_idx=res_i%10
                if res_i>=len(im_paths):
                    print("finished")
                    stop_flag=1
                    break;
                # im_path=str(im_paths[res_i], encoding = "utf-8")
                im_path=str(im_paths[res_i])
                curr_images[res_idx]=cv2.imread(im_path)
                timage[int(res_idx/2)*im_height:int(res_idx/2+1)*im_height,int(res_idx%2)*im_width:int(res_idx%2+1)*im_width]=curr_images[res_idx]
            if stop_flag==1:
                continue;
            all_box=KLTTracking(10,act_num_rois,curr_images,rois[roi_idx:roi_idx+num_rois[i],:])
            for res_i in range (i+1,i+10):  # rest object
                res_idx=res_i%10
                for j in range(act_num_rois):
                    curr_box =np.array([all_box[res_idx][j,0,0],all_box[res_idx][j,0,1],all_box[res_idx][j,-1,0],all_box[res_idx][j,-1,1]])
                    if isvaliad(curr_box,10*im_height*im_width*args.valid_threshold):
                        obj=record_obj(res_idx,object_counter,curr_box)
                        objects.append(obj)
                        object_counter=object_counter+1                   
                    else:
                        obj=record_obj(res_idx,object_counter,first_boxs[j])
                        objects.append(obj)
                        object_counter=object_counter+1
                rel_objs=objects[res_idx*act_num_rois:(res_idx+1)*act_num_rois] # res rels      
                for j, obj in enumerate(rel_objs):
                    for k,sub in enumerate(rel_objs):
                        if math.sqrt((obj['x']-sub['x'])**2+(obj['y']-sub['y'])**2)<min(im_height,im_width):
                            rel=record_rel(rel_counter,obj,sub)
                            relationships.append(rel)
                            rel_counter=rel_counter=1
                            
                if res_i%10==9:
                    objs_data.append(obj_data)
                    rels_data.append(rel_data)
                    timages_meta.append(timage_meta)
                    if args.save_img:
                        cv2.imwrite(MAIN_DIR+'/data/avenue/'+MODE+'_video_images/'+str(tim_id)+'.jpg',timage)
                    tim_id=tim_id+1
    
        else:
            continue;
    
    with open(MAIN_DIR+'/data/avenue/avenue_video_'+MODE+'_objects.json','w') as json_file:
        json.dump(objs_data,json_file)
    with open(MAIN_DIR+'/data/avenue/avenue_video_'+MODE+'_relationships.json','w') as json_file:
        json.dump(rels_data,json_file)
    with open(MAIN_DIR+'/data/avenue/avenue_video_'+MODE+'_image_data.json','w') as json_file:
        json.dump(timages_meta,json_file)
