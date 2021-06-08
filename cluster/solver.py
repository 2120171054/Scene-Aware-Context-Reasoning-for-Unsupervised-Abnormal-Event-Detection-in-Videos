import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm
from scipy.linalg import block_diag
from sklearn import preprocessing
import random
from shutil import copyfile
def cal_num(gamma):
    gt_num=np.zeros((10,)).astype(int)
    gt=np.argmax(gamma,1)
    for i in range (gamma.shape[0]):
        gt_num[gt[i]]+=1
    print(gt_num)
def add_noise(adj,noise_level):
    for i in range(adj.shape[0]):
        for j in range(i):
            if random.random()<=noise_level:
                adj[i,j]=1-adj[i,j]
                adj[j,i]=1-adj[j,i]
    return adj
class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader

        # Build tensorboard if use
        self.build_model()
        self.opti_loss=np.inf
        self.curr_loss=np.inf
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(self.gmm_k)
        # Optimizers
        # self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)
        self.optimizer = torch.optim.RMSprop(self.dagmm.parameters(), lr=self.lr)

        # Print networks
        self.print_network(self.dagmm, 'DaGMM')

        if torch.cuda.is_available():
            self.dagmm.cuda()
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
      """Convert a scipy sparse matrix to a torch sparse tensor."""
      sparse_mx = sparse_mx.tocoo().astype(np.float32)
      indices = torch.from_numpy(
           np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
      values = torch.from_numpy(sparse_mx.data)
      shape = torch.Size(sparse_mx.shape)
      return torch.sparse.FloatTensor(indices, values, shape)
    def normalize(self,mx):
      """Row-normalize sparse matrix"""
      rowsum = np.array(mx.sum(1))
      r_inv = np.power(rowsum, -1).flatten()
      r_inv[np.isinf(r_inv)] = 0.
      r_mat_inv = sp.diags(r_inv)
      mx = r_mat_inv.dot(mx)
      return mx
    def l2normalize(self,mx):
       # return preprocessing.normalize(mx,norm='l2')
       return mx
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        if self.mode=='train':
            self.dagmm.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}_dagmm.pth'.format(self.pretrained_model))))

            print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)

            print('loaded trained models (step: {})..!'.format(self.pretrained_model))
        else:
             self.dagmm.load_state_dict(torch.load(os.path.join(self.model_save_path, 'opti_dagmm.pth')))
             print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
             print('load test')
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        print("======================Train MODE======================")
        
        vertdb=h5py.File("../data/avenue/avenue_vert_feature.h5",'r')
        vertfeature=vertdb['vert_cls'][:]
        vertfeature=self.l2normalize(vertfeature)
        iters_per_epoch = len(self.data_loader)
        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()


        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            self.curr_loss=0
            # for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
            for i, (img_to_first_box,img_to_last_box,img_num_rois,index) in enumerate(tqdm(self.data_loader)):
                batch_szie=img_to_first_box.size()[0]
                iter_ctr += 1
                start = time.time()
                obj_num=0
                obejcts=vertfeature[img_to_first_box[0]:img_to_last_box[0]]
                # print(img_to_first_box[0],img_to_last_box[0])



                c_obj_num=img_num_rois[0]
                obj_num=obj_num+c_obj_num
                adj=np.ones((c_obj_num,c_obj_num))
                # adj[np.eye(c_obj_num,dtype=np.bool)]=0
                adj=add_noise(adj,self.noise_level)
                # print(c_obj_num,img_to_first_box[0],img_to_last_box[0])
                graph_to_first_batch=np.zeros((batch_szie,))
                graph_to_last_batch=np.zeros((batch_szie,))
                graph_to_last_batch[0]=obj_num
                for batch_index in range(1,batch_szie):
                    obejcts=np.vstack((obejcts,vertfeature[img_to_first_box[batch_index]:img_to_last_box[batch_index]]))
                    c_obj_num=img_num_rois[batch_index]
                    obj_num=obj_num+c_obj_num
                    # adj=np.ones((self.objects.shape[0],self.objects.shape[0]))
                    graph_to_first_batch[batch_index]=graph_to_last_batch[batch_index-1]+1
                    graph_to_last_batch[batch_index]=obj_num
                    adj_tmp=np.ones((c_obj_num,c_obj_num))
                    # adj_tmp[np.eye(c_obj_num,dtype=np.bool)]=0
                    adj_tmp=add_noise(adj_tmp,self.noise_level)

                    adj=block_diag(adj,adj_tmp)

                features=sp.csr_matrix(obejcts, dtype=np.float32)
                features = self.normalize(features)
                adjs=sp.csr_matrix(adj, dtype=np.float32)
                adj = self.normalize(adj)
                labels=np.zeros((batch_szie,10))
                labels[:,1]=1

                features = torch.FloatTensor(np.array(features.todense()))
                labels = torch.LongTensor(np.where(labels)[1])
                adj = self.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
                graph_to_last_batch = torch.IntTensor(graph_to_last_batch)
                features = features.cuda()
                adj = adj.cuda()
                graph_to_last_batch=graph_to_last_batch.cuda()

                # output = model(features, adj, graph_to_last_batch)
                # input_data = self.to_var(input_data)

                total_loss,sample_energy,recon_error, cov_diag, gamma= self.dagmm_step(features, adj, graph_to_last_batch)
                # Logging
                self.curr_loss+=sample_energy.cpu().detach().numpy()
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()
                # loss['gamma1'] = torch.max(gamma[-1,:]).item()
                # loss['gamma2'] = torch.argmax(gamma[-1,:]).item()

                # loss['gamma1'] = torch.max(gamma,1).item()
                # loss['gamma2'] = torch.argmax(gamma,1).item()
                # Print out log info
            if self.curr_loss<self.opti_loss and (e+1)>=10:
                print('saving opti-model--loss {}'.format(self.curr_loss))
                torch.save(self.dagmm.state_dict(),os.path.join(self.model_save_path, 'opti_dagmm.pth'))
                self.opti_loss=self.curr_loss
            if (e+1) % self.log_step == 0:
                elapsed = time.time() - start_time
                total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                
                epoch_time = str(datetime.timedelta(seconds=epoch_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                elapsed = str(datetime.timedelta(seconds=elapsed))

                lr_tmp = []
                for param_group in self.optimizer.param_groups:
                    lr_tmp.append(param_group['lr'])
                tmplr = np.squeeze(np.array(lr_tmp))

                log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                    elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                
                gamma_numpy=gamma.cpu().detach().numpy()
                cal_num(gamma_numpy)

                IPython.display.clear_output()
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                else:
                    plt_ctr = 1
                    if not hasattr(self,"loss_logs"):
                        self.loss_logs = {}
                        for loss_key in loss:
                            self.loss_logs[loss_key] = [loss[loss_key]]
                            # plt.subplot(2,2,plt_ctr)
                            # plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            # plt.legend()
                            plt_ctr += 1
                    else:
                        for loss_key in loss:
                            self.loss_logs[loss_key].append(loss[loss_key])
                            # plt.subplot(2,2,plt_ctr)
                            # plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            # plt.legend()
                            plt_ctr += 1

                    # plt.show()

                # print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
            # Save model checkpoints
            # print('epoch num {}'.format(e+1))
            if (e+1) % self.model_save_step == 0:
                print('saving model--'+os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))
                torch.save(self.dagmm.state_dict(),
                os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))



    def dagmm_step(self, input_data, adj, graph_to_last_batch):
        self.dagmm.train()
        dec, z, gamma = self.dagmm(input_data, adj, graph_to_last_batch)
        # print(dec)
        total_loss, sample_energy,recon_error, cov_diag = self.dagmm.loss_function(input_data,dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss,sample_energy,recon_error, cov_diag,gamma

    def test(self):
        print("======================TEST MODE======================")
        self.dagmm.eval()
        self.data_loader.dataset.mode="test"

        vertdb=h5py.File("../data/avenue/avenue_vert_feature.h5",'r')
        vertfeature=vertdb['vert_cls'][:]
        vertfeature=self.l2normalize(vertfeature)
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for it, (img_to_first_box,img_to_last_box,img_num_rois,index) in enumerate(self.data_loader):
            obj_num=0
            obejcts=vertfeature[img_to_first_box[0]:img_to_last_box[0]]
            batch_szie=img_to_first_box.size()[0]
            c_obj_num=img_num_rois[0]
            obj_num=obj_num+c_obj_num
            adj=np.ones((c_obj_num,c_obj_num))

            graph_to_first_batch=np.zeros((batch_szie,))
            graph_to_last_batch=np.zeros((batch_szie,))
            graph_to_last_batch[0]=obj_num
            for batch_index in range(1,batch_szie):
                obejcts=np.vstack((obejcts,vertfeature[img_to_first_box[batch_index]:img_to_last_box[batch_index]]))
                c_obj_num=img_num_rois[batch_index]
                obj_num=obj_num+c_obj_num
                graph_to_first_batch[batch_index]=graph_to_last_batch[batch_index-1]+1
                graph_to_last_batch[batch_index]=obj_num
                adj_tmp=np.ones((c_obj_num,c_obj_num))

                adj=block_diag(adj,adj_tmp)

            features=sp.csr_matrix(obejcts, dtype=np.float32)
            features = self.normalize(features)
            adjs=sp.csr_matrix(adj, dtype=np.float32)
            adj = self.normalize(adj)
            labels=np.zeros((batch_szie,10))
            labels[:,1]=1



            features = torch.FloatTensor(np.array(features.todense()))
            labels = torch.LongTensor(np.where(labels)[1])
            adj = self.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            graph_to_last_batch = torch.IntTensor(graph_to_last_batch)
            features = features.cuda()
            adj = adj.cuda()
            graph_to_last_batch=graph_to_last_batch.cuda()


            dec,z, gamma = self.dagmm(features, adj, graph_to_last_batch)
            prob=self.dagmm.compute_prob(z, gamma, phi=self.dagmm.phi, mu=self.dagmm.mu, cov=self.dagmm.cov, sample_mean=False)

            gamma_data=gamma.cpu().detach().numpy()
            if it==0:
                savedata=gamma_data
            else:
                savedata=np.concatenate((savedata,gamma_data),axis=0)


        np.set_printoptions(threshold=np.inf)

        np.save('labeldata.npy',savedata)
        gt_num=np.zeros((10,)).astype(int)
        gt=np.argmax(savedata,1)
        for i in range (savedata.shape[0]):
            gt_num[gt[i]]+=1
        print(gt_num)
        data_dis=gt_num/np.sum(gt_num)
        np.save('../data/avenue/scene_dis.npy',data_dis)
        
        if self.save_to_sgg==True:
            ### balance samples..

            ###-----------------------
            ### modify labels
            with h5py.File("../data/avenue/avenue_train_yolo_-SGG.h5",'r') as input_f:
                output_f = {}
                for key in input_f.keys():
                    if key == 'labels' or key =='predicates':
                        continue
                    else:
                        output_f[key]=input_f[key]
                img_to_first_box=input_f['img_to_first_box']
                img_to_last_box=input_f['img_to_last_box']
                
                img_to_first_rel=input_f['img_to_first_rel']
                img_to_last_rel=input_f['img_to_last_rel']
                
                raw_labels=input_f['labels']
                raw_predicates=input_f['predicates']
                update_labels=np.zeros((raw_labels.shape)).astype(int)
                update_predicates=np.zeros((raw_predicates.shape)).astype(int)
                
                for im_id in range(img_to_first_box.shape[0]):
                    for box_id in range (img_to_first_box[im_id],img_to_last_box[im_id]+1):
                        update_labels[box_id]=gt[im_id]+1
    
                for im_id in range(img_to_first_rel.shape[0]):
                    for rel_id in range (img_to_first_rel[im_id],img_to_last_rel[im_id]+1):
                        update_predicates[rel_id]=gt[im_id]+1
                ###-----------------------
                ### one-class classify..
                # update_labels=np.ones((raw_labels.shape[0],1)).astype(int)
                # update_predicates=np.ones((raw_predicates.shape[0],1)).astype(int)
                ###------------------------
                
                ### re-sampling 
                max_number_per_class=np.max(gt_num)
                max_number_label=np.argmax(gt_num)
                # sample_addition_num=np.zeros((10,)).astype(int)
                re_samples_ids=np.zeros((0,)).astype(int)
                mask_label=np.zeros((10,))
                missed_label=[]
                for label_id in range(1,11):
                    if gt_num[label_id-1]/max_number_per_class<1/4: # no enough sample
                        update_labels[update_labels==label_id]=0 #ignore
                        update_predicates[update_predicates==label_id]=0 #ignore

                    else:
                        sample_addition_num=max_number_per_class-gt_num[label_id-1]
                        re_samples_ids=np.concatenate((re_samples_ids,np.random.choice(np.where(gt==label_id-1)[0],sample_addition_num)),0)
                        
                        mask_label[label_id-1]=1
                        # for re_samples_idx in current_samples_ids:
                            
                            # update_labels=np.vstack((update_labels,update_labels[re_samples_idx]))
                            # update_predicates=np.vstack((update_predicates,update_predicates[re_samples_idx]))
                            # for key in output_f.keys():
                                # if output_f[keys].shape[0]==gt.shape[0]:
                                    # output_f[key]=np.vstack((output_f[key],output_f[key][re_samples_idx,:]))
                ### --------------------
                #for missed in missed_label:
                update_labels[update_labels==10]=2
                update_predicates[update_predicates==10]=2
                with h5py.File("../avenue_train_update_-SGG.h5",'w') as update_f:
                    for key in input_f.keys():
                        if key=='labels':
                            update_f.create_dataset(key, data=update_labels)
                        elif key=='predicates':
                            update_f.create_dataset(key, data=update_predicates)
                        else:
                            update_f.create_dataset(key, data=output_f[key])
        copyfile("../data/avenue/avenue_train_yolo_-SGG-dicts.json","../avenue_train_update_-SGG-dicts.json")
        np.save('../data/avenue/re_samples_ids.npy',re_samples_ids)
        np.save('../data/avenue/mask_label.npy',mask_label)
