import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import os
import sys
import glob
import numpy as np
import random
import copy
from random import shuffle
import argparse
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'data_s3dis'))
from dataset import Data_Configs as Data_Configs
from dataset import Data_S3DIS as Data
from gicn_model import backbone_pointnet2, pmask_net , box_center_net , FocalLoss , Multi_Encoding_net , FocalLoss2  , Size_predict_net
from tqdm import tqdm
import importlib
import datetime, time
import logging
import lera
import math
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp

parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--batchsize', type=int, default= 8, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=80, help='number of epochs for training')
parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate for training')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
parser.add_argument('--model_name', type=str, default='Bo-net-revise', help='Name of model')
parser.add_argument('--test_area', type=str, default= 'Area_1', help='test areaa')
FLAGS = parser.parse_args()

theme = ''
time_now = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
LOG_DIR = os.path.join(BASE_DIR, 'experiment2', time_now + theme + '_Area_5' )
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
os.system('cp train_new.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp new_bonet.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT_test = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT_test.write(str(FLAGS)+'\n')
LOG_FOUT_train = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT_train.write(str(FLAGS)+'\n')


instance_sizes = [[0.48698184 ,0.25347686 , 0.42515151] , [0.26272924 ,0.25347686, 0.42515151] , [0.26272924 , 0.48322966 ,0.42515151] , [0.07527845,0.25347686 ,0.42515151] , [0.26272924, 0.06318584, 0.42515151],[0.48698184 , 0.06318584, 0.13261693]]



save_model_dir = os.path.join(LOG_DIR, 'checkpoints')
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

def log_string_test(out_str):
    LOG_FOUT_test.write(out_str+'\n')
    LOG_FOUT_test.flush()
    print(out_str)

def log_string_train(out_str):
    LOG_FOUT_train.write(out_str+'\n')
    LOG_FOUT_train.flush()
    print(out_str)



def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 6], but could be any shape.
    """
    diff = torch.abs(y_true - y_pred)
    less_than_one = (diff < 1.0).float()
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def Get_instance_center(batch_pc , batch_group, batch_bbvert ):
    batch_size = batch_pc.shape[0]
    num_point = batch_pc.shape[1]
    gt_mask = torch.zeros((batch_size, num_point)).cuda()
    sd1 = 0.004
    sd2 = 0.012
    for i in range(batch_size):
        pc = batch_pc[i]
        pc_group = batch_group[i]
        pc_group_unique = torch.unique(pc_group)
        pc_bbvert = batch_bbvert[i]
        pc_bbvert = pc_bbvert[:,1,:] - pc_bbvert[:,0,:]
        pc_bbvert = pc_bbvert[:,0]*pc_bbvert[:,1] * pc_bbvert[:,2]
        count = 0 

        for ins in pc_group_unique:
            sd = sd1 + pc_bbvert[count]*(sd2-sd1)
            pos = (pc_group == ins).nonzero().squeeze(-1)
            pc_instance = pc[(pc_group == ins),:]
            center = torch.mean((pc_instance) , dim = 0)[:3].unsqueeze(0)
            dist = torch.sum((pc_instance[:,0:3] -  center)**2 , dim = 1)
            new_idx = torch.topk(dist , 1 , largest=False)[1]
            new_center = pc_instance[new_idx,:3]
            new_dist = torch.sum((pc_instance[:,0:3] -  new_center)**2 , dim = 1)
            final_value = torch.exp(-(new_dist/(2*sd)))/(torch.sqrt(sd)*np.sqrt(2*math.pi))
            final_min = torch.min(final_value)
            final_max = torch.max(final_value)
            if final_max == final_min:
                final_value = final_value
            else :
                final_value = (final_value - final_min) / (final_max - final_min)
            count  = count + 1

            gt_mask[i,pos] = final_value.cuda()      

    gt_mask = torch.clamp(gt_mask, min=0.0, max=1.0)

    return gt_mask      



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    lr = max(0.0005/(2**(epoch//20)), 0.00001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Get_instance_size(batch_pc , batch_group ,  bat_bbvert , instance_size):
    batch_size = batch_pc.shape[0]
    num_point = batch_pc.shape[1]
    gt_instance_size = torch.zeros((batch_size, 20)).cuda()
    for i in range(batch_size):
        pc = batch_pc[i]
        pc_group = batch_group[i]
        pc_sem = batch_sem[i]
        pc_group_unique = torch.unique(pc_group)
        pc_bbvert = bat_bbvert[i]
        idx = -1
        for ins in pc_group_unique:
            if ins == -1: continue
            idx += 1
            pos = (pc_group == ins).nonzero().squeeze(-1)
            i_size = (pc_bbvert[ins.long(), 1 , :] - pc_bbvert[ins.long() , 0 , :])/2
            size_cal =  torch.sum(torch.abs(instance_size -  i_size) , dim = 1)
            size_idx = torch.argmin(size_cal)
            gt_instance_size[i , ins.long()] = size_idx

    return gt_instance_size


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ='0,1'

    # original areas
    test_areas =['Area_5']
    train_areas =['Area_4', 'Area_1', 'Area_6', 'Area_2', 'Area_3']

    # script
    # _train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    # train_areas = []
    # test_areas = []
    # for area in _train_areas:
    #     if area != FLAGS.test_area:
    #         train_areas.append(area)
    # test_areas.append(FLAGS.test_area)

    instance_size = torch.zeros((6,3))
    instance_sizes = np.array(instance_sizes)
    instance_sizes = torch.tensor(instance_sizes)
    
    for i in range(6):
        instance_size[i] = instance_sizes[i]
    
    instance_size = instance_size.cuda()


    dataset_path = '/home/andy0826/s3dis_dataset/Stanford3dDataset/s3dis_data_h5/'
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=8)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=FLAGS.batchsize , shuffle=True, 
                    num_workers=4)
    # val_dataloader   = torch.utils.data.DataLoader(data2, batch_size=FLAGS.batchsize , shuffle=False, 
    #                 num_workers=4)
    num_feature = 128

    
    model = backbone_pointnet2().cuda()
    model =  torch.nn.DataParallel(model, device_ids= [0,1])

    multi_Encoding_net = Multi_Encoding_net([512,256,256] , 3 ,[[64,128,256], [64,128,256], [64,128,256]]).cuda()
    multi_Encoding_net = torch.nn.DataParallel(multi_Encoding_net, device_ids= [0,1])

    pmask_net = pmask_net(num_feature).cuda()
    pmask_net = torch.nn.DataParallel(pmask_net, device_ids= [0,1])

    box_center_net = box_center_net().cuda()
    box_center_net = torch.nn.DataParallel(box_center_net, device_ids= [0,1])
    max_output_size = 64



    size_predict_net = Size_predict_net([0.25, 0.58 , 0.82] , [256,256,512] , 3 ,[[64,128,256], [64,128,256], [64,128,256]]).cuda()
    size_predict_net = torch.nn.DataParallel(size_predict_net, device_ids= [0,1])


    optim_params = [
        {'params' : model.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08 },
        {'params' : box_center_net.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08},
        {'params' : multi_Encoding_net.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08},
        {'params' : pmask_net.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08},
        # {'params' : sem_net.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08},
        {'params' : size_predict_net.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08}
    ]
    optimizer = optim.Adam(optim_params)

    for epoch in range(81):
        loss_list = []
        adjust_learning_rate(optimizer , epoch)
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            bat_pc, batch_sem , bat_ins , bat_psem_onehot, bat_bbvert, bat_pmask = data
            bat_pc , bat_ins , batch_sem , bat_bbvert, bat_pmask = bat_pc.cuda() , bat_ins.cuda() , batch_sem.cuda() , bat_bbvert.cuda(), bat_pmask.cuda()
            #extract global and local feature
            point_features , global_features  = model(bat_pc[:,:,0:3] , bat_pc[:,:,3:9].transpose(1, 2))

            #predict center using focal loss 
            bbox_center = box_center_net(global_features , point_features).squeeze(-1)
            gt_center = Get_instance_center(bat_pc , bat_ins , bat_bbvert)
            center_loss = FocalLoss(alpha=0.25, gamma=2)
            ct_loss = center_loss(bbox_center , gt_center)
            ct_loss = torch.mean(ct_loss)

            #get gt instance size 
            gt_instance_size = Get_instance_size(bat_pc , bat_ins , bat_bbvert  , instance_size)

            #pick topk heatmap value
            gt_instance_idx = torch.topk(gt_center , max_output_size , dim = 1)[1]
            c_gt_instance_idx = (gt_instance_idx>=0).nonzero()
            gt_instance_idx = torch.cat((c_gt_instance_idx[:,0].unsqueeze(-1) , gt_instance_idx.view(-1,1)) ,1) 
            gt_instance_idx = gt_instance_idx.detach()
            group_label_seed = bat_ins[gt_instance_idx[:,0] , gt_instance_idx[:,1]].view(-1,max_output_size)
            idx = (group_label_seed >= 0).nonzero()
            group_label_seed_aug = torch.cat((idx[:,0].unsqueeze(-1) , group_label_seed.long().view(-1,1)),1)
            pc_ins_bound_gt =  bat_bbvert[group_label_seed_aug[:,0] , group_label_seed_aug[:,1],:,:].view(-1,max_output_size,2,3).float()

            #get instance size, bbox and mask
            size_gt = gt_instance_size[group_label_seed_aug[:,0] , group_label_seed_aug[:,1]].view(-1,max_output_size).float()
            bbox_shape = bat_bbvert[:,:,1,:] - bat_bbvert[:,:,0,:]
            bbox_center =  torch.mean((bat_bbvert) , dim = -2)
            bbox_gt = torch.cat((bbox_center ,  bbox_shape) , dim = -1)
            bbox_gt = bbox_gt[group_label_seed_aug[:,0] , group_label_seed_aug[:,1],:].view(-1,max_output_size , 6)
            mask_gt = bat_pmask[group_label_seed_aug[:,0] , group_label_seed_aug[:,1],:].view(-1,max_output_size , 4096)


            sidx = gt_instance_idx[:,1].view(-1 , max_output_size).detach()
            size = size_predict_net(bat_pc[:,:,:3] , bat_pc[:,:,3:9] , global_features ,sidx , 0)
            radius = instance_size[size_gt.long()]
            radius_size = torch.sqrt(torch.sum(radius**2 , dim = -1)).unsqueeze(-1)
            pre_bbox = multi_Encoding_net(bat_pc[:,:,:3] , bat_pc[:,:,3:9] , global_features ,sidx , 0 , radius_size*1.5)
            pre_box_center =  torch.mean((pre_bbox) , dim = -2)
            pre_box_len = pre_bbox[:,:,1,:] - pre_bbox[:,:,0,:]
            pre_box = torch.cat((pre_box_center , pre_box_len) ,dim = -1)

            #smooth l1 for 6 max min
            box_bound_loss = smooth_l1_loss(pc_ins_bound_gt , pre_bbox)
            box_bound_loss = torch.mean(box_bound_loss)*40

            #radius loss             
            RL = nn.CrossEntropyLoss(reduction='none')
            radius_loss = RL(size.transpose(1,2) , size_gt.long())
            radius_loss = torch.mean(radius_loss)*2
            
            #GIOU loss
            proposal_volume = pre_box[:,:,3]*pre_box[:,:,4]*pre_box[:,:,5]
            gt_boxes_volume = bbox_gt[:,:,3]*bbox_gt[:,:,4]*bbox_gt[:,:,5]
            vA = torch.max(pre_box[:,:,:3] - pre_box[:,:,3:]/2 ,bbox_gt[:,:,:3] - bbox_gt[:,:,3:]/2)
            vB = torch.min(pre_box[:,:,:3] + pre_box[:,:,3:]/2 ,bbox_gt[:,:,:3] + bbox_gt[:,:,3:]/2)
            vA2 = torch.max(pre_box[:,:,:3] + pre_box[:,:,3:]/2 ,bbox_gt[:,:,:3] + bbox_gt[:,:,3:]/2)
            vB2 = torch.min(pre_box[:,:,:3] - pre_box[:,:,3:]/2 ,bbox_gt[:,:,:3] - bbox_gt[:,:,3:]/2)
            constant = torch.zeros(1,1).cuda()
            intersection_cube =torch.max(vB-vA , constant)
            include_box = torch.max(vA2-vB2 , constant)
            intersection_volume = intersection_cube[:,:,0]*intersection_cube[:,:,1]*intersection_cube[:,:,2]
            include_volumn =  include_box[:,:,0]*include_box[:,:,1]*include_box[:,:,2]
            ious =  intersection_volume / (proposal_volume+gt_boxes_volume-intersection_volume)
            giou = ious - ((include_volumn - (proposal_volume+gt_boxes_volume-intersection_volume))/(include_volumn+1e-8))
            ious_loss = 1-giou
            ious_loss = torch.mean(ious_loss)

            #predict mask
            pre_mask =  pmask_net(point_features , global_features , pre_bbox)
            mask_loss = FocalLoss2(alpha=0.25, gamma=2 , reduce=True)
            ms_loss = mask_loss(pre_mask ,mask_gt)



            total_loss =  ms_loss + ct_loss + radius_loss + box_bound_loss + ious_loss 
            total_loss.backward()
            optimizer.step()
            loss_list.append([total_loss.item(), ct_loss.item(), radius_loss.item() ,box_bound_loss.item() ,ious_loss.item() , ms_loss.item()])
            if i % 200 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Epoch %3d Iteration %3d (train)" % (epoch, i))
                print("%.3f %.3f %.3f %.3f %.3f %.3f" % (total_loss.item(), ct_loss.item(), radius_loss.item() ,box_bound_loss.item() , ious_loss.item() , ms_loss.item()))
                print('')
            
        loss_list_final = np.mean(loss_list, axis=0)
        log_string_train(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        log_string_train("Epoch %3d Iteration %3d (train)" % (epoch, i))
        log_string_train("%.3f %.3f %.3f %.3f %.3f %.3f" % (loss_list_final[0], loss_list_final[1], loss_list_final[2] , loss_list_final[3] , loss_list_final[4], loss_list_final[5]))
        log_string_train('')

        if epoch % 5 == 0:
                torch.save(model.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'model', epoch))
                torch.save(box_center_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'box_center_net', epoch))
                torch.save(multi_Encoding_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'multi_Encoding_net_model', epoch))
                torch.save(pmask_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'pmask_net_model', epoch))
                torch.save(size_predict_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'size_net', epoch))

        