import os
import shutil
import sys
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'Pointnet2.PyTorch'))


from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG , PointnetSAModule
import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils

class backbone_pointnet2(nn.Module):
    def __init__(self):
        super(backbone_pointnet2, self).__init__()
        self.sa1 = PointnetSAModule(mlp=[6,32,32,64] , npoint=1024, radius= 0.1 , nsample= 32 , bn=True)
        self.sa2 = PointnetSAModule(mlp=[64,64,64,128] , npoint=256, radius= 0.2 , nsample= 64 , bn=True)
        self.sa3 = PointnetSAModule(mlp=[128,128,128,256] , npoint=64, radius= 0.4 , nsample= 128 , bn=True)
        self.sa4 = PointnetSAModule(mlp=[256,256,256,512] , npoint=None, radius= None , nsample= None , bn=True)
        self.fp4 = PointnetFPModule(mlp = [768,256,256])
        self.fp3 = PointnetFPModule(mlp = [384,256,256])
        self.fp2 = PointnetFPModule(mlp = [320,256,128])
        self.fp1 = PointnetFPModule(mlp = [137,128,128,128,128])

    def forward(self, xyz , points):
        
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points  = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points  = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points  = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points  = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1,2) ,points), dim = 1), l1_points)

        global_features = l4_points.view(-1,512)
        point_features = l0_points.transpose(1,2)
        
        return  point_features , global_features 




class box_center_net(nn.Module):
    def __init__(self):
        super(box_center_net, self).__init__()
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256, 1)
        self.bn1= nn.BatchNorm1d(512)
        self.bn2= nn.BatchNorm1d(256)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, global_features , point_features):
        
        p_num = point_features.shape[1]
        global_features = global_features.unsqueeze(1)
        global_features = global_features.repeat(1,p_num,1)
        all_feature = torch.cat((global_features ,point_features) , dim = -1)

        b1 = F.leaky_relu(self.fc1(all_feature) , negative_slope = 0.2)
        b2 = F.leaky_relu(self.fc2(b1) , negative_slope = 0.2)

        #sub_branch 1
        b3 = F.leaky_relu(self.fc3(b2) , negative_slope = 0.2)
        bb_center = self.sigmoid(self.fc3_2(b3))
    
        return bb_center



class Multi_Encoding_net(nn.Module):
    def __init__(self, nsample_list, in_channel, mlp_list):
        super(Multi_Encoding_net, self).__init__()
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.conv1 = nn.Conv1d(768 + 512 , 512 , 1)
        self.conv2 = nn.Conv1d(512 , 128 , 1)
        self.conv3 = nn.Conv1d(128, 6 , 1)
        self.conv4 = nn.Conv1d(256, 1 , 1)
        self.bn1= nn.BatchNorm1d(512)
        self.bn2= nn.BatchNorm1d(128)
        self.sigmoid = nn.Sigmoid()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 6
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points , global_features , select_idx ,use_sample , radius_size):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, N, C = xyz.shape

        if use_sample == 1:
            num_points = select_idx.shape[1]
            new_xyz =  torch.from_numpy(select_idx).transpose(1,2).float().cuda()
            
        else:
            new_xyz = pointnet2_utils.gather_operation(xyz.permute(0,2,1).contiguous(), select_idx.int())
            num_points = select_idx.shape[-1]
        
        new_points_list = []


        for i in range(3):
            
            K = self.nsample_list[i]
            query_and_group =  pointnet2_utils.QueryAndGroup2(K, use_xyz=True)
            grouped_points = query_and_group(radius_size, xyz.contiguous(), new_xyz.permute(0,2,1).contiguous() , points.permute(0,2,1).contiguous())
            radius_size /= 1.5
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.leaky_relu(bn(conv(grouped_points)) , negative_slope=0.2)
            new_points = torch.max(grouped_points, 3)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        global_features = global_features.unsqueeze(-1)
        global_features = global_features.repeat(1,1,num_points)
        new_points_concat  = torch.cat((new_points_concat ,global_features) , dim = 1)
        new_points_concat = F.leaky_relu(self.conv1(new_points_concat) , negative_slope=0.2)
        num_point = new_points_concat.shape[2]
        new_points_concat = F.leaky_relu(self.conv2(new_points_concat) , negative_slope= 0.2)
        new_points_concat = self.conv3(new_points_concat).transpose(1,2).view(-1,num_point,2,3)
        new_points_concat_1 = torch.min((new_points_concat) , dim = -2)[0].unsqueeze(-2)
        new_points_concat_2 = torch.max((new_points_concat) , dim = -2)[0].unsqueeze(-2)
        new_points_concat = torch.cat((new_points_concat_1 , new_points_concat_2) , dim = -2)

        return new_points_concat

        
class pmask_net(nn.Module):
    def __init__(self , p_f_num):
        super(pmask_net , self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.conv1 = nn.Conv2d(1, 256, (1, p_f_num))
        self.conv2 = nn.Conv2d(512, 128, (1,1))
        self.conv3 = nn.Conv2d(128, 128, (1,1))
        self.conv4 = nn.Conv2d(1, 64, (1,134))
        self.conv5 = nn.Conv2d(64, 32, (1,1))
        self.conv6 = nn.Conv2d(32, 1, (1,1))
        self.bn1= nn.BatchNorm2d(128)
        self.bn2= nn.BatchNorm2d(128)
        self.bn3= nn.BatchNorm2d(64)
        self.bn4= nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_features, global_features, bbox):
        p_num = point_features.shape[1]
        num_box = bbox.shape[1]
        global_features = F.leaky_relu(self.fc1(global_features), negative_slope = 0.2).unsqueeze(1).unsqueeze(1).repeat(1,p_num  ,1 ,1)
        point_features = F.leaky_relu(self.conv1(point_features.unsqueeze(-1).permute(0,3,1,2)), negative_slope = 0.2)
        point_features  = torch.cat((point_features , global_features.permute(0,3,1,2)) , dim = 1)
        point_features = F.leaky_relu(self.conv2(point_features) , negative_slope = 0.2)
        point_features = F.leaky_relu(self.conv3(point_features) , negative_slope = 0.2)
        point_features = point_features.squeeze(-1)
 

        bbox_info = bbox.view(-1 , num_box , 6).unsqueeze(-2).repeat(1,1 ,p_num ,1)
        pmask0 = point_features.transpose(1,2).unsqueeze(1).repeat(1,num_box,1,1)
        pmask0 = torch.cat((pmask0 , bbox_info) , dim = -1)
        pmask0 = pmask0.view(-1, p_num , pmask0.shape[-1] ,1)
        pmask1 = F.leaky_relu(self.conv4(pmask0.permute(0,3,1,2)), negative_slope = 0.2)
        pmask2 = F.leaky_relu(self.conv5(pmask1), negative_slope = 0.2)
        pmask3 = self.conv6(pmask2).permute(0,2,3,1)
        pmask3 = pmask3.view(-1 , num_box , p_num)

        pred_mask = self.sigmoid(pmask3)


        return pred_mask




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=False , weight=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight

    def forward(self, inputs, targets):

        F_loss = -(targets >= 0.4).float() *self.alpha*((1.-inputs)**self.gamma)*torch.log(inputs+1e-8)\
                        -(1.-(targets >= 0.4).float())*(1.-self.alpha)*(inputs** self.gamma)*torch.log(1.-inputs+1e-8)

        if self.reduce:
            return F_loss*60
        else:
            return F_loss*60     

       


class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=False):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):

        F_loss = -targets*self.alpha*((1.-inputs)**self.gamma)*torch.log(inputs+1e-8)\
                               -(1.-targets)*(1.-self.alpha)*(inputs** self.gamma)*torch.log(1.-inputs+1e-8)

        if self.reduce:
            return torch.mean(F_loss)*60
        else:
            return F_loss*60



# class sem(nn.Module):
#     def __init__(self):
#         super(sem , self).__init__()
#         self.fc1 = nn.Linear(512, 256)
#         self.conv1 = nn.Conv2d(128, 128, (1, 1))
#         self.conv2 = nn.Conv2d(128, 64, (1,1))
#         self.conv3 = nn.Conv2d(64, 13 , (1,1))
#         self.drop1 = nn.Dropout(0.5)
#         self.bn1= nn.BatchNorm2d(128)
#         self.bn2= nn.BatchNorm2d(64)

#     def forward(self, point_features):

#         features = point_features.unsqueeze(-2)
#         features = features.permute(0,3,1,2)
#         features = F.leaky_relu(self.conv1(features), negative_slope = 0.2)
#         features = F.leaky_relu(self.conv2(features), negative_slope = 0.2)
#         # features = self.drop1(features)
#         features = self.conv3(features)
#         features = features.squeeze(-1)
#         features = features.transpose(1,2)

#         return features


class Size_predict_net(nn.Module):
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list):
        super(Size_predict_net, self).__init__()
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.conv1 = nn.Conv1d(768 , 512 , 1)
        self.conv2 = nn.Conv1d(512 , 128 , 1)
        self.conv3 = nn.Conv1d(128, 6 , 1)
        self.bn1= nn.BatchNorm1d(512)
        self.bn2= nn.BatchNorm1d(128)

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 6
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points , global_features , select_idx ,use_sample):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, N, C = xyz.shape

        if use_sample == 1:
            num_points = select_idx.shape[1]
            new_xyz =  torch.from_numpy(select_idx).transpose(1,2).float().cuda()
            
        else:
            new_xyz = pointnet2_utils.gather_operation(xyz.permute(0,2,1).contiguous(), select_idx.int())
            num_points = select_idx.shape[-1]

        new_radius_list = []
        
        for i, radius in enumerate(self.radius_list):

            K = self.nsample_list[i]
            new_radius = torch.zeros([select_idx.shape[0] , select_idx.shape[1]])
            query_and_group =  pointnet2_utils.QueryAndGroup(radius, K, use_xyz=True)
            grouped_points = query_and_group(xyz.contiguous(), new_xyz.permute(0,2,1).contiguous() , points.permute(0,2,1).contiguous())

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.leaky_relu(bn(conv(grouped_points)) , negative_slope=0.2)
            new_radius = torch.max(grouped_points, 3)[0]  # [B, D', S]
            new_radius_list.append(new_radius)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_radius_concat = torch.cat(new_radius_list, dim=1)
        new_radius_concat = F.leaky_relu(self.conv1(new_radius_concat) , negative_slope=0.2)
        num_point = new_radius_concat.shape[2]
        new_radius_concat = F.leaky_relu(self.conv2(new_radius_concat) , negative_slope= 0.2)
        new_radius_concat = self.conv3(new_radius_concat).transpose(1,2).view(-1,num_point,6)



        return  new_radius_concat