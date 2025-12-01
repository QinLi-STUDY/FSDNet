import torch
import torch.nn as nn
from tqdm import tqdm
from utils.misc_helper import to_device
import torch.nn.functional as F
import torch.distributed as dist
import copy
from models.hlfae import GLFA
import matplotlib.pyplot as plt
import os


class AFPM(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 structure,
                 init_bsn,
                 ):

        super(AFPM, self).__init__()
        """
        model_helper:
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()
        """

        self.inplanes=inplanes # {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        self.instrides=instrides # {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
        self.structure=structure # [{'name': 'block1', 'layers': [{'idx': 'layer1', 'planes': 256}], 'stride': 4}, {'name': 'block2', 'layers': [...], 'stride': 8}, {'name': 'block3', 'layers': [...], 'stride': 16}, {'name': 'block4', 'layers': [...], 'stride': 32}]
        self.init_bsn=init_bsn # 64

        self.indexes=nn.ParameterDict()
        
        # self.inplanes[layer['idx']]
        self.layers = []
       
        # self.layers.append(GLFA(in_channels=512))
        # self.layers.append(GLFA(in_channels=512))
        # self.layers.append(GLFA(in_channels=256))
        # print("===============")
        values = self.inplanes.values()
        values_list = list(values)
        
        # print(self.inplanes)
        for plane in values_list:
            # print("layer:",self.inplanes[layer])
            self.layers.append(GLFA(in_channels=plane))
        self.glfa_1 = GLFA(in_channels=256)
        self.glfa_2 = GLFA(in_channels=512)
        self.glfa_3 = GLFA(in_channels=512)
        self.glfa_4 = GLFA(in_channels=256)
        self.enhance_blocks = nn.ModuleList()
        
        self.enhance_blocks.append(nn.Sequential(*self.layers))
        
        for block in self.structure:
            for layer in block['layers']:
                self.indexes["{}_{}".format(block['name'],layer['idx'])]=nn.Parameter(torch.zeros(layer['planes']).long(),requires_grad=False)#block1_layer1 = nn.Parameter(torch.zeros(256))
                self.add_module("{}_{}_upsample".format(block['name'],layer['idx']),# block1_layer1_upsample = nn.UpsamplingBilinear2d(scale_factor=self.instrides['layer1']/block['stride'])
                                nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']]/block['stride']))

    @torch.no_grad()
    def forward(self, inputs,train=False):
        block_feats = {}
        feats = inputs["feats"]# layer1:[16, 256, 64, 64],layer2[16, 512, 32, 32],layer3[16, 1024, 16, 16],layer4[16, 2048, 8, 8]
        for block in self.structure:
            block_feats[block['name']]=[]

            for layer in block['layers']:
                # print(feats[layer['idx']]['feat'])
                feat_c=torch.index_select(feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'],layer['idx'])].data) # [16, 256, 64, 64] [16, 512, 32, 32]
                # print(self.indexes["{}_{}".format(block['name'],layer['idx'])].data) 
                feat_c=getattr(self,"{}_{}_upsample".format(block['name'],layer['idx']))(feat_c)# block1_layer1_upsample:[16, 256, 64, 64] [16, 512, 32, 32]
                block_feats[block['name']].append(feat_c) # [16, 256, 64, 64] [16, 512, 32, 32]
            block_feats[block['name']]=torch.cat(block_feats[block['name']],dim=1)# block_feats['block1'] = [16, 256, 64, 64] block_feats['block2'] = [16, 512, 32, 32]

        if train:
            gt_block_feats = {}
            gt_feats = inputs["gt_feats"]
            for block in self.structure:
                gt_block_feats[block['name']] = []
                for layer in block['layers']:
                    feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'], layer['idx'])].data)
                    feat_c = getattr(self, "{}_{}_upsample".format(block['name'], layer['idx']))(feat_c)
                    gt_block_feats[block['name']].append(feat_c)
                gt_block_feats[block['name']] = torch.cat(gt_block_feats[block['name']], dim=1)
            return {'block_feats':block_feats,"gt_block_feats":gt_block_feats}

        return {'block_feats':block_feats}



    def get_outplanes(self):
        return { block['name']:sum([layer['planes'] for layer in block['layers']])  for block in self.structure}

    def get_outstrides(self):
        return { block['name']:block['stride']  for block in self.structure}


    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        anomaly_types = copy.deepcopy(train_loader.dataset.anomaly_types)# {'normal': 0.5, 'sdas': 0.5}

        if 'normal' in train_loader.dataset.anomaly_types:
            del train_loader.dataset.anomaly_types['normal']# 从异常类型字典中删除正常类型的样本，确保后续计算只聚焦于其他异常

        for key in train_loader.dataset.anomaly_types:
            train_loader.dataset.anomaly_types[key] = 1.0/len(list(train_loader.dataset.anomaly_types.keys()))# 将每种异常类型的权重设置为相等的值,所有异常类型在训练过程中将被视为同等重要

        model.eval()# 确保在执行初始化过程中模型的参数不会被更新
        criterion = nn.MSELoss(reduce=False).to(model.device)
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion,distributed=distributed)# 对每个块进行特征索引
        train_loader.dataset.anomaly_types = anomaly_types # 在函数结束前，恢复原始的异常类型权重，以确保不会影响后续的计算
        model.train()


    def init_block_idxs(self,block,model,train_loader,criterion,distributed=True):
        if distributed:
            # 检查是否在分布式环境中运行，获取总进程数量（world_size）和当前进程的排名（rank）。
            world_size = dist.get_world_size() # 1
            rank = dist.get_rank() # 0
            # 如果当前进程的 rank 为 0，使用 tqdm 显示进度条；否则，直接使用简单的循环。self.init_bsn 是预定义的迭代次数，通常是批次数。
            if rank == 0:
                tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
            else:
                tq = range(self.init_bsn)
        # 如果不是在分布式环境中，使用 tqdm 直接显示进度条。
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
        # 为每层初始化一个零向量 cri_sum_vec，其大小与该层的输入通道数相同。这将用于累积每层的损失。
        cri_sum_vec=[torch.zeros(self.inplanes[layer['idx']]).to(model.device) for layer in block['layers']]# self.inplanes: layer1 256 layer2 512 layer3 1024 layer4 2048
        layer_pred_similarity = []
        # 使用 train_loader 创建一个迭代器以通过训练数据进行迭代。
        iterator = iter(train_loader)
        
        #保存
        os.makedirs("label_pred_images",exist_ok=True)
        os.makedirs("label_images",exist_ok=True)
        
        # 开始循环遍历，迭代次数为 self.init_bsn。
        for bs_i in tq:
            # 尝试从迭代器中获取下一个批次的数据。如果迭代器结束（即已经遍历完所有数据），则重置为新的迭代器并获取下一个数据批次。
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)
            # 将输入数据传输到设备（如 GPU），并获取模型的特征。在此处调用模型的 backbone 进行推理，返回包含异常特征和原始特征的字典。
            # feats{'layer1':{'feat': [16, 256, 64, 64], 'stride': 4, 'planes': 256},'layer2':{'feat': [16, 512, 32, 32], 'stride': 8, 'planes': 512}...},
            # gt_feats{'layer1':{'feat': [16, 256, 64, 64], 'stride': 4, 'planes': 256},'layer2':{'feat': [16, 512, 32, 32], 'stride': 8, 'planes': 512}...}
            bb_feats = model.backbone(to_device(input),train=True)

            ano_feats=bb_feats['feats'] # anomaly feats
            ori_feats=bb_feats['gt_feats'] # origin feats
            gt_mask = input['mask'].to(model.device)# [16, 1, 256, 256]
            # 获取当前批次的样本数量（即异常图像的数量），存储在 B 中。
            B= gt_mask.size(0) # 16
            # 根据层索引提取原始特征和异常特征，分别存储在 ori_layer_feats 和 ano_layer_feats 中。
            ori_layer_feats=[ori_feats[layer['idx']]['feat'] for layer in block['layers']]# [16, 256, 64, 64] [16, 512, 32, 32]
            ano_layer_feats=[ano_feats[layer['idx']]['feat'] for layer in block['layers']]# [16, 256, 64, 64] [16, 512, 32, 32]
            # 使用 enumerate 遍历异常特征和原始特征的对。i 是层的索引。
            # n = 0
            for i,(enhance_block,ano_layer_feat,ori_layer_feat) in enumerate(zip(self.enhance_blocks,ano_layer_feats,ori_layer_feats)):
                # 从块中获取当前层的名称。
                layer_name=block['layers'][i]['idx']# layer1 layer2
                # 获取当前层异常特征的通道数。
                C = ano_layer_feat.size(1)# 256 512
                # ano_layer_feat = self.layers[n](ano_layer_feat)
                # n = n + 1
                
                # 通过动态获取方法名（根据块名称和层名称）对异常特征和原始特征进行上采样。这通常是为了使特征匹配相同的空间尺寸以进行比较。
                ano_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ano_layer_feat)# [16, 256, 64, 64] [16, 512, 32, 32]
                # print(ano_layer_feat.shape)
                
                ori_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ori_layer_feat)# [16, 256, 64, 64] [16, 512, 32, 32]
                # 计算异常特征和原始特征之间的平方差，用于后续的损失计算。
                layer_pred = (ano_layer_feat - ori_layer_feat) ** 2 # [16, 256, 64, 64] [16, 512, 32, 32]
                

                
                
                # 获取预测差异图像的高度和宽度。
                _, _, H, W = layer_pred.size()
                # 将 layer_pred 的维度排列调整为 (C, B * H * W)，方便对通道进行后续处理。
                layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(C, B * H * W) # [256, 65536] [512, 16384]
                # 计算 layer_pred 每个通道的最小和最大值，用于后续的归一化过程
                (min_v, _), (max_v, _) = torch.min(layer_pred, dim=1), torch.max(layer_pred, dim=1)
                # 将 layer_pred 按通道进行归一化，使得每通道的值在 [0, 1] 之间。使用 1e-4 作为防止除零错误的微小值
                layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1)+ 1e-4)
                # 使用 F.interpolate 将真实掩码 gt_mask 重新调整为与 layer_pred 相同的空间尺寸
                label = F.interpolate(gt_mask, (H, W), mode='nearest') # gt_mask:[16,1,256,256] label:[16, 1, 64, 64] [16, 1, 32, 32]
                vis_label = label
                # 将 label 的维度调整为 (C, B * H * W)，使其与 layer_pred 进行比较
                label = label.permute(1, 0, 2, 3).contiguous().view(1, B * H * W).repeat(C, 1) # [256, 65536] [512, 16384]
                # 使用提供的损失函数（criterion）计算当前层的均方误差（MSE），并对通道维度进行平均
                mse_loss = torch.mean(criterion(layer_pred, label), dim=1) # 256 512
                mse_loss = mse_loss
                
                #######################################################
                # layer_pred = layer_pred.view(B,C,H,W)
                
                # for k in range(layer_pred.shape[0]):
                #     image = layer_pred[k,:,:,:]
                #     for j in range(layer_pred.shape[1]):
                #         img_path = os.path.join("label_pred_images",str(k)+"_"+str(j)+".png")
                #         img = image.cpu().detach().numpy().transpose(1,2,0)[:,:,j]
                #         plt.imsave(img_path,img,cmap='gray')
                # for k in range(vis_label.shape[0]):
                #     image = vis_label[k,:,:,:]
                #     for j in range(vis_label.shape[1]):
                #         img_path = os.path.join("label_images",str(k)+"_"+str(j)+".png")
                #         img = image.cpu().detach().numpy().transpose(1,2,0)[:,:,j]
                #         plt.imsave(img_path,img,cmap='gray')
                #####################################################
                # mse_loss = torch.mean(criterion(layer_pred_similarity, label), dim=1) # 256 512
                # 如果在分布式环境中，首先构建一个与每个进程相同的损失列表，然后使用 dist.all_gather 将每个进程的 mse_loss 收集起来，最终对所有进程的损失取平均，确保各个进程之间的损失一致
                
                # mse_loss = mse_loss * layer_pred_similarity
                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list,dim=0),dim=0,keepdim=False)
                # 将当前层的损失加到相应层的累计损失向量 cri_sum_vec 中
                cri_sum_vec[i] += mse_loss
                
                # cri_sum_vec[i] = layer_pred_similarity
        # 开始对每层的累计损失进行处理
        for i in range(len(cri_sum_vec)):
            # 将 cri_sum_vec[i] 中的 NaN 值替换为该层有效值的最大值，避免后续计算中的问题
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            
            # layer_pred_similarity[torch.isnan(layer_pred_similarity)] = torch.max(layer_pred_similarity[~torch.isnan(layer_pred_similarity)])            
            
            # 通过 torch.topk 获取当前层中损失最小的特征索引，选择的数量为该层的参数数量（planes）
            # values, indices = torch.topk(layer_pred_similarity, k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            # 对选中的特征索引进行排序
            values, _ = torch.sort(indices)
            # 如果处于分布式环境，构建一个包含所有进程当前特征索引的列表，使用 dist.all_gather 收集这些索引，并将第一个进程的值复制到相应的 self.indexes 中。
            if distributed:
                tensor_list = [values for _ in range(world_size)]
                # print(tensor_list)
                # print(tensor_list[0].long())
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            # 如果不是分布式环境，将选中的特征索引直接复制到相应的 self.indexes 中。
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())
    def calculate_similarity(self, ano_layer_feat, ori_layer_feat):
        # 归一化
        ano_min_v, _ = torch.min(ano_layer_feat, dim=1, keepdim=True)  # [256, 1]
        ano_max_v, _ = torch.max(ano_layer_feat, dim=1, keepdim=True)  # [256, 1]
        ori_min_v, _ = torch.min(ori_layer_feat, dim=1, keepdim=True)  # [256, 1]
        ori_max_v, _ = torch.max(ori_layer_feat, dim=1, keepdim=True)  # [256, 1]

        # 归一化运算
        ano_layer_feat = (ano_layer_feat - ano_min_v) / (ano_max_v - ano_min_v + 1e-4)  # [256, 66456]
        ori_layer_feat = (ori_layer_feat - ori_min_v) / (ori_max_v - ori_min_v + 1e-4)  # [256, 66456]

        # 逐元素相似度计算
        sum_ano_ori = torch.sum(ano_layer_feat * ori_layer_feat, dim=1, keepdim=True)  # [256, 1]
        sum_ano_square = torch.sum(ano_layer_feat ** 2, dim=1, keepdim=True)  # [256, 1]
        sum_ori_square = torch.sum(ori_layer_feat ** 2, dim=1, keepdim=True)  # [256, 1]

        # 计算权重，广播运算
        weight = sum_ano_ori / (torch.sqrt(sum_ano_square) * torch.sqrt(sum_ori_square) + 1e-4)  # [256, 1]
        
        # 将权重扩展到 [256, 66456]
        weight = weight.expand(-1, ano_layer_feat.size(1))  # [256, 66456]

        return weight

        
