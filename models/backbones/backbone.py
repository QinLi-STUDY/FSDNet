
import torch
import timm
from .feature_extraction import get_graph_node_names, create_feature_extractor
from models.hlfae import GLFA
from models.ssa import SSA
from models.essa import ESSAttn
from models.masa import MaSA
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils

class Backbone(torch.nn.Module):
    def __init__(self,
                 backbone,
                 outlayers,
                 ):

        super(Backbone, self).__init__()
        self.backbone=backbone
        self.outlayers=outlayers

        assert self.backbone in ['resnet18','resnet34','resnet50','efficientnet_b4','wide_resnet50_2']

        if self.backbone =='resnet50' or self.backbone=='wide_resnet50_2':
            layers_idx ={'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes= {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}

        elif self.backbone=='resnet34' or self.backbone=='resnet18':
            layers_idx = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}

        elif self.backbone == 'efficientnet_b4':
            # if you use efficientnet_b4 as backbone, make sure timm==0.5.x, we use 0.5.4
            layers_idx = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'layer5': 4}
            layers_strides = {'layer1': 2, 'layer2': 4, 'layer3': 8, 'layer4': 16, 'layer5': 32}
            layers_planes = {'layer1': 24, 'layer2': 32, 'layer3': 56, 'layer4': 160, 'layer5': 448}
        else:
            raise NotImplementedError("backbone must in [resnet18, resnet34,resnet50, wide_resnet50_2, efficientnet_b4]")

        self.feature_extractor = timm.create_model(self.backbone, features_only=True,pretrained=True,
                                                  out_indices=[layers_idx[outlayer] for outlayer in self.outlayers])
        self.layers_strides=layers_strides # {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
        self.layers_planes=layers_planes # {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        
        self.layers = []
       
        # self.layers.append(GLFA(in_channels=512))
        # self.layers.append(GLFA(in_channels=512))
        # self.layers.append(GLFA(in_channels=256))
        # print("===============")
        values = layers_planes.values()
        values_list = list(values)
        self.gt_layers = []
        
        # print(self.inplanes)
        for plane in values_list:
            # print("layer:",self.inplanes[layer])
            # self.layers.append(GLFA(in_channels=plane))
            self.layers.append(ESSAttn(dim=plane))
            self.gt_layers.append(ESSAttn(dim=plane))

    @torch.no_grad()
    def forward(self, inputs,train=False):
        feats_dict={}
        image = inputs["image"] # [16, 3, 256, 256]
        feats=self.feature_extractor(image)# 0:[16,256,64,64],1:[16,512,32,32],2:[16,1024,16,16],3:[16,2048,8,8]

        # feats[0] = self.layers[0](feats[0].to('cpu'))
        # for i in range(len(self.layers)):
        #     # print("feats:",feats[i].shape)
        #     feats[i] = self.layers[i](feats[i].to('cpu'))
            
        feats= {self.outlayers[idx]:{
                # "feat": feats[idx].to('cuda'),
                "feat": feats[idx],
                "stride": self.layers_strides[self.outlayers[idx]],
                "planes": self.layers_planes[self.outlayers[idx]],
                } for idx in range(len(self.outlayers))}# layer1:{'feat': [16, 256, 64, 64], 'stride': 4, 'planes': 256}

        

        feats_dict.update({"feats":feats})
        if train:
            gt_image = inputs["gt_image"] # [16, 3, 256, 256]
            
            # for k in range(gt_image.shape[0]):
            #     image = gt_image[k,:,:,:].to('cpu')
            #     img_path = os.path.join("GT_images",str(k)+".png")
            #     vutils.save_image(image,img_path)
                # img = image.detach().numpy().transpose(1,2,0)[:,:,:]
                # plt.imsave(img_path,img)
                
                    
            
            
            gt_feats = self.feature_extractor(gt_image)# 0:[16,256,64,64],1:[16,512,32,32],2:[16,1024,16,16],3:[16,2048,8,8]
            # gt_feats[0] = self.gt_layers[0](gt_feats[0].to('cpu'))
            # for i in range(len(self.layers)):
            #     # print("feats:",feats[i].shape)
            #     gt_feats[i] = self.gt_layers[i](gt_feats[i].to('cpu'))
            gt_feats = {self.outlayers[idx]:{
                        # "feat":gt_feats[idx].to('cuda'),
                        "feat":gt_feats[idx],
                        "stride": self.layers_strides[self.outlayers[idx]],
                        "planes": self.layers_planes[self.outlayers[idx]],
                } for idx in range(len(self.outlayers))}# layer1:{'feat': [16, 256, 64, 64], 'stride': 4, 'planes': 256}

            feats_dict.update({"gt_feats": gt_feats})

        return feats_dict


    def get_outplanes(self):
        return { outlayer: self.layers_planes[outlayer] for outlayer in self.outlayers}

    def get_outstrides(self):
        return { outlayer: self.layers_strides[outlayer] for outlayer in self.outlayers}
