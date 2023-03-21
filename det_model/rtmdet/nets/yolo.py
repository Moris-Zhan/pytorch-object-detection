import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.append(parentdir)

import torch
import torch.nn as nn

from det_model.rtmdet.nets.CSPnext import CSPnext, C3, Conv, ShareConv


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成CSPnext53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80,80,256
        #   40,40,512
        #   20,20,1024
        #---------------------------------------------------#
        self.backbone   = CSPnext(base_channels, base_depth)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

       
        self.yolo_head_bbox_P3 = nn.Sequential(
                ShareConv(base_channels * 4, 
                        base_channels * 4, 3, 1, 1,
                        shared_conv = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1)),
                nn.Conv2d(base_channels * 4, 4, 1, 1, 0)       
            )
        
        self.yolo_head_cls_P3 = nn.Sequential(
                ShareConv(base_channels * 4, 
                        base_channels * 4, 3, 1, 1,
                        shared_conv = None),
                nn.Conv2d(base_channels * 4, num_classes, 1, 1, 0)       
            )

        self.yolo_head_bbox_P4 = nn.Sequential(
                ShareConv(base_channels * 8, 
                        base_channels * 8, 3, 1, 1,
                        shared_conv = nn.Conv2d(base_channels * 8, base_channels * 8, 3, 1, 1)),
                nn.Conv2d(base_channels * 8, 4, 1, 1, 0)       
            )
        
        self.yolo_head_cls_P4 = nn.Sequential(
                ShareConv(base_channels * 8, 
                        base_channels * 8, 3, 1, 1,
                        shared_conv = None),
                nn.Conv2d(base_channels * 8, num_classes, 1, 1, 0)       
            )
        
        self.yolo_head_bbox_P5 = nn.Sequential(
                ShareConv(base_channels * 16, 
                        base_channels * 16, 3, 1, 1,
                        shared_conv = nn.Conv2d(base_channels * 16, base_channels * 16, 3, 1, 1)),
                nn.Conv2d(base_channels * 16, 4, 1, 1, 0)       
            )
        
        self.yolo_head_cls_P5 = nn.Sequential(
                ShareConv(base_channels * 16, 
                        base_channels * 16, 3, 1, 1,
                        shared_conv = None),
                nn.Conv2d(base_channels * 16, num_classes, 1, 1, 0)       
            )

        pass

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)

        P5          = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4          = torch.cat([P5_upsample, feat2], 1)
        P4          = self.conv3_for_upsample1(P4)

        P4          = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3          = torch.cat([P4_upsample, feat1], 1)
        P3          = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        # out2 = self.yolo_head_P3(P3)
        out2_bbox = self.yolo_head_bbox_P3(P3)
        out2_cls = self.yolo_head_cls_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        # out1 = self.yolo_head_P4(P4)
        out1_bbox = self.yolo_head_bbox_P4(P4)
        out1_cls = self.yolo_head_cls_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        # out0 = self.yolo_head_P5(P5)
        out0_bbox = self.yolo_head_bbox_P5(P5)
        out0_cls = self.yolo_head_cls_P5(P5)

       
        
        

        

        return (out0_bbox, out0_cls), (out1_bbox, out1_cls), (out2_bbox, out2_cls)
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == '__main__' :
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask, 10, "l")
    print(model)
    rndm_input = torch.autograd.Variable(
                torch.rand(1, 3, 640, 640), 
                requires_grad = False).cpu()
    (out0_bbox, out0_cls), (out1_bbox, out1_cls), (out2_bbox, out2_cls) = model(rndm_input)
    print("-------bbox-----------")
    print(out0_bbox.shape)
    print(out1_bbox.shape)
    print(out2_bbox.shape)
    print("-------cls-----------")
    print(out0_cls.shape)
    print(out1_cls.shape)
    print(out2_cls.shape)      