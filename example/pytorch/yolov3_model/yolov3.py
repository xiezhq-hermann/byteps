import sys
sys.path.append("..")

import torch
import torch.nn as nn
from yolov3_model.backbones.darknet53 import Darknet53
from yolov3_model.necks.yolo_fpn import FPN_YOLOV3
from yolov3_model.head.yolo_head import Yolo_head
from yolov3_model.layers.conv_module import Convolutional

DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}

# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }


class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, init_weights=True):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(MODEL["STRIDES"])
        self.__nC = DATA["NUM"]
        self.__out_channel = MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = Darknet53()
        self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)


    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                # print("initing {}".format(m))


if __name__ == '__main__':
    net = Yolov3()
    p, p_d = net(torch.rand(12, 3, 448, 448))
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
    label_sbbox = torch.rand(12,  56, 56, 3,26)
    label_mbbox = torch.rand(12,  28, 28, 3, 26)
    label_lbbox = torch.rand(12, 14, 14, 3,26)
    sbboxes = torch.rand(12, 150, 4)
    mbboxes = torch.rand(12, 150, 4)
    lbboxes = torch.rand(12, 150, 4)
    
    from yolov3_model.loss.yolo_loss import YoloV3Loss

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(MODEL["ANCHORS"], MODEL["STRIDES"])(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
