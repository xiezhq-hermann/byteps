import sys
bandwidth = {}
# input nccl benchmark result
for line in sys.stdin:
    bandwidth[int(line.split()[1])] = (float(line.split()[-4]), float(line.split()[-2]))
print(bandwidth)

import torch

from torchvision import models
resnet50 = models.resnet50(num_classes=1000)
from torchaudio import models
speech = models.DeepSpeech(256)
transformer = torch.nn.Transformer(batch_first=True)
from yolov3_model.yolov3 import Yolov3
yolov3 = Yolov3()

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

latency = 0
for model in [resnet50, speech, transformer, yolov3]:
    for p in model.parameters():
        if p.requires_grad:
            p_num = p.numel()
            p_next = next_power_of_2(p_num)
            if p_num < 1024:
                latency = latency + bandwidth[p_next][0] / 1000
            else:
                latency = latency + p_num * 4 / 1024 / 1024 / 1024 / bandwidth[p_next][1] * 1000
        else:
            pass
    print("estimated latency: {}ms ".format(latency))