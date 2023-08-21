import torch
import torch.nn as nn

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.dw_bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dw_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.dw_bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dw_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.dw_bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dw_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dw_conv5 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1 ,groups=256)
        self.dw_bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256 , 256 ,kernel_size=1,stride=1 )
        self.bn6 = nn.BatchNorm2d(256 )
        self.dw_conv6 =nn.Conv2d (256 , 256 ,kernel_size=(3 ,3),stride=(2 ,2),padding=(0 ,0),groups