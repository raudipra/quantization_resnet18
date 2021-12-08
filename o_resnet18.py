import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        stride = outplanes // inplanes
        downsample = stride == 2

        self.conv1 = nn.Conv2d(
            inplanes, outplanes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(outplanes, affine=True)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = norm_layer(outplanes, affine=True)
        self.act2 = act_layer(inplace=True)
        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, (1, 1), (2, 2), bias=False),
            nn.BatchNorm2d(outplanes, affine=True)
        ) if downsample else None

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Resnet18_(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(Resnet18_, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512), BasicBlock(512, 512))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        if pretrained:
            import timm
            self.fc = nn.Linear(512, 1000)
            model_tmp = timm.create_model('resnet18', pretrained=True)
            state_dict = model_tmp.state_dict().copy()
            del model_tmp
            self.load_state_dict(state_dict)
        self.fc = nn.Linear(512, num_class)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model1 = Resnet18_(2, pretrained=True)
