import torch
import torch.nn as nn
import torch.nn.functional as F
from q_layers import QConvBnReLU, QLinear, QAdd, QReLU, QMaxPool2d, QAdaptiveAvgPool2d
from c_model import CResnet50
from c_layers import QParam


class QBigBlock(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, downsample=False, downsample_shortcut=False):
        super(QBigBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.midplanes = midplanes
        stride = 2 if downsample else 1
        self.conv1 = QConvBnReLU(inplanes, midplanes, (1, 1), relu=True, stride=stride, padding=0, dilation=1)
        self.conv2 = QConvBnReLU(midplanes, midplanes, (3, 3), relu=False, stride=(1, 1), padding=1, dilation=1)
        self.conv3 = QConvBnReLU(midplanes, outplanes, (1, 1), relu=False, stride=(1, 1), padding=0, dilation=1)
        self.act2 = QReLU()
        self.stride = stride
        if downsample_shortcut:
            self.downsample = QConvBnReLU(inplanes, outplanes, kernel_size=(1, 1), stride=stride, relu=False)
        else:
            self.downsample = None
        self.add = QAdd()
        self.act2 = QReLU()

        # self.conv1 = QConvBnReLU()
        # self.conv2 = QConvBnReLU(q_block.conv2, q_block.bn2)
        #
        # self.downsample = nn.Sequential(
        #     QDConvBnReLU(q_block.downsample[0], q_block.downsample[1])
        # ) if q_block.downsample else None
        # self.add = QDAdd(q_block.add)
        # self.act2 = QDReLU(q_block.act2)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x = self.add(x, shortcut)
        x = self.act2(x)
        return x

    def convert_from(self, c_biglock, q_in):
        self.conv1.convert_from(c_bigblock.conv1, q_in)
        self.conv2.convert_from(c_bigblock.conv2, c_bigblock.conv1.q_out)
        self.conv3.convert_from(c_bigblock.conv3, c_bigblock.conv2.q_out)
        if self.downsample:
            self.downsample.convert_from(c_bigblock.downsample, q_in)
            self.add.convert_from(c_bigblock.add, [c_bigblock.conv3.q_out, c_bigblock.downsample.q_out])
        else:
            self.add.convert_from(c_bigblock.add, [c_bigblock.conv3.q_out, q_in])
        self.act2.convert_from(c_bigblock.add.q_out)


class QResnet50(nn.Module):
    def __init__(self, num_class):
        super(QResnet50, self).__init__()

        self.conv1 = QConvBnReLU(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, relu=True)
        self.maxpool = QMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(QBigBlock(64, 64, 256, downsample_shortcut=True)
                                    QBigBlock(256, 64, 256)
                                    QBigBlock(256, 64, 256))
        self.layer2 = nn.Sequential(QBigBlock(256, 128, 512, downsample=True, downsample_shortcut=True)
                                    QBigBlock(512, 128, 512)
                                    QBigBlock(512, 128, 512)
                                    QBigBlock(512, 128, 512))
        self.layer3 = nn.Sequential(QBigBlock(512, 256, 1024, downsample=True, downsample_shortcut=True)
                                    QBigBlock(1024, 256, 1024)
                                    QBigBlock(1024, 256, 1024)
                                    QBigBlock(1024, 256, 1024)
                                    QBigBlock(1024, 256, 1024)
                                    QBigBlock(1024, 256, 1024))
        self.layer4 = nn.Sequential(QBigBlock(1024, 512, 2048, downsample=True, downsample_shortcut=True)
                                    QBigBlock(2048, 512, 2048)
                                    QBigBlock(2048, 512, 2048))

        self.global_pool = QAdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc = QLinear(2048, num_class)
        self.q_in = QParam(8)
        self.q_out = QParam(8)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.q_in.quantize_tensor(x)
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.q_out.dequantize_tensor(x)
        return x

    def convert_from(self, c_resnet50):
        self.q_in = c_resnet50.conv1.q_in
        self.conv1.convert_from(c_resnet50.conv1, c_resnet50.conv1.q_in)
        last_q_out = c_resnet50.conv1.q_out
        for layer, c_layer in zip([self.layer1, self.layer2, self.layer3, self.layer4],
                                  [c_resnet50.layer1, c_resnet50.layer2, c_resnet50.layer3, c_resnet50.layer4]):
            for block, c_block in zip(layer, c_layer):
                block.convert_from(c_block, last_q_out)
                last_q_out = c_block.add.q_out
        self.fc.convert_from(c_resnet50.fc, last_q_out)
        self.q_out = c_resnet50.fc.q_out


if __name__ == '__main__':
    import os

    model = CResnet50(10)
    with open('checkpoint/quantization_aware_training/c_resnet50_w.pt', 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.eval()
    q_model = QResnet50(10)
    q_model.convert_from(model)
    q_model.eval()

    os.makedirs('checkpoint/quantized_model', exist_ok=True)
    state_dict = q_model.state_dict()
    with open('checkpoint/quantized_model/q_resnet50.pt', 'wb') as f:
        torch.save(q_model.state_dict(), f)
    #
    # model2 = QResnet50(10)
    # model2.load_state_dict(state_dict)
    #
    # for key in state_dict.keys():
    #     s1, s2 = 'q_model.' + key, 'model2.' + key
    #     s1, s2 = s1.replace('.0.', '[0].'), s2.replace('.0.', '[0].')
    #     s1, s2 = s1.replace('.1.', '[1].'), s2.replace('.1.', '[1].')
    #     if not (eval(s1) == eval(s2)).all():
    #         print(s1,s2)
    #
    # # import tqdm
    # # from data import val_dl
    # # acc = []
    # # bar = tqdm.tqdm(val_dl)
    # # for x, label in bar:
    # #     y = q_model(x)
    # #     acc.extend((y.argmax(dim=1) == label).tolist())
    # #     bar.set_postfix({'acc': sum(acc) / len(acc)})
