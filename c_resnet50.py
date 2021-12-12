import torch
import torch.nn as nn
import torch.nn.functional as F
from c_layers import CConvBNReLU2d, CLinear, CAdd


class CBigBlock(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, downsample=False, layer_name=""):
        super(CBigBlock, self).__init__()
        self.inplanes = inplanes
        self.midplanes = midplanes
        self.outplanes = outplanes
        stride = 2 if downsample else 1
        self.layer_name = layer_name

        state_dict_names1 = [layer_name + '.' + name for name in
                             ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                              'bn1.num_batches_tracked']]
        self.conv1 = CConvBNReLU2d(inplanes, midplanes, (1, 1), stride, padding=0, bias=False, dilation=1, affine=True,
                                   relu=True, state_dict_names=state_dict_names1)
        state_dict_names2 = [layer_name + '.' + name for name in
                             ['conv2.weight', "", 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
                              'bn2.num_batches_tracked']]
        self.conv2 = CConvBNReLU2d(midplanes, midplanes, (3, 3), (1, 1), padding=1, bias=False, dilation=1, affine=True,
                                   relu=False, state_dict_names=state_dict_names2)
        state_dict_names3 = [layer_name + '.' + name for name in
                             ['conv3.weight', "", 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var',
                              'bn3.num_batches_tracked']]
        self.conv3 = CConvBNReLU2d(midplanes, outplanes, (1, 1), (1, 1), padding=0, bias=False, dilation=1, affine=True,
                                   relu=False, state_dict_names=state_dict_names2)
        self.act2 = nn.ReLU(inplace=True)
        self.stride = stride
        if downsample:
            state_dict_names_d = [layer_name + '.' + name for name in
                                  ['downsample.0.weight', "", 'downsample.1.weight', 'downsample.1.bias',
                                   'downsample.1.running_mean', 'downsample.1.running_var',
                                   'downsample.1.num_batches_tracked']]
            self.downsample = CConvBNReLU2d(inplanes, outplanes, kernel_size=(1, 1), stride=(2, 2), bias=False,
                                            affine=True, relu=False, state_dict_names=state_dict_names_d)
        else:
            self.downsample = None
        self.add = CAdd()
        self.act2 = nn.ReLU()

    def load_pretrained(self, state_dict):
        self.conv1.load_pretrained(state_dict)
        self.conv2.load_pretrained(state_dict)
        self.conv3.load_pretrained(state_dict)
        if self.downsample:
            self.downsample.load_pretrained(state_dict)

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

    def quantize(self, if_quantize):
        self.conv1.quantize(if_quantize)
        self.conv2.quantize(if_quantize)
        self.conv3.quantize(if_quantize)
        self.add.quantize(if_quantize)
        if self.downsample:
            self.downsample.quantize(if_quantize)


class CResnet50(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(CResnet50, self).__init__()
        state_dict_names = ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                            'bn1.num_batches_tracked']
        self.conv1 = CConvBNReLU2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False, start=True,
                                   affine=True, relu=True, state_dict_names=state_dict_names)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = nn.Sequential(CBigBlock(64, 64, 256, layer_name='layer1.0'),
                                    CBigBlock(256, 64, 256, layer_name='layer1.1'),
                                    CBigBlock(256, 64, 256, layer_name='layer1.2'))
        self.layer2 = nn.Sequential(CBigBlock(256, 128, 512, downsample=True, layer_name='layer2.0'),
                                    CBigBlock(512, 128, 512, layer_name='layer2.1'),
                                    CBigBlock(512, 128, 512, layer_name='layer2.2'),
                                    CBigBlock(512, 128, 512, layer_name='layer2.3'))
        self.layer3 = nn.Sequential(CBigBlock(512, 256, 1024, downsample=True, layer_name='layer3.0'),
                                    CBigBlock(1024, 256, 1024, layer_name='layer3.1'),
                                    CBigBlock(1024, 256, 1024, layer_name='layer3.2'),
                                    CBigBlock(1024, 256, 1024, layer_name='layer3.3'),
                                    CBigBlock(1024, 256, 1024, layer_name='layer3.4'),
                                    CBigBlock(1024, 256, 1024, layer_name='layer3.5'))
        self.layer4 = nn.Sequential(CBigBlock(1024, 512, 2048, downsample=True, layer_name='layer4.0'),
                                    CBigBlock(2048, 512, 2048, layer_name='layer4.1'),
                                    CBigBlock(2048, 512, 2048, layer_name='layer4.2'))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = CLinear(512, num_class)
        if pretrained:
            if pretrained is True:
                import timm
                state_dict = timm.create_model('resnet50', pretrained=True).state_dict()
            else:
                with open(pretrained, 'rb') as f:
                    state_dict = torch.load(f)
            self.conv1.load_pretrained(state_dict)
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    block.load_pretrained(state_dict)
            self.fc.load_pretrained(state_dict)
            print('remained state dict', state_dict.keys())

    def forward_features(self, x):
        x = self.conv1(x)
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

    def quantize(self, if_quantize):
        self.conv1.quantize(if_quantize)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.quantize(if_quantize)
        self.fc.quantize(if_quantize)


EPOCHS = 3

if __name__ == '__main__':
    import timm
    from data import train_dl, val_dl
    import tqdm
    import os

    SAVE = True
    SAVE_FOLDER = 'quantization_aware_training'

    # model = CResnet50(10, pretrained='checkpoint/origin_training/resnet50_w.pt')
    model = CResnet50(10, pretrained=True)

    # with open('checkpoint/quantization_aware/c_resnet50_w.pt','rb') as f:
    #     state_dict = torch.load(f)
    # model.load_state_dict(state_dict)

    # model = timm.create_model('resnet50',pretrained=True)
    # model.fc = nn.Linear(512,10)

    model.quantize(True)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-5)
    min_loss = float('inf')
    _step = 0
    for epoch in range(EPOCHS):
        train_bar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl.dataset) // train_dl.batch_size)
        train_bar.set_description(f'train - epoch:{epoch:3d}')
        loss_mean = []
        model.train()
        for step, (x, label) in train_bar:
            x, label = x.cuda(), label.cuda()
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, label)
            loss.backward()
            loss_mean.append(loss.item())
            train_loss = sum(loss_mean) / len(loss_mean)
            train_bar.set_postfix({'loss': train_loss})
            optimizer.step()
            _step += 1
        train_loss = sum(loss_mean) / len(loss_mean)

        val_bar = tqdm.tqdm(enumerate(val_dl), total=len(val_dl.dataset) // val_dl.batch_size)
        val_bar.set_description(f'val - epoch:{epoch:3d}')
        model.eval()
        loss_mean, acc_mean = [], []
        for step, (x, label) in val_bar:
            x, label = x.cuda(), label.cuda()
            y = model(x)
            loss = criterion(y, label)
            loss_mean.append(loss.item())
            acc = (y.argmax(dim=1) == label)
            acc_mean.extend(acc.tolist())
            val_bar.set_postfix({'val_loss': sum(loss_mean) / len(loss_mean), 'train_loss': train_loss,
                                 'acc': sum(acc_mean) / len(acc_mean)})
        val_loss = sum(loss_mean) / len(loss_mean)
        val_acc = sum(acc_mean) / len(acc_mean)
        if val_loss < min_loss:
            min_loss = val_loss
            if SAVE:
                os.makedirs(f'checkpoint/{SAVE_FOLDER}', exist_ok=True)
                with open(f'checkpoint/{SAVE_FOLDER}/c_resnet50_w.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)
                with open(f'checkpoint/{SAVE_FOLDER}/simple_log.txt', 'w') as f:
                    f.write(f'epoch:{epoch}\n'
                            f'train_loss:{train_loss}\n'
                            f'val_loss:{val_loss}\n'
                            f'acc:{val_acc}')
                # print(model.state_dict())
