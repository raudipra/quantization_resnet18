import torch
import torch.nn as nn
import torch.nn.functional as F
from q_layers import QConvBnReLU, QLinear, QAdd, QReLU, QMaxPool2d, QAdaptiveAvgPool2d
from c_model import CResnet50
from c_layers import QParam





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
