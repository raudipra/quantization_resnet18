import torch
from q_model import QResnet18
from data import val_dl
import tqdm

if __name__ == '__main__':
    model = QResnet18(10)
    with open('checkpoint/quantized_model/q_resnet18.pt', 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.eval()

    # print(state_dict.keys())

    acc = []
    bar = tqdm.tqdm(val_dl)
    for x, label in bar:
        y = model(x)
        acc.extend((y.argmax(dim=1) == label).tolist())
        bar.set_postfix({'acc':sum(acc) / len(acc)})
    print('acc:', sum(acc) / len(acc))
