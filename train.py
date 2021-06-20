#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=100,
                                            shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

criterion = nn.CrossEntropyLoss()
net = Net()
optimizer = optim.Adam(net.parameters())

dataloaders_dict = {
    "train": trainloader,
    "test": testloader
}

for i in tqdm(range(2)):
    for phase in ["train", "test"]:
        if phase == "train":
            net.train()
        else:
            net.eval()

        epoch_loss = 0.0
        epoch_corrects = 0

        for inputs, labels in dataloaders_dict[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)

                # 損失を計算
                loss = criterion(outputs, labels)

                # ラベルを予測
                _, preds = torch.max(outputs, 1)

                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    # パラメータの更新
                    optimizer.step()

                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)

                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

#%%
parameters = list(net.parameters())
parameters = [i.detach().numpy().tolist() for i in parameters]
weight, bias = parameters
with open("./weight.txt", "w") as f:
    weight = "\n".join([",".join(map(str, [bias[i]] + weight[i])) for i in range(10)])
    f.write(weight)