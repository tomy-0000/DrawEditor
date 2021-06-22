#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

trainset = torchvision.datasets.EMNIST(root='./data',
                                        split="digits",
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())

testset = torchvision.datasets.EMNIST(root='./data',
                                        split="digits",
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())

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
                inputs = torch.where(inputs > 0, 1.0, 0.0)
                inputs = inputs.permute(0, 1, 3, 2)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
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