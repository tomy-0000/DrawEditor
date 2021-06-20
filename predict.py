import numpy as np
import torchvision

weight = []
with open("./weight.txt") as f:
    for i in range(10):
        tmp = f.readline()
        tmp = list(map(float, tmp.split(",")))
        weight.append(tmp)

mnist = torchvision.datasets.MNIST(root="./data", train=False)
tmp = mnist[0]
img = np.array(tmp[0])
label = tmp[1]
img = img.reshape(-1)
score = []
for i in range(10):
    s = weight[i][0]
    for j in range(28*28):
        s += img[j]*weight[i][j + 1]
    score.append(s)
print(label, np.argmax(score))