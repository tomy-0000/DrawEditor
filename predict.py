import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

weight = []
with open("./weight.txt") as f:
    for i in range(10):
        tmp = f.readline()
        tmp = list(map(float, tmp.split(",")))
        weight.append(tmp)

mnist = torchvision.datasets.EMNIST(root="./data", split="digits", train=False)
tmp = mnist[2]
img = np.array(tmp[0])
# img = np.where(img == 0, 0, 1)
print(img)
img = np.array(Image.open("./tmp.png").convert("L"))
img = np.where(img == 255, 0, 1)
print("-"*20)
print(img)

label = tmp[1]
img = img.reshape(-1)
score = []
for i in range(10):
    s = weight[i][0]
    for j in range(28*28):
        s += img[j]*weight[i][j + 1]
    score.append(s)
score = np.array(score)
score -= max(score)
print(score)
print(label, np.argmax(score), np.exp(score[np.argmax(score)])/sum(np.exp(score)))