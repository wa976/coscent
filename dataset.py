import torch
import torchvision
#from matplotlib import get_label
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from glob import glob
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
from torch.utils import data

  #transforms.Resize(28, 28),


trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((28, 28)),
     transforms.Normalize((0.1307,), (0.3081,))
     ])

trainset = torchvision.datasets.ImageFolder(root="./room_scenes(train)", transform=trans)

len(trainset)
print(len(trainset))



batch_size_train = 16

trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size = batch_size_train,
                                          shuffle = True)


#trainloader = data.DataLoader(
    #dataset     = trainset,
    #batch_size  = 4
#)


classes = trainset.classes
print(classes)
print("classes name = ", classes )
num_classes = len(classes)

print("num classes = ",num_classes)

examples = enumerate(trainloader)


batch_idx, (example_data, example_targets) = next(examples)
example_data.shape


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    np.array(img, np.int32)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()



dataiter = iter(trainloader)
images, labels = dataiter.next()

# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(16)))

# 이미지 보여주기
#plt.imshow((out * 255).astype(np.uint8))
imshow(torchvision.utils.make_grid(images))


