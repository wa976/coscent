import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = Image.open('./room_scenes(test)/white/2434.jpg')
custom_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = custom_transform(image).unsqueeze(0)
print(image.shape, type(image))

PATH = 'cifar_net2.pth'

classes = ['black', 'blue', 'green', 'pink', 'white']



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        #print(torch.Tensor.size(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        


model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

outputs = model(image)
_, predicted = torch.max(outputs, 1)

print(classes[predicted])

#_, predicted = torch.max(outputs, 1)

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #img =  torch.ones(1, dtype=torch.uint8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow((img * 255).astype(np.uint8))
    plt.show()

imshow(image.squeeze(0))


'''
print('GroundTruth: ', ' '.join('%5s' % classes_test[labels[j]] for j in range(16)))
print('Predicted: ', ' '.join('%5s' % classes_test[predicted[j]]
                              for j in range(16)))

imshow(torchvision.utils.make_grid(images))
'''