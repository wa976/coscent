import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


EPOCHS     = 40
BATCH_SIZE = 16



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




# ## 데이터셋 불러오기
trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((28, 28)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root="./room_scenes(train)", transform=trans)
testset = torchvision.datasets.ImageFolder(root="./room_scenes(test)", transform=trans)

len(trainset)
print(len(trainset))

len(testset)
print(len(testset))


batch_size_train = 16

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size = batch_size_train,
                                          shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size = batch_size_train,
                                          shuffle = True)

classes_train = trainset.classes
print(classes_train)
print("classes name = ", classes_train )
num_classes = len(classes_train)

classes_test = testset.classes
print(classes_test)
print("classes name = ", classes_test )
num_classes = len(classes_test)


print("num classes = ",num_classes)


# ## 뉴럴넷으로 Fashion MNIST 학습하기

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


# ## 하이퍼파라미터
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다. 일반적으로 CPU 1개만 사용할 경우 필요는 없지만, GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다. 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.
PATH = './cifar_net.pth'
model = Net()
model.load_state_dict(torch.load(PATH))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy





# ## 학습하기
epoch_list = []
test_loss_list = []
test_accuracy_list = []

for epoch in range(5):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 15 == 14:    # print every 2000 mini-batches
            running_loss = 0.0
            test_loss, test_accuracy = evaluate(model, test_loader)

            print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                epoch, test_loss, test_accuracy))

            epoch_list.append(epoch)
            test_loss_list.append(test_loss)

print('Finished Training')
# ## 테스트하기



# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


dataiter = iter(test_loader)
images, labels = dataiter.next()

# 이미지를 출력합니다.



model = Net()
model.load_state_dict(torch.load(PATH))

outputs = model(images)

_, predicted = torch.max(outputs, 1)

print('GroundTruth: ', ' '.join('%5s' % classes_test[labels[j]] for j in range(16)))
print('Predicted: ', ' '.join('%5s' % classes_test[predicted[j]]
                              for j in range(16)))

imshow(torchvision.utils.make_grid(images))