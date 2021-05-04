import os
from flask import Flask, request
from werkzeug.utils import secure_filename
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


app = Flask(__name__)

@app.route('/')
def upload_main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <form action="http://localhost:5000/file-upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit">
        </form>
    </body>
    </html>"""

@app.route('/file-upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        fstr = request.files['file'].read()
        f = np.fromstring(fstr, np.uint8)
        f = cv2.imdecode(f, cv2.IMREAD_UNCHANGED)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        print(f.shape, type(f))

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
                # print(torch.Tensor.size(x))
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 4 * 4)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        custom_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = custom_transform(f).unsqueeze(0)

        classes = ['black', 'blue', 'green', 'pink', 'white']

        model = Net()
        model.load_state_dict(torch.load('cifar_net2.pth'))

        outputs = model(image)
        _, predicted = torch.max(outputs, 1)


        return classes[predicted]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)