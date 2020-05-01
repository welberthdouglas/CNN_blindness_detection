import os

import pandas as pd
import numpy as np
import time

import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class TestData(Dataset):
    
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        diagnosis = self.data.diagnosis[idx]
        return image,diagnosis
    
    

train_transform = transforms.Compose([
    transforms.Resize((264, 264)),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    

validation_split = .2
shuffle_dataset = True
random_seed= 42



dataset = TestData(csv_file='aptos2019-blindness-detection/train.csv',
                                      transform=train_transform)


# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
    
    


class DR_net(nn.Module):
    
    def __init__(self):
        
        super(DR_net,self).__init__()
        
        self.conv1 = nn.Conv2d(3,10,5)
        self.conv2 = nn.Conv2d(10,16,7)
        self.conv3 = nn.Conv2d(16,20,5)
        self.conv4 = nn.Conv2d(20,40,7)        
        
        self.fc1 = nn.Linear(40*26*26,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.out = nn.Linear(10,5)
        
    def forward(self, x):
        
        # convolutional layers
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = F.max_pool2d(F.relu(self.conv4(x)),2)
        
        # Dense layers
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # output layer
        x = self.out(x)
        # x = F.softmax(x, dim=1) not required here since the loss function (cross entropy) uses sofmax implilcitly
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
#params = list(network.parameters())
#print(len(params))
#print(params[0].size())
              
CNN = DR_net()

batch_size = 25

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

optimizer = optim.Adam(CNN.parameters(), lr=0.003)


#batch = next(iter(train_loader))
#images,labels = batch
#
#
#optimizer.zero_grad()
#pred = CNN(images)
#loss = F.cross_entropy(pred,labels)
#cor_pred = pred.argmax(dim=1).eq(labels).sum()
#print(loss)
#print(cor_pred)
#loss.backward()
#optimizer.step()



start = time.time()
acc = []

for epoch in range(2):  # loop over the dataset multiple times
    
    print("Epoch: {}".format(epoch))
    running_loss = 0.0
    
    for i, batch in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = CNN(inputs)
        loss = F.cross_entropy(outputs, labels)
        
        cor_pred = outputs.argmax(dim=1).eq(labels).sum()

        print("Test loss: {}  Test accuracy: {}".format(loss,(cor_pred.numpy()/batch_size)))
        print('----------------------')
        
#        loss.backward()
#        optimizer.step()
#        
#        # print statistics
#        running_loss += loss.item()
#        if i % 10 == 9:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 100))
#            running_loss = 0.0
    
#    correct = 0
#    total = 0
#    with torch.no_grad():
#        for data in validation_loader:
#            images, labels = data
#            outputs = CNN(images)
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()

#    print('Accuracy of the network on the ',str(len(valid_sampler)),' test images: %d %%' % (100 * correct / total))
#    acc.append(correct / total)

end = time.time()
runingtime = (end - start)
print('Finished Training  {}'.format(runingtime))

batch_size = 100
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

preds=[]
labels=[]
with torch.no_grad():
    for data in validation_loader:
        batch = next(iter(validation_loader))
        images,label = batch
        out = CNN(images)
        print(out.argmax(dim=1))
        print(label)
        print(out.argmax(dim=1).eq(label).sum())



