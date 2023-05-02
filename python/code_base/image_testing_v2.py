import torch 
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lbfgsnew import LBFGSNew
import torch.nn.functional as F
import numpy as np
import time

batch_size = 128
epoch = 10
correct=0
total=0
lambda1=0.000001
lambda2=0.001
mydevice=torch.device('cpu')

### Data Loading
data_transform=transforms.Compose(
   [transforms.Resize(size=(32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# Setup path to data folder
data_path = Path("C:/Users/friedele/Repos/DRFM/images/")
image_path = data_path / "reducedTgts(64x64)"

# If the image folder doesn't exist, download it and prepare it... 
# if image_path.is_dir():
#    # print(f"{image_path} directory exists.")
# else:
#     print(f"Did not find {image_path} directory")
    
# Setup train and testing paths
train_dir = image_path / "training"
test_dir = image_path / "test"

# Use ImageFolder to create dataset(s)
trainset = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

testset = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform) 

# Get class names as a list
classes = trainset.classes
num_classes= len(classes)
 
# Turn train and test Datasets into DataLoaders
trainloader = DataLoader(dataset=trainset, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=2, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

testloader = DataLoader(dataset=testset, 
                             batch_size=batch_size, 
                             num_workers=2, 
                             shuffle=False) # don't usually need to shuffle testing data
#####################################################
def verification_error_check(net):
   correct=0
   total=0
   for data in testloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)

   return 100*correct//total
#####################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(32*32, 512) 
        self.linear2 = nn.Linear(512, 512) 
        self.final = nn.Linear(512, 4)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 32*32)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1])

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
         #print("Using zero gradient")
        outputs=net(inputs)
        l1_penalty=lambda1*(torch.norm(layer1,1)+torch.norm(layer2,1)+torch.norm(layer3,1)+torch.norm(layer4,1))
        l2_penalty=lambda2*(torch.norm(layer1,2)+torch.norm(layer2,2)+torch.norm(layer3,2)+torch.norm(layer4,2))
        loss=cross_entropy(outputs,labels)+l1_penalty+l2_penalty   # Cross entropy loss with softmax
        #Its using this: print("Apply penalty in Loss function")

        if loss.requires_grad:
          loss.backward()
         #Its using this: print('Backward Loss: %f l1 %f l2 %f'%(loss,l1_penalty,l2_penalty))
        return loss   
start_time = time.time()
   
if __name__ == '__main__':
    net=ResNet9().to(mydevice)
    cross_entropy = nn.CrossEntropyLoss()  # This combines with softmax
   # optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #e-1
    optimizer=LBFGSNew(net.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)
     
    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    
    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    use_lbfgs=True
    adam_loss = []
    cnn_loss = []

    for epoch in range(epoch):
        net.train()
        running_loss = 0.0
        print("--- %s seconds ---" % (time.time() - start_time))
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # wrap them in variable
            inputs,labels=Variable(inputs).to(mydevice),Variable(labels).to(mydevice)
            
            if not use_lbfgs:
                # Use for Adam
                # zero gradients
                optimizer.zero_grad()
                # forward+backward optimize
                outputs=net(inputs)
                loss=cross_entropy(outputs,labels)
                loss.backward()
                adam_loss.append(loss.item())
                optimizer.step()
            else:            
            # Use for LBFGS
                layer1=torch.cat([x.view(-1) for x in net.layer1.parameters()])
                layer2=torch.cat([x.view(-1) for x in net.layer2.parameters()])
                layer3=torch.cat([x.view(-1) for x in net.layer3.parameters()])
                layer4=torch.cat([x.view(-1) for x in net.layer4.parameters()])    
                outputs=net(inputs)
                optimizer.step(closure)
                loss=cross_entropy(outputs,labels)
                # print running statistics
                running_loss += loss.item()
                
                cnn_loss.append(loss.item())
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print("--- %s seconds ---" % (time.time() - start_time))
#### Analysis Code ####
# whole dataset
    correct=0
    total=0
    for data in trainloader:
       images,labels=data
       outputs=net(Variable(images).to(mydevice)).cpu()
       _,predicted=torch.max(outputs.data,1)
       total += labels.size(0)
       correct += (predicted==labels).sum()
       
    print('Accuracy of the network on the %d training images: %d %%'%(total,torch.div(100*correct, total, rounding_mode='trunc')))
       
    correct=0
    total=0
    for data in testloader:
       images,labels=data
       outputs=net(Variable(images).to(mydevice)).cpu()
       _,predicted=torch.max(outputs.data,1)
       total += labels.size(0)
       correct += (predicted==labels).sum()
       
    print('Accuracy of the network on the %d test images: %d %%'%(total,torch.div(100*correct, total, rounding_mode='trunc')))
    
    class_correct=list(0. for i in range(num_classes))
    class_total=list(0. for i in range(num_classes))
    
    for data in testloader:
      images,labels=data
      outputs=net(Variable(images).to(mydevice)).cpu()
      _,predicted=torch.max(outputs.data,1)
      c=(predicted==labels).squeeze()
      for i in range(4):
        label=labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
        
    for i in range(num_classes):
      if (class_total[i]==0):
          print('Class is 0.0: ',classes[i])
      else:
        print('Accuracy of %5s : %2d %%' %
        (classes[i],100*float(class_correct[i])/float(class_total[i])))
        
   ##### Plots ##### 
    plt.plot(cnn_loss)
    plt.title('Model Loss')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()