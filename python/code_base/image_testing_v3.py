import torch 
import pandas as pd
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lbfgs_modify import LBFGSModify
from LBFGS import LBFGS
import torch.nn.functional as F
import math
import time
import csv

batch_size = 128
epochs = 5
correct=0
total=0
# lambda1=1e-6
# lambda2=1e-3
lambda1=1e-6 # L1
lambda2=1e-6 # L2
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

 # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
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

# Reference https://pytorch.org/hub/pytorch_vision_resnet/
def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

# enable this to use wide ResNet
wide_resnet=False
if not wide_resnet:
  net=ResNet9().to(mydevice)
  #print("Using resnet9")
else:
  # use wide residual net https://arxiv.org/abs/1605.07146
  net=torchvision.models.resnet.wide_resnet50_2().to(mydevice)
  #print("Using resnet50")

### Define accuracy and loss for training and test
def training_accuracy(net):
   correct=0
   total=0
   for data in trainloader:
    # This will load based on the batch size 
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)
   return 100*correct/total

def testing_accuracy(net):
   correct=0
   total=0
   # This will load based on the batch size 
   for data in testloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)
   return 100*correct/total

def training_loss(net):
  train_loss=[]
  for data in trainloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice)).cpu()
     train_loss=criterion(outputs,labels)
     train_loss.backward()
  return train_loss 

def testing_loss(net):
  test_loss=[]
  for data in testloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice)).cpu()
     test_loss=criterion(outputs,labels)
     test_loss.backward()
  return test_loss  

def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        outputs=net(inputs)
        if not wide_resnet:
          l1_penalty=lambda1*(torch.norm(layer1,1)+torch.norm(layer2,1)+torch.norm(layer3,1)+torch.norm(layer4,1))
          l2_penalty=lambda2*(torch.norm(layer1,2)+torch.norm(layer2,2)+torch.norm(layer3,2)+torch.norm(layer4,2))
          loss=criterion(outputs,labels)+l1_penalty+l2_penalty
        else:
          l1_penalty=0
          l2_penalty=0
          loss=criterion(outputs,labels)
        if loss.requires_grad:
          loss.backward()
        return loss
lambda1=1e-6
lambda2=1e-10

# loss function and optimizer
criterion=nn.CrossEntropyLoss()
#optimizer=torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer=torch.optim.Adam(net.parameters(), lr=0.001)
#optimizer=LBFGS(net.parameters(), history_size=7)
#optimizer=LBFGSNew(net.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)
#optimizer=LBFGSModify(net.parameters(), history_size=7, max_iter=5, line_search_fn=True,batch_mode=True,cost_use_gradient=False)

use_lbfgs=False
load_model=False
# update from a saved model 
if load_model:
  checkpoint=torch.load('./res18.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.train() # initialize for training (BN,dropout)

start_time=time.time()

if __name__ == '__main__':
    training_loss_values=[]
    test_loss_values=[]
    test_acc_values=[]
    training_acc_values=[]
    display_data=1
    # train network
    for epoch in range(epochs):  # Setting to 20 Epochs
      running_loss=0.0
      # The trainloader samples the given batch size without replacement
      for i,data in enumerate(trainloader,0):
        # get the inputs
        inputs,labels=data
        # wrap them in variable
        inputs,labels=Variable(inputs).to(mydevice),Variable(labels).to(mydevice)
    
        if not use_lbfgs:
         # zero gradients
         optimizer.zero_grad()
         # forward+backward optimize
         outputs=net(inputs)
         loss=criterion(outputs,labels)
         loss.backward()
         optimizer.step()
        else:
          if not wide_resnet:
            layer1=torch.cat([x.view(-1) for x in net.layer1.parameters()])
            layer2=torch.cat([x.view(-1) for x in net.layer2.parameters()])
            layer3=torch.cat([x.view(-1) for x in net.layer3.parameters()])
            layer4=torch.cat([x.view(-1) for x in net.layer4.parameters()])
            optimizer.step(closure) 
            # End for loop
     
     # Training accuracy
      if display_data:
       current_training_acc=0
       correct=0
       total=0
       outputs=net(inputs)
       cnn_loss=criterion(outputs,labels)         
       training_loss_values.append(cnn_loss.item())
       _,predicted=torch.max(outputs.data,1)
       correct += (predicted==labels.to(mydevice)).sum()
       total += labels.size(0)
       current_training_acc=100*torch.div(correct, total, rounding_mode='trunc')
       training_acc_values.append(current_training_acc.item())   
     
     # Testing loss
       current_test_loss=testing_loss(net)
       test_loss_values.append(current_test_loss.item())
    
       #print('%f: [%d] Training Loss: %d Training Acc: %d %% Testing Loss: %d Testing Acc: %d %%'%(time.time()-start_time,epoch+1,training_loss,current_training_acc,current_test_loss,current_test_acc))
      current_test_acc=0  
      current_test_acc=testing_accuracy(net)
      test_acc_values.append(current_test_acc.item())
      print('%f: [Epoch: %d] Test Acc: %d %%'%(time.time()-start_time,epoch+1,current_test_acc))
      if (current_test_acc>40):
          print('Break out condition met for optimal values')
          break
      
      # if math.isnan(loss.data.item()):
      #   print('loss became nan at %d'%i)
      #   break
    # End epoch loop
    
    print('%f: Finished Training'%(time.time()-start_time))
    if display_data:
      plt.plot(training_acc_values)
      plt.plot(test_acc_values)
      plt.title('Model Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()
      print('Training Loss Values: ',training_loss)
      print('Test Loss Values: ',test_loss_values)
      print ('Training Accuracy Values: ',training_acc_values)
      print ('Test Accuracy Values: ',test_acc_values)
    # save model (and other extra items)
    torch.save({
                'model_state_dict':net.state_dict(),
                'epoch':epoch,
                'optimizer_state_dict':optimizer.state_dict(),
                'running_loss':running_loss,
               },'./res.model')
     
    # Final values
    print('Training Accuracy: %d %%'%
        (training_accuracy(net)))
    print('Testing Accuracy: %d %%'%
        (testing_accuracy(net)))
    print('Training Loss: %f'%
        (training_loss(net)))
    print('Testing Loss: %f'%
        (testing_loss(net)))
    # class_correct=list(0. for i in range(10))
    # class_total=list(0. for i in range(10))
    # for data in testloader:
    #   images,labels=data
    #   outputs=net(Variable(images).to(mydevice)).cpu()
    #   _,predicted=torch.max(outputs.data,1)
    #   c=(predicted==labels).squeeze()
    #   for i in range(4):
    #     label=labels[i]
    #     class_correct[label] += c[i]
    #     class_total[label] += 1
    
    # for i in range(10):
    #   print('Accuracy of %5s : %2d %%' %
    #     (classes[i],100*float(class_correct[i])/float(class_total[i])))
      
      ### Save data for later
    
    # field names 
    #if display_data: 
       # fields = ['Training Loss', 'Testing Loss', 'Training Accuracy', 'Testing Accuracy']    
       # filename = "c:/users/friedele/Repos/DRFM/image_test_data.csv"
       # list_dict = [training_loss_values,
       #              test_loss_values,
       #              training_acc_values,
       #              test_acc_values]
       # df = pd.DataFrame(list_dict)
       # df.to_csv(filename, columns = fields,index=False)

