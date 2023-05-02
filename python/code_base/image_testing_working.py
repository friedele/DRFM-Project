import torch 
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
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
# Done with Data loading
###################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

 # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
       self,
       inplanes: int,
       planes: int,
       stride: int = 1,
       downsample: Optional[nn.Module] = None,
       groups: int = 1,
       base_width: int = 64,
       dilation: int = 1,
       norm_layer: Optional[Callable[..., nn.Module]] = None,
   ) -> None:
       super().__init__()
       if norm_layer is None:
           norm_layer = nn.BatchNorm2d
       width = int(planes * (base_width / 64.0)) * groups
       # Both self.conv2 and self.downsample layers downsample the input when stride != 1
       self.conv1 = conv1x1(inplanes, width)
       self.bn1 = norm_layer(width)
       self.conv2 = conv3x3(width, width, stride, groups, dilation)
       self.bn2 = norm_layer(width)
       self.conv3 = conv1x1(width, planes * self.expansion)
       self.bn3 = norm_layer(planes * self.expansion)
       self.relu = nn.ReLU(inplace=True)
       self.downsample = downsample
       self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
       identity = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)
       out = self.relu(out)

       out = self.conv3(out)
       out = self.bn3(out)

       if self.downsample is not None:
           identity = self.downsample(x)

       out += identity
       out = self.relu(out)

       return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model



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
  print("Using resnet9")
else:
  # use wide residual net https://arxiv.org/abs/1605.07146
  net=torchvision.models.resnet.wide_resnet50_2().to(mydevice)
  print("Using resnet50")

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
   return 100*correct//total

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
   return 100*correct//total

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
#optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer=optim.Adam(net.parameters(), lr=0.001)
#optimizer=LBFGS(net.parameters(), history_size=7)
#optimizer=LBFGSNew(net.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)
optimizer=LBFGSModify(net.parameters(), history_size=7, max_iter=1, line_search_fn=True,batch_mode=True,cost_use_gradient=False)

use_lbfgs=True
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
    display_data=0
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
       cnn_loss=criterion(outputs,labels)         
       training_loss_values.append(cnn_loss.item())
       _,predicted=torch.max(outputs.data,1)
       correct += (predicted==labels.to(mydevice)).sum()
       total += labels.size(0)
       current_training_acc=100*correct//total
       training_acc_values.append(current_training_acc.item())   
     
     # Testing loss
       current_test_loss=testing_loss(net)
       test_loss_values.append(current_test_loss.item())
    
       #print('%f: [%d] Training Loss: %d Training Acc: %d %% Testing Loss: %d Testing Acc: %d %%'%(time.time()-start_time,epoch+1,training_loss,current_training_acc,current_test_loss,current_test_acc))
      current_test_acc=0  
      current_test_acc=testing_accuracy(net)
      test_acc_values.append(current_test_acc.item())
      print('%f: [Epoch: %d] Test Acc: %d %%'%(time.time()-start_time,epoch+1,current_test_acc))
      
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
    class_correct=list(0. for i in range(10))
    class_total=list(0. for i in range(10))
    for data in testloader:
      images,labels=data
      outputs=net(Variable(images).to(mydevice)).cpu()
      _,predicted=torch.max(outputs.data,1)
      c=(predicted==labels).squeeze()
      for i in range(4):
        label=labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
    
    for i in range(10):
      print('Accuracy of %5s : %2d %%' %
        (classes[i],100*float(class_correct[i])/float(class_total[i])))
      
      ### Save data for later
    
    # field names 
    if display_data: 
      fields = ['Training Loss', 'Testing Loss', 'Training Accuracy', 'Testing Accuracy']  
            
      # data rows of csv file  
      rows = [training_loss,  
              test_loss_values,  
              training_acc_values,  
              test_acc_values]   
      # name of csv file  
      filename = "image_test_data.csv"
            
      # writing to csv file  
      with open(filename, 'w') as csvfile:  
          # creating a csv writer object  
          csvwriter = csv.writer(csvfile)  
                
          # writing the fields  
          csvwriter.writerow(fields)  
                
          # writing the data rows  
          csvwriter.writerows(rows)
