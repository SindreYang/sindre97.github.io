---
title: 回顾-CNN
date: 2018-12-17 20:49:57
tags:
    - CNN
categories: 
    - 框架
    - pytorch
    - CNN
---
### 卷积神经网络通常包含以下几种层：

**1.卷积层（Convolutional layer）:**
卷积神经网路中每层卷积层由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法优化得到的。卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征。
**2.线性整流层（Rectified Linear Units layer, ReLU layer）:**
这一层神经的激活函数（Activation function）使用线性整流（Rectified Linear Units, ReLU）。
就像容器  不管你水怎样流  放进去都是容器形状(这叫容器整流)

**3.池化层（Pooling layer）:**
通常在卷积层之后会得到维度很大的特征，将特征切成几个区域，取其最大值或平均值，得到新的、维度较小的特征。
**4.全连接层（ Fully-Connected layer）:**
把所有局部特征结合变成全局特征，用来计算最后每一类的得分。

#### 卷积层:
**1.局部感知:**
相当于看一副很大很大的图,你必须从图中一区域(感受野)开始挨个浏览(扫描),你就能知道这幅图到底是啥!
有啥好处,一下看很大的图--身心疲惫,而且还不一定能看懂
慢慢看,又优雅,又高尚,还能快速抓到要点(就像你看到有德玛西亚,你就可以知道跟lol有关)
**2.权值共享(参数共享)**
比说,刚花了一万的力气创造了猪八戒和孙悟空,猪八戒找的到吃的,孙悟空能打小怪兽,
后面我遇到小怪兽就不用创造悟空了,直接拿过来打小怪兽就可以了!
如果遇到肚子饿了,孙悟空就没用了,就可以用猪八戒找吃的!

所以参数共享能节省不必要的重复消耗,加快计算
#### 池化层:
池化（pool）即下采样（downsamples），目的是为了减少特征图。池化操作对每个深度切片独立，规模一般为 2＊2，相对于卷积层进行卷积运算，池化层进行的运算一般有以下几种： 
* 最大池化（Max Pooling）。取4个点的最大值。这是最常用的池化方法。 
* 均值池化（Mean Pooling）。取4个点的均值。 
    * 注:
        池化操作将保存深度大小不变。
        如果池化层的输入单元大小不是二的整数倍，
        一般采取边缘补零（zero-padding）的方式补成2的倍数，然后再池化。
        
#### 全连接层:
**全连接层** <font color=red> **可相互转换** </font> **卷积层**

#### 常见卷积网络架构:
输入 -> [[卷积层(CONV) -> 激活层(RELU)]  * N -> 池化层(POOL)] *  M 
-> [全连接(fc) -> 激活层(RELU)]* K -> 全连接(fc)


### PyTorch:
##### 网络架构:**CNN.py文件:**

```python
import  torch.nn as nn

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()                                      # 输入MNIST 图片大小是(1,28,28)
		self.layer=nn.Sequential(nn.Conv2d(1,8,kernel_size=5,padding=2),#第一个参数,输入是1,表示输入图片通道为1 ,8表示输出,5卷积核大小,2补边大小
								 nn.BatchNorm2d(8),#归一化
								 nn.ReLU(),#激活层
								 nn.MaxPool2d(2),#池化层 到这 (8,28,28)图片就被池化成(8,14,14)了
								 )
		self.fc=nn.Linear(14*14*8,10)  #全连接层 第一个输入的特征数,第二个 输出的特征 (0-9) 共10特征
	
	#前向传播	
	def forward(self, x):
		out=self.layer(x)
		out=out.view(out.size(0),-1)#展平多维的卷积图成 (batch_size, 32 * 7 * 7)
		out=self.fc(out)
		return out
```

**训练架构:train.py文件**
```python
#pytorch
import torch
import torch.nn as nn
import  torchvision.datasets as datasets
import  torchvision.transforms as  transforms
from torch.autograd import Variable

#工具包
import argparse
#载入网络
from CNN import CNN

#############参数设置#############
####命令行设置########
parser = argparse.ArgumentParser(description='CNN') #导入命令行模块
#对于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='训练batch-size大小 (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='训练epochs大小 (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='学习率 (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='不开启cuda训练')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='随机种子 (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='记录等待n批次 (default: 50)')
args = parser.parse_args()#相当于激活命令


args.cuda = not args.no_cuda and torch.cuda.is_available()#判断gpu

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
    
    
###############数据载入################
train_dataset=datasets.MNIST(root="./data/",#保存目录
                             train=True,  #选择训练集
                             transform=transforms.ToTensor(), #把数据转换成pytorch张量 Tensor
                             download=True) #是否下载数据集
test_dataset=datasets.MNIST(root='./data/',
                            train=False,#关闭 表示选择测试集
                            transform=transforms.ToTensor(),
                            download=True)

##############数据装载###############
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,#装载数据
                                         batch_size=args.batch_size,#设置批大小
                                         shuffle=True)#是否随机打乱
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True)

#############模型载入#################
cnn=CNN()
if not args.no_cuda:
    print('正在使用gpu')
    cnn.cuda()
print(cnn)

###############损失函数##################
criterion=nn.CrossEntropyLoss()#内置标准损失
optimizer=torch.optim.Adam(cnn.parameters(),lr=args.lr)#Adam优化器

#############训练过程#####################
for epoch in range (args.epochs):
    for i, (images,labels) in enumerate(train_loader):#枚举出来
        if not  args.no_cuda:#数据处理是否用gpu
            images=images.cuda()
            labels=labels.cuda()
        
        
        images=Variable(images)#装箱
        labels=Variable(labels)
        
        ##前向传播
        optimizer.zero_grad()
        outputs=cnn(images)
        #损失
        loss=criterion(outputs,labels)
        #反向传播
        loss.backward()
        optimizer.step ()
        ##打印记录
        
        if (i+1)% args.log_interval==0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, args.epochs, i+1, len(train_dataset)//args.batch_size, loss.item()))
            
            
        #保存模型
        torch.save(cnn.state_dict(), 'cnn.pkl')
```
* 注:
* 数据集下载不下来,修改pytorch里面MNIST.py文件
* 效果如图:
* 
![](https://blog.mviai.com/images/回顾-CNN/20181218125050651.png)
