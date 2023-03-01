---
title: 回顾-AlexNet
date: 2018-12-18 20:49:57
tags:
    - AlexNet
categories:
    - 框架
    - pytorch
    - AlexNet
---
# 1. AlexNet 简介:
AlexNet由Alex Krizhevsky于2012年提出，夺得2012年ILSVRC比赛的冠军，top5预测的错误率为16.4%，远超第一名。
******************
# 2. AlexNet 网络结构:
AlexNet采用8层的神经网络，5个卷积层和3个全连接层(3个卷积层后面加了最大池化层)，包含6亿3000万个链接，6000万个 参数和65万个神经元。
* 如图:

* ![](https://blog.mviai.com/images/alexnet.png)
**********
*******************
* 官方图:
* ![](https://blog.mviai.com/images/a2.png)
************
# 3. AlexNet 改进点:
* 1.使用ReLU作为CNN的激活函数，验证了其效果在较深的网络中超过了Sigmoid.解决了Sigmoid在网络较深时的梯度弥散问题。
* 2.使用最大池化可以避免平均池化的模糊效果。同时重叠效果可以提升特征的丰富性。
* 3.提出LRN（Local Response Normalization，即局部响应归一化）层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
* 4.数据增强，随机的从256` *`256的图片中截取224` *`224大小的区域（以及水平翻转的镜像），相当于增加了（256-224)\*(2^2)=2048 倍的数据量，如果没有数据增强，模型会陷入过拟合中，使用数据增强可以增大模型的泛化能力。
*  5.使用CUDA加速神经网络的训练，利用了GPU强大的计算能力。
*  6.训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合，一般在全连接层使用，在预测的时候是不使用Dropout的，即Dropout为1.
*****************
# 4.PyTorch实现:
* 网络架构:AlexNet.py文件:
```python
import torch

#跟着第一幅图走
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 输入图片大小 227*227*3
		#第一层
        self.conv1 = torch.nn.Sequential(
			#卷积
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            # (227-11)/4+1=55,  输出图片大小:55*55*96
            torch.nn.ReLU(),#激活层
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            # (55-3)/2+1=27, 输出图片大小: 27*27*96
        )

        # 从上面获得图片大小27*27*96
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # （27-5 + 2*2）/ 1 + 1 = 27, 输出图片大小:27*27*256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            # (27 - 3 )/2 + 1 = 13, 输出图片大小:13*13*256
        )

        # 从上面获得图片大小13*13*256
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # (13 - 3 +1*2)/1 + 1 = 13 , 输出图片大小:13*13*384
            torch.nn.ReLU()
        )

        # 从上面获得图片大小13*13*384
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # (13 - 3 + 1*2)/1 +1 = 13, 输出图片大小:13*13*384
            torch.nn.ReLU()
        )

        # 从上面获得图片大小13*13*384
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # (13 - 3 + 1*2) +1 = 13, 13*13*256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            # (13 - 3 )/2 +1 =6, 6*6*256
        )

        # 从上面获得图片大小 6*6*256 = 9216 共9216输出特征
        self.lostlayer = torch.nn.Sequential(
			#第六层
            torch.nn.Linear(9216, 4096),#全连接
            torch.nn.ReLU(),#激活层
            torch.nn.Dropout(0.5),#以0.5&几率随机忽略一部分神经元
			
			#第七层
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
			
			#第八层
            torch.nn.Linear(4096, 2)
            # 最后输出2 ,因为只分猫狗两类
        )

	#前向传播
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)#展平多维的卷积图成 一维(batch_size, 4096)
        out = self.lostlayer(res)
        return out
```

*********************
******************
* 训练架构:train.py文件 猫狗10张图:
```python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as  transforms
from torch.autograd import Variable

# 工具包
import argparse
# 载入网络
from AlexNet import   AlexNet

#############参数设置#############
####命令行设置########
parser = argparse.ArgumentParser (description='CNN')  # 导入命令行模块
# 对于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
parser.add_argument ('--batch-size', type=int, default=32, metavar='N',
					 help='训练batch-size大小 (default: 32)')
parser.add_argument ('--epochs', type=int, default=10, metavar='N',
					 help='训练epochs大小 (default: 10)')
parser.add_argument ('--lr', type=float, default=0.001, metavar='LR',
					 help='学习率 (default: 0.001)')
parser.add_argument ('--no-cuda', action='store_true', default=False,
					 help='不开启cuda训练')
parser.add_argument ('--seed', type=int, default=1, metavar='S',
					 help='随机种子 (default: 1)')
parser.add_argument ('--log-interval', type=int, default=1, metavar='N',
					 help='记录等待n批次 (default: 1)')
args = parser.parse_args ()  # 相当于激活命令

args.cuda = not args.no_cuda and torch.cuda.is_available ()  # 判断gpu

torch.manual_seed (args.seed)
if args.cuda:
	torch.cuda.manual_seed (args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的


##########数据转换#####################
data_transforms = transforms.Compose([transforms.Scale(227),#通过调整比例调整大小,会报警
    								transforms.CenterCrop(227),#在中心裁剪给指定大小方形PIL图像
									transforms.ToTensor ()])#转换成pytorch 变量tensor
###############数据载入################
train_dataset = datasets.ImageFolder(root="./data/train/",  # 保存目录
									transform=data_transforms) # 把数据转换成上面约束样子

test_dataset = datasets.ImageFolder (root='./data/test/',
									transform=data_transforms)


##########数据如下####
# # root/dog/xxx.png
# # root/dog/xxy.png
# # root/dog/xxz.png
# #
# # root/cat/123.png
# # root/cat/nsdf3.png
# # root/cat/asd932_.png
######################

##############数据装载###############
train_loader = torch.utils.data.DataLoader (dataset=train_dataset,  # 装载数据
											batch_size=args.batch_size,  # 设置批大小
											shuffle=True)  # 是否随机打乱
test_loader = torch.utils.data.DataLoader (dataset=test_dataset,
										   batch_size=args.batch_size,
										   shuffle=True)

#############模型载入#################
AlexNet = AlexNet ()
if not args.no_cuda:
	print ('正在使用gpu')
	AlexNet.cuda ()
print (AlexNet)

###############损失函数##################
criterion = nn.CrossEntropyLoss ()  # 内置标准损失
optimizer = torch.optim.Adam (AlexNet.parameters (), lr=args.lr)  # Adam优化器
#############训练过程#####################
for epoch in range (args.epochs):
	for i, (images, labels) in enumerate (train_loader):  # 枚举出来
		if not args.no_cuda:  # 数据处理是否用gpu
			images = images.cuda ()
			labels = labels.cuda ()
		
		images = Variable (images)  # 装箱
		labels = Variable (labels)
		
		##前向传播
		optimizer.zero_grad ()
		outputs = AlexNet(images)
		# 损失
		loss = criterion (outputs, labels)
		# 反向传播
		loss.backward ()
		optimizer.step ()
		##打印记录
		
		if (i + 1) % args.log_interval == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
				   % (epoch + 1, args.epochs, i + 1, len (train_dataset) // args.batch_size, loss.item ()))
		
		# 保存模型
		torch.save (AlexNet.state_dict (), 'AlexNet.pkl')
```
************
* 效果如图:
![](https://blog.mviai.com/images/回顾-AlexNet/20181218112800405.png)
