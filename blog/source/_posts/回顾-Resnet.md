---
title: 回顾-Resnet
date: 2018-12-20 13:40:56
tags:
    - Resnet

categories: 
    - 框架
    - pytorch
    - Resnet
---
# 1.Resnet 简介:
MSRA（微软亚洲研究院）何凯明团队的深度残差网络（[Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)）在2015年的ImageNet上取得冠军，该网络简称为ResNet（由算法Residual命名），

层数达到了152层，top-5错误率降到了3.57，而2014年冠军GoogLeNet的错误率是6.7。ResNet取得了5项第一，并又一次刷新了CNN模型在ImageNet上的历史!
****************
# 2. Resnet 网络结构:
** 1.昨天我们看了vgg,深刻的感觉到了加深网络所带来的好处,即:**

* <font color=red> 1.从我写的cnn一层→→Alexnet 的八层→→vgg16的十六层,可以看出层数越深,获取的特征越丰富,拟合能力越强
</font>

** 2.当然肯定想说,想通过VGG堆叠那样无限制加深网络,来加强拟合能力! 当然你会遇到以下问题**
* 1.对于原来的网络，如果简单地增加深度，会导致梯度弥散或梯度爆炸(注:就是前向传的过去,反向传不回来(想成小时候传话游戏,传着传着就变味了)).
 * <font color=red> 遇到这个问题,其实可以正则化初始化和加入正则化层（Batch Normalization）或中间层添加辅助损失（auxiliary loss）作为额外的监督。,这样训练几十层应该没问题!</font>
* 2.上面加深,没啥问题,但是在更深的网络有更高的训练误差和测试误差,即退化问题?   
(你肯定想问,我不是刚说越深越强吗?现在怎么又不行了?
(回答这个,最简单就是名字,退化嘛,就像无限逼近定理一样,理想很丰满,现实很骨感))
    * <font color=red> 退化问题,推断能是因为深层的网络并不是那么好训练(也就是求解器很难去利用多层网络拟合同等函数(再简单一点,课程太多,学的有点崩溃))</font>
***********
### 3.如何解决所谓的退化问题?
** 1. 首先我们看作者怎么玩的?**
如图:
![](https://blog.mviai.com/images\回顾-Resnet\res.png)

** 注 : 图太大,就保存在桌面缩小慢慢对比**
*************
** 1.1:我们继续看图说故事!**
   * 1.我们首先看vgg19和34层网络(图左和图中),作者跟大家一样想法,加深到34层,并且通过无数爬坑失败经验!进行改进了,如下:
     * 图中与图左差别:
         * 1.把每个maxpool(池化层)都改成了stride=2的卷积层做下采样(下采样就是缩小图片图片)
         * 2.去掉了最后那个耗费资源的全连接,用global average pool层替换
*************** 
然后就遇到了退化问题........
然后尝试了..................次
***************
** 终于在一次尝试中使用改进[Highway Network](https://arxiv.org/pdf/1507.06228v2.pdf)带参数gate控制shortcut(就是图右的弯曲黑线,我叫它脑筋急转弯),以前我觉得是学LSTM(学这个最深感触,就是感觉在学电路分析)中也有一个forget gate来控制着流入下一阶段的信息量.**

---------所以知识都是互通的!
* 1,怎么改进呢?
    *  上面是Highway Network(公路网络),不是还有更快的高速公路吗,就改成高速公路(刚才说的图右的弯曲黑线),什么是高速公路(官方叫恒等式变化),即简单,直接
 
********* 
**下面我们来看看这所谓高速路是不是水泥构造:
如图:**
![](https://blog.mviai.com/images\回顾-Resnet\c.png)
*************
* 从这个图,我们能看出什么呢?
    * 假设直下来表示很努力很努力的同学,弯弯曲曲的代表喜欢耍的同学!
x代表刚入学,大家都一样,最下面(F(x)+x)代表清华北大哈!
<font color=red> 
* 现在,我要喜欢耍的同学考入清华北大!
     * 怎么做呢?
         * 清华北大=F(x)+x,我是x!
我是不是只要学习我没有的F(x)呢, F(x)怎么来呢?
        * F(x)=清华北大-x,这就是残差,
            所以我们不用想怎么考清华北大,而是想考清华北大同学有啥优点!,学习其优点,就可以无限接近清华北大!</font>
*****************    
** 注: 后面三层残差,这个函数变成清华北大=F(x)+wx,多了个w,W是卷积操作，用来调整x的channel维度的。 
意思喊你不要一下啥都学,适合自己才是最好的!**
***********    
 # 3.Resnet创新点: 
 * 1.shortcut连接的方式使用恒等映射，如果residual block的输入输出维度不一致，对增加的维度用0来填充(使用0填充时，可以保证模型的复杂度最低，这对于更深的网络是更加有利的)；
 * 2.shortcut连接的方式维度一致时使用恒等映射,不一致时使用线性投影以保证维度一致
 * 3.ResNets并不能更好的表征某一方面的特征，但是ResNets允许逐层深入地表征更多的模型。(可看经典的残差网络变体)
 * 4.残差网络使得前馈式/反向传播算法非常顺利进行，在极大程度上，残差网络使得优化较深层模型更为简单 
 * 5.“shortcut”快捷连接添加既不产生额外的参数，也不会增加计算的复杂度。
 * 6.快捷连接简单的执行身份映射，并将它们的输出添加到叠加层的输出。通过反向传播的SGD，整个网络仍然可以被训练成终端到端的形式。

*************
*** 更多信息,可以看原文( 标题1处的链接 ) ***
***********

# 4.PyTorch实现:
** 为了简化,采用resnet32网络!**
如图:
![](https://blog.mviai.com/images/回顾-Resnet/r.png)
![](https://blog.mviai.com/images/回顾-Resnet/b.png)
* 网络架构:Resnet.py文件:
```python
import torch


'''
今天在vgg基础上再优雅点
'''
#遵从原来步骤
class Resnet(torch.nn.Module):
	#初始化
	def __init__(self):
		super(Resnet, self).__init__()
		# self.Sumlayers=self.make_layers()#这里我们交由make_layers函数创建相似的模块
		# #假设我们上面已经创好五个模块(文章第一幅图block1->block5)
	
		#现在最顶端的不同层,看文章34层那个最上面橘色简化图的7*7
		self.top=torch.nn.Sequential(
			torch.nn.Conv2d(3,64,7,2,3),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU (inplace=True),
			torch.nn.MaxPool2d(3,2,1))#最顶部构建完成

			
			
		#中间重复太多,交给make_layers函数创建相似的模块
		#第三个参数来表示有多少个捷径(高速公路)
		
		#先来一打紫色的(文中图)
		self.layer1 = self.make_layer (64,64,3)
		# 再来一打绿色的(文中图)
		self.layer2 = self.make_layer (64, 128, 4,stride=2)#图中第一个有/2
		# 再来一打橘色的(文中图)
		self.layer3 = self.make_layer ( 128,256, 6,stride=2)#图中第一个有/2
		# 再来一打银色的(文中图)
		self.layer4= self.make_layer (256, 512, 3, stride=2)  # 图中第一个有/2
		#中间重复的构造完了
		
		
		#开始最后的了
		
		self.avgPool =torch.nn.AvgPool2d(7)#全局平均化
		self.fc = torch.nn.Linear (512, 2)#最后分成猫狗两类
		self.last=torch.nn.Softmax (dim=1)
		
	
		
	#前向传播
	def forward (self, x):
		x = self.top (x)
		x = self.layer1 (x)
		x = self.layer2 (x)
		x = self.layer3 (x)
		x = self.layer4 (x)
		x = self.avgPool(x)
		res = x.view (x.size (0), -1)  # 展平多维的卷积图成 一维
		out = self.fc(res)
		out = self.last(out)
		return out
	
	
	#构建刚才的构建模块函数make_layers
	def make_layer(self,in_c,out_c,n_block,stride=1):

		#创建一个列表,用来放后面层 ,后面我们直接往里面添加就可以了
		Sumlayers=[]
		
		#构建捷径(高速公路)
		shortcut=torch.nn.Sequential(
			torch.nn.Conv2d(in_c,out_c,1,stride),#1*1卷积
			torch.nn.BatchNorm2d(out_c),
		)
		#构建完成残差
		Sumlayers.append(ResBlock(in_c,out_c,stride,shortcut))

		#构建右边的公路
		for i in range(1,n_block):
			Sumlayers.append (ResBlock (out_c, out_c))#注意输入,输出应该一样
		
		return torch.nn.Sequential (*Sumlayers) #然后把构建好模型传出
	

#构建残差块 因为参数是变动的,所以引入变量,最后一个变量表示快捷通道个数,默认没有
class ResBlock(torch.nn.Module):
	def __init__(self,in_c,out_c,stride=1,shortcut=None):
		super(ResBlock, self).__init__()
		#左边的公路
		self.left=torch.nn.Sequential(
			torch.nn.Conv2d (in_c,out_c,3,stride,1),
			torch.nn.BatchNorm2d (out_c),
			torch.nn.ReLU (inplace=True),

			torch.nn.Conv2d (out_c,out_c,3,1,1),#注意 这里输入输出应该一样
			torch.nn.BatchNorm2d (out_c)
		)

		#右边的高速公路
		self.right=shortcut

		#最后
		self.last_y=torch.nn.ReLU()
	#前向
	def forward(self, x):
		y_l=self.left(x)
		y_r = x if self.right is None else self.right (x) #如果有高数路为空,就直接保存在res中,否则执行高速路保存在res
		sum_x=y_l+y_r #两个总和
		out=self.last_y(sum_x)
		return out




```
* 网络架构:Train.py文件:
```python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as  transforms
from torch.autograd import Variable

# 工具包
import argparse
# 载入网络
from Resnet import Resnet

#############参数设置#############
####命令行设置########
parser = argparse.ArgumentParser (description='CNN')  # 导入命令行模块
# 对于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
parser.add_argument ('--batch-size', type=int, default=2, metavar='N',
					 help='训练batch-size大小 (default: 2)')
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
data_transforms = transforms.Compose ([transforms.Scale (224),  # 通过调整比例调整大小,会报警
									   transforms.CenterCrop (224),  # 在中心裁剪给指定大小方形PIL图像
									   transforms.ToTensor ()])  # 转换成pytorch 变量tensor
###############数据载入################
train_dataset = datasets.ImageFolder (root="./data/train/",  # 保存目录
									  transform=data_transforms)  # 把数据转换成上面约束样子

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
Resnet=Resnet ()
if not args.no_cuda:
	print ('正在使用gpu')
	Resnet.cuda ()
print (Resnet)

###############损失函数##################
criterion = nn.CrossEntropyLoss ()  # 内置标准损失
optimizer = torch.optim.Adam (Resnet.parameters (), lr=args.lr)  # Adam优化器
#############训练过程#####################
total_loss = 0 #内存循环使用
for epoch in range (args.epochs):
	for i, (images, labels) in enumerate (train_loader):  # 枚举出来
		if not args.no_cuda:  # 数据处理是否用gpu
			images = images.cuda ()
			labels = labels.cuda ()
		
		images = Variable (images)  # 装箱
		labels = Variable (labels)
		
		##前向传播
		optimizer.zero_grad ()
		outputs = Resnet (images)
		# 损失
		loss = criterion (outputs, labels)
		# 反向传播
		loss.backward ()
		optimizer.step ()#更新参数
		total_loss += loss#内存循环使用 防止cuda超出内存
		##打印记录
		
		if (i + 1) % args.log_interval == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
				   % (epoch + 1, args.epochs, i + 1, len (train_dataset) // args.batch_size, loss.item ()))
		
		# 保存模型
		torch.save (Resnet.state_dict (), 'Resnet.pkl')


```
结果:
![](https://blog.mviai.com/images/回顾-Resnet/ret.png)
* 注意残差的前向传播 ,一不小心都不知道自己哪里错了!!
