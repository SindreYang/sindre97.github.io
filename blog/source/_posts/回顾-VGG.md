---
title: 回顾-VGG
date: 2018-12-19 18:46:12
tags:
    - VGG
categories: 
    - 框架
    - pytorch
    - VGG
---
# 1. VGGNet 简介:
1.VGG是由Simonyan 和Zisserman在文献《[Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/abs/1409.1556)》中提出卷积神经网络模型，其名称来源于作者所在的牛津大学视觉几何组(Visual Geometry Group)的缩写。

2.该模型参加2014年的 ImageNet图像分类与定位挑战赛，取得了优异成绩：在分类任务上排名第二，在定位任务上排名第一。

**************
# 2. VGGNet 网络结构:
根据VGG中根据<font color=red>卷积核大小和卷积层数目的不同,</font>分为A,A-LRN,B,C,D,E共6个网络结构配置!
*********
D,E就是常用配置,也就是常说的<font color=red>VGG16和VGG19</font>

* 如图:
 ![](https://blog.mviai.com/images/回顾-VGG/table.png)
 
 ************
 ** 1.为什么叫vgg16? **
 如图绿色(D配置),即VGG16配置,
 * 1.用conv3表示3\*3卷积,conv3-xx中的xx表示通道数(也就是有多少个这样的卷积核)
 * 2.一共数下来 有13个conv3,3个FC(最下面那里),5个池化层(每个模块中间)
 * 3.13(卷积)+3(全连接)=16,不加池化层,是因为池化层不涉及权重.所以d配置叫VGG16,其他以此类推!
 ************
* 官方图:
  * 如图:
  ![](https://blog.mviai.com/images/回顾-VGG/block.png)


********
* 这个图又看出什么?
    * 1.每个卷积层(上面的conv3-xx)里面是表示conv+relu(图中黑色方块注释)
    * 2.看到红色方块(池化层)没,是不是经过它,似乎方块都变小1/2了!
        * 即池化层采用的参数均为2\*2,步幅stride=2，max的池化方式，这样就能够使得每一个池化层的宽和高是前一层的1/2。
*********

* 输入图片大小变化图:
![](https://blog.mviai.com/images/回顾-VGG/VGG16.png)

*********
* 这个图又看出什么?
    * 1.图片从224\*244大小→112\*112→56\*56→28\*28→ 14\*14→7\*7大小(图最左边Size),大小一直减半
    * 2.卷积通道数(卷积核数量)从64-->128-->256-->512,然后固定在512!   都是2倍数(因为计算机以2进制计算)----------------2是个好东西
***********
* 计算量图:
 ![](https://blog.mviai.com/images/回顾-VGG/c.png)
** 注:memory=内存计算(红色),params=参数量计算(蓝色)**
***********


# 3.VGG优缺点:
** 优点:**
* 1.VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）
* 2.几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好
* 3.通过不断加深网络结构可以提升性能.具有很高的拟合能力(怎么加深?(<font color=red>这个网络告诉我们就像汉堡一样一层层堆叠上去</font>))
*************
** 缺点:**
* 1.训练时间过长(3个全连接啊(<font color=red>听说:发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量</font>))，调参难度大
* 2.需要的存储容量大，不利于部署。(单片机就那几MB可怜的空间)
************

# 4.PyTorch实现:
* 网络架构:VGG.py文件:
```python
import torch


'''
今天这个有点难得堆
一个个写,重复性太高,而且不利于观看,不优雅
重复性事我们交给你喊函数完成
'''
#遵从原来步骤
class VGG(torch.nn.Module):
	#初始化
	def __init__(self):
		super(VGG, self).__init__()
		self.Sumlayers=self.make_layers()#这里我们交由make_layers函数创建相似的模块
		#假设我们上面已经创好五个模块(文章第一幅图block1->block5)
	
		#现在创建最后的Fc
		self.fc=torch.nn.Sequential(
			torch.nn.Linear (7*7*512, 4096),  # 第一个全连接
			torch.nn.ReLU(),
			
			torch.nn.Linear (4096, 4096), # 第二个全连接
			torch.nn.ReLU(),
			
			torch.nn.Linear (4096, 2),  # 第三个全连接
			#  原版1000类       最后分成2类 因为只有猫狗两个类
			torch.nn.ReLU(),
			
			torch.nn.Softmax (dim=1),  # 最后一个 softmax 不填dim=1会报警 1.0以前好像可以直接写Softmax ()
			
			
		)
		
	#前向传播
	def forward (self, x):
		conv = self.Sumlayers (x)
		res = conv.view (conv.size (0), -1)  # 展平多维的卷积图成 一维
		out = self.fc(res)
		return out
	
	
	#构建刚才的构建模块函数make_layers
	def make_layers(self):
		#创建一个列表,用来快速构造模块,你也可以测试vgg19等等
		vgg16=[64, 64, 'Maxpool', 128, 128, 'Maxpool', 256, 256, 256, 'Maxpool',
					   512, 512, 512, 'Maxpool', 512, 512, 512, 'Maxpool']
		#创建一个列表,用来放后面层 ,后面我们直接往里面添加就可以了
		Sumlayers=[]
		
		#创建一个变量，来控制 卷积参数输入大小（in_channels）和输出大小（out_channels）
		in_c = 3 #第一次输入大小
		#遍历列表
		for x in vgg16: #获取到每个配置,我这只有vgg16这一行
			if x =='Maxpool':#如果遇到Maxpool ,我们就创建maxpool层
				Sumlayers+=[torch.nn.MaxPool2d(kernel_size=2, stride=2)]#参数看上文
			else: #否则我们创建conv(卷积模块)
				Sumlayers+= [torch.nn.Conv2d (in_channels=in_c,out_channels=x , kernel_size=3, padding=1), #x是列表中的参数
						   	torch.nn.BatchNorm2d (x),#标准化一下
						   	torch.nn.ReLU ()]
				in_c=x #输出大小成为下个输入大小
		
		return torch.nn.Sequential (*Sumlayers) #然后把构建好模型传出
```
**********
*********
* 网络架构:train.py文件:
```python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as  transforms
from torch.autograd import Variable

# 工具包
import argparse
# 载入网络
from VGG import VGG

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
VGG=VGG ()
if not args.no_cuda:
	print ('正在使用gpu')
	VGG.cuda ()
print (VGG)

###############损失函数##################
criterion = nn.CrossEntropyLoss ()  # 内置标准损失
optimizer = torch.optim.Adam (VGG.parameters (), lr=args.lr)  # Adam优化器
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
		outputs = VGG (images)
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
		torch.save (VGG.state_dict (), 'VGG.pkl')


```

**************
**********
结果如图:
![](https://blog.mviai.com/images/回顾-VGG/t.png)

**注:如果提示内存不够,可以在命令行用下面命令训练:**
```python
python train.py --no-cuda --batch-size=2
```
** 因为我自带内存8g,显卡内存2g,所以我不用gpu跑**
