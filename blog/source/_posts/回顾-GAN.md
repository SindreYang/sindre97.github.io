---
title: 回顾-GAN
date: 2018-12-21 13:40:56
tags:
    - GAN

categories: 
    - 框架
    - pytorch
    - GAN
---
# 1.GAN 简介：
[GAN](https://arxiv.org/abs/1406.2661)是2014年放出的一篇开山之作.LSTM作者认为GAN是其他模型的变种,因为他在92年提出PM（PredictabilityMinimization）模型,所以他一直认为GAN是goodfellow在自己PM模型上的改进!哈哈!!
*********
GAN的主要灵感来源于博弈论中零和博弈的思想,即一个网络与另一个网络对抗!从而互相提升自己,我以前曾想有没有互助网络!答案是肯定的,遇到在说!
***********
# 2. GAN网络结构:
如图:
![](https://blog.mviai.com/images/回顾-GAN/net.png)

* 从这图告诉我什么呢?
    * 首先看到是两个大字 G  D ,G表示生成网络,D表示判别网络!
    * 然后我们看这类似电路的网络图,我们从左往右看!
      * (主干线) 在变量空间构造Noise(噪声)→送给→G(生成网络)→生出→假东西
      * （看G上面） 一堆真东西
      * 真东西（上面），假东西（G生出来的）交给D（判别网络）给出两个的有多像真东西几率<font color=red>  （当然给出真东西（上面）几率为100%（因为本来就是真的），假东西（G生出来的）几率为0~100%）</font>
          * 如果很差，就告诉D（生成网络）你不行啊！！，然后D借鉴上次失败重新生成假东西，去骗G！
          * 就这样持续了。。。。。。世纪，最终达到平衡，这时候G肯定很强大了
   
   
* 然后我们说下平衡核心：
![](https://blog.mviai.com/images/回顾-GAN/g.jpg)
    * 然后我们从右到左说下:
        * 首先想解释E E就是期望(期望就是概率求和)
还有有就是log,这是逻辑回归那取对数似然出来的
        * 最右边,G(z):其中z表示噪声.G(z)就表示生成的假东西,D(G(z))就表示D是否要告诉G('你不行')的概率
            * 下角的 z-Pz表示一个噪声到噪声数据集
        * 然后是D(x)表示看模板(真东西)是不是真东西概率
            * 下角的 x-Pdata 表示x到真实数据集(表示要慢慢来,慢慢学,不然容易崩溃哟)
        * 最后 min max ,你肯定有点不解,不是应该maxG吗? 简单理解下,当G最失败(最小)时候,就是让D看假东西得出概率最大.   (最垃圾时候,D都说你行,那就别说G强大时候了)   即在局部最大中试图找出全局最大
  
  
  
**************
# 3.GAN的创新点:
* 1. 相比较传统的模型，他存在两个不同的网络，而不是单一的网络，并且训练方式采用的是对抗训练方式
* 2. GAN中G的梯度更新信息自于判别模型D的一个反传梯度。，而不是来自数据样本
***********
# 4.GAN的优缺点:
#### 优点:

●   训练时不需要对隐变量做推断,而且G参数来源于D

●   GAN是一种生成式模型，相比较其他生成模型（玻尔兹曼机和GSNs）只用到了反向传播,而不需要复杂的马尔科夫链

●  相比其他所有模型, GAN可以产生更加清晰，真实的样本

●  GAN采用的是一种无监督的学习方式训练，可以被广泛用在无监督学习和半监督学习领域

●  相比于变分自编码器, GANs没有引入任何决定性偏置( deterministic bias),变分方法引入决定性偏置,因为他们优化对数似然的下界,而不是似然度本身,这看起来导致了VAEs生成的实例比GANs更模糊

●  相比VAE, GANs没有变分下界,如果鉴别器训练良好,那么生成器可以完美的学习到训练样本的分布.换句话说,GANs是渐进一致的,但是VAE是有偏差的

●  GAN应用到一些场景上，比如图片风格迁移，超分辨率，图像补全，去噪，避免了损失函数设计的困难，不管三七二十一，只要有一个的基准，直接上判别器，剩下的就交给对抗训练了。
#### 缺点:
●  很不好训练,一是DG的同仇敌忾问题(同步)

●  G参数来源于D,所以很难解释G的模型分布

●  训练GAN需要达到纳什均衡,有时候可以用梯度下降法做到,有时候做不到.我们还没有找到很好的达到纳什均衡的方法,所以训练GAN相比VAE或者PixelRNN是不稳定的,但我认为在实践中它还是比训练玻尔兹曼机稳定的多

●  GAN不适合处理离散形式的数据，比如文本

●  GAN存在训练不稳定、梯度消失、模式崩溃的问题（就是自我欺骗,最后谁也不行!解决在WGAN）


# 5.GAN的训练建议:
注:来自https://github.com/soumith/ganhacks#authors
** 1.标准化输入**
* 在-1和1之间标准化图像
* Tanh作为发电机输出的最后一层

** 2：修改的损失函数**
在GAN论文中，优化G的损失函数是min (log 1-D)，但实际上人们实际使用max log D

* 因为第一个损失在早期就已经消失了
* Goodfellow et。al（2014）
** 在实践中，运作良好：** 
* 训练生成器时调换标签：real = fake，fake = real


** 3：使用球形**

* 不要从统一分布中取样
![](https://blog.mviai.com/images/回顾-GAN/cube.png)
* 来自高斯分布的样本
![](https://blog.mviai.com/images/回顾-GAN/sphere.png)

* 进行插值时，通过大圆进行插值，而不是从A点到B点的直线
* Tom White的采样生成网络参考代码https://github.com/dribnet/plat



** 4：BatchNorm **
* 构建真实和假的不同小批次，即每个小批量只需要包含所有真实图像或所有生成的图像。
* 当batchnorm不是一个选项时，使用实例标准化（对于每个样本，减去平均值并除以标准偏差）。
![](https://blog.mviai.com/images/回顾-GAN/batchmix.png)


** 5：避免稀疏梯度：ReLU，MaxPool**
* 如果你的梯度稀疏，GANs的稳定性就会受到影响
* LeakyReLU =优良（G和D都有）
* 对于Downsampling，使用：Average Pooling，Conv2d+ stride
* 对于Upsampling，请使用：PixelShuffle,ConvTranspose2d + stride
    * [PixelShuffle](https：//arxiv.org/abs/1609.05158)


** 6：使用平滑和复杂的标签**
* 标签平滑，即如果你有两个目标标签：Real = 1和Fake = 0，那么对于每个传入的样本，如果它是真的，那么用0.7到1.2之间的随机数替换标签，如果它是假的样本，将其替换为0.0和0.3
* 使标签对于鉴别器来说是有噪声的：在训练鉴别器时偶尔会调换标签


** 7：DCGAN /混合模型** 
* 尽可能使用DCGAN。有用！
* 如果您无法使用DCGAN并且没有模型稳定，请使用混合型号：KL + GAN或VAE + GAN


** 8：使用RL的稳定性技巧** 
* 重复性经验
    * 保留过去几代的重复缓冲区并偶尔展示它们
    * 保持G和D过去的检查点，并偶尔交换它们几次迭代
* 所有稳定性技巧都适用于深层确定性策略梯度( policy gradient)
* 见Pfau＆Vinyals（2016）


** 9：使用Adam Optimizer**
* optim.Adam规则！

* 使用SGD作为鉴别器，使用Adam作为生成器


** 10：尽早跟踪故障**
* D损失为0：故障模式
* 检查渐变的规范：如果他们超过100件事情搞砸了
* 当事情正在发挥作用时，D损失的方差很小，并且随着时间的推移而下降，而且存在巨大的差异和尖峰
* 如果生成器的损失稳定下降，那就是用垃圾愚弄D（马丁说）


** 11：不要通过统计平衡损失（除非你有充分的理由）**
* 不要试图找到一个（G的数量/ D的数量）计划来解开训练这很难，我们都尝试过。

* 如果您确实尝试过，请采用原则性方法，而不是直觉
 * 例如
    ```python
    while lossD > A:
      train D
    while lossG > B:
      train G
     ```
     
     
** 12：如果你有标签，请使用它们** 
* 如果您有可用的标签，则训练鉴别器以对样品进行分类：辅助GAN


** 13：向输入添加噪声，随时间衰减**
* 在D的输入上添加一些人为噪声（Arjovsky et.al.Huszar，2016）
    * http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    * https://openreview.net/forum?id=Hk4_qw5xe

* 在生成器的每一层都加入高斯噪声（Zhao et al。EBGAN）
    * 改进的GAN：OpenAI代码也有它（注释掉）


** 14：[notsure]（有时）更多的训练鉴别器**
* 特别是当你有噪音时
* 很难找到D迭代次数与G次迭代的时间表


** 15：[notsure]批量鉴别 ** 
* 结果喜忧参半


** 16：条件GAN中的离散变量**
* 使用嵌入层
* 添加为图像的附加通道
* 保持低维度嵌入和上采样以匹配图像通道大小


** 17：在train和test阶段使用G中的Dropouts **
* 以随机的形式提供噪音（50％）。
* 在训练和测试时间应用于我们的发电机的几个层
* https://arxiv.org/pdf/1611.07004v1.pdf

** 18.其他小技巧 **
* 输入规范化到（-1，1）之间，最后一层的激活函数使用tanh
* 使用wassertein GAN的损失函数
* 学习率不要设置太大，初始1e-4可以参考，另外可以随着训练进行不断缩小学习率
* 给D的网络层增加高斯噪声，相当于是一种正则




****************
************
# 6.GANs的发展:
![](https://blog.mviai.com/images/回顾-GAN/2014.png)
![](https://blog.mviai.com/images/回顾-GAN/2016.png)
![](https://blog.mviai.com/images/回顾-GAN/2017.png)
更多,可以浏览:https://github.com/hindupuravinash
# 7. pytorch实现:
** 目录结构:**
![](https://blog.mviai.com/images/回顾-GAN/m.png)
* 注:从上到下依次是模型保存文件夹->数据集文件夹->生成文件夹


** 设计网络常用计算:**
* 反卷积:
![](https://blog.mviai.com/images/回顾-GAN/反conv.png)
* 卷积:
 ![](https://blog.mviai.com/images/回顾-GAN/conv.png)
 
 

** network.py文件:** 
```python
'''
为了简单明了,所以今天还是创最最简单的GAN

首先 有两个网络 分别是G D

'''
import torch.nn as nn

#我们首先建立G
class G(nn.Module):
    def __init__(self,args):
        super(G,self).__init__()
        ngf = args.ngf  # 生成器feature map数(该层卷积核的个数，有多少个卷积核，经过卷积就会产生多少个feature map)
        self.G_layer= nn.Sequential(
                #输入是一个nz维度的噪声，我们可以认为它是一个1 * 1 * nz的feature map
                nn.ConvTranspose2d(args.nz, 3,5,1,0),# 反conv2d
                nn.BatchNorm2d (3),
                nn.LeakyReLU(True),)
                # 输出大小为3*5*5

    #前向传播
    def forward(self,x):
        out=self.G_layer(x)
        return out

#建立D
class D(nn.Module):
    def __init__(self,args):
        super(D,self).__init__()
        ndf = args.ndf  # 生成器feature map数(该层卷积核的个数，有多少个卷积核，经过卷积就会产生多少个feature map)
        self.D_layer= nn.Sequential(
                # 输入 3 x 5 x 5,
                nn.Conv2d(3,ndf, 3),
                nn.BatchNorm2d (ndf),
                nn.LeakyReLU (True),
                #输出 (ndf)*1*1
                nn.Conv2d (ndf,1,1),
                # 输出 1*0*0
                nn.Sigmoid())#告诉D概率
                
    #前向传播
    def forward(self,x):
        out=self.D_layer(x)
        return out


```

** train.py文件:** 
```python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as  transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# 工具包
import argparse
# 载入网络
from network import G, D

#############参数设置#############
####命令行设置########
parser = argparse.ArgumentParser (description='GAN')  # 导入命令行模块
# 对关于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
#关于训练参数
parser.add_argument ('--batch_size', type=int, default=12,
					 help='训练batch-size大小 (default: 64)')
parser.add_argument ('--imageSize', type=int, default=5,
					 help='图片尺寸')
parser.add_argument ('--max_epoch', type=int, default=5,
					 help='最大迭代数 (default: 5)')
#关于网络参数
parser.add_argument ('--lr_g', type=float, default=2e-4,
					 help='生成器学习率 (default: 2e-4)')
parser.add_argument ('--lr_d', type=float, default=2e-4,
					 help='判别器学习率 (default: 2e-4)')
parser.add_argument ('--ngf', type=int, default=32,
					 help='生成器feature map数')
parser.add_argument ('--ndf', type=int, default=32,
					 help='判别器feature map数')
parser.add_argument ('--d_every', type=int, default=1,
					 help='每几个batch训练一次判别器')
parser.add_argument ('--g_every', type=int, default=2,
					 help='每几个batch训练一次生成器')
parser.add_argument ('--nz', type=int, default=5,
					 help='噪声维度')

#关于优化器参数
parser.add_argument ('--beta1', type=int, default=0.5,
					 help='Adam优化器的beta1参数')
#路径
parser.add_argument ('--dataset', default='data/',
					 help='数据集路径')

parser.add_argument ('--save_data',  default='save/',
					 help='保存路径')

#可视化
parser.add_argument ('--vis', action='store_true',
					 help='是否使用visdom可视化')
parser.add_argument ('--plot_every', type=int, default=1,
					 help='每间隔_batch，visdom画图一次')
# 其他

parser.add_argument ('--cuda', action='store_true',
					 help='开启cuda训练')
parser.add_argument ('--plt', action='store_true',
					 help='开启画图')
parser.add_argument ('--test', action='store_true',
					 help='开启测试生成')
parser.add_argument ('--save_every', type=int, default=3,
					 help='几个epoch保存一次模型 (default: 3)')
parser.add_argument ('--seed', type=int, default=1,
					 help='随机种子 (default: 1)')
args = parser.parse_args ()  # 相当于激活命令


#训练过程
def train():
	###############判断gpu#############
	device = torch.device ('cuda') if args.cuda else torch.device ('cpu')
	
	####### 为CPU设置种子用于生成随机数，以使得结果是确定的##########
	torch.manual_seed (args.seed)
	if args.cuda:
		torch.cuda.manual_seed (args.seed)
	cudnn.benchmark = True
	
	#################可视化###############
	if args.vis:
		vis = Visualizer ('GANs')
	##########数据转换#####################
	data_transforms = transforms.Compose ([transforms.Scale (args.imageSize),  # 通过调整比例调整大小,会报警
										   transforms.CenterCrop (args.imageSize),  # 在中心裁剪给指定大小方形PIL图像
										   transforms.ToTensor (),# 转换成pytorch 变量tensor
										   transforms.Normalize ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	###############数据载入################
	train_dataset = datasets.ImageFolder (root=args.dataset,  # 数据路径目录
										  transform=data_transforms)  # 把数据转换成上面约束样子
	
	# test_dataset = datasets.ImageFolder (root=args.dataset,
	# 									 transform=data_transforms)
	
	##############数据装载###############
	train_loader = torch.utils.data.DataLoader (dataset=train_dataset,  # 装载数据
												batch_size=args.batch_size,  # 设置批大小
												shuffle=True)  # 是否随机打乱
	# test_loader = torch.utils.data.DataLoader (dataset=test_dataset,
	# 										   batch_size=args.batch_size,
	# 										   shuffle=True)
	
	#############模型载入#################
	netG ,netD= G (args),D (args)
	netG.to (device)
	netD.to (device)
	print (netD, netG)
	
	###############损失函数##################
	optimizerD = torch.optim.Adam (netD.parameters (), lr=args.lr_d,betas=(0.5, 0.999))  # Adam优化器
	optimizerG = torch.optim.Adam (netG.parameters (), lr=args.lr_g,betas=(0.5, 0.999))  # Adam优化器
	###############画图参数保存##################
	G_losses = []
	D_losses = []
	img_list = []
	#############训练过程#####################
	import tqdm
	# Tqdm是一个快速，可扩展的Python进度条，可以在Python
	# 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
	# tqdm (iterator)。
	for epoch in range (args.max_epoch):
		for i, (images, labels) in tqdm.tqdm(enumerate (train_loader)):  # 枚举出来
			#数据处理
			images = images.to (device)
			# 装箱
			images = Variable (images)
			noises=Variable(torch.randn(args.batch_size, args.nz, 1, 1).to(device))
			
			#遍历每张图片,并且根据指定的训练机制训练
			if i % args.d_every==0:#满足此条件训练D
				#D前向传播
				optimizerD.zero_grad ()
				#D网络输出
				output_r = netD (images)
				#G网络输出
				noises.data.copy_ (torch.randn (args.batch_size, args.nz, 1, 1))
				fake = netG (noises).detach ()  # 根据噪声生成假图
				#把假图给d
				output_f = netD (fake)
				#D的损失
				#print(fake.size(),output_f.size(),output_r.size())
				D_loss = - torch.mean (torch.log (output_r) + torch.log (1. - output_f))
				#D反向传播
				D_loss.backward ()
				#度量
				D_x = output_r.mean ().item ()
				D_G_z1 = output_f.mean ().item ()
				#D更新参数
				optimizerD.step ()
				
			
			if i % args.g_every==0:#满足此条件训练G
				#G前向传播
				optimizerG.zero_grad ()
				#G网络输出
				fake = netG (noises)  # 根据噪声生成假图
				#把假图给G
				output_f = netD (fake)
				#G的损失
				G_loss = torch.mean (torch.log (1. - output_f))
				#G反向传播
				G_loss.backward ()
				#度量
				D_G_z2 = output_f.mean ().item ()
				#D更新参数
				optimizerG.step ()
				
			###########################################
			##########可视化(可选)#####################
			if args.vis and i % args.plot_every == args.plot_every - 1:
				fake = netG (noises)
				vis.images (fake.detach ().cpu ().numpy () [:64] * 0.5 + 0.5, win='fixfake')
				vis.images (images.data.cpu ().numpy () [:64] * 0.5 + 0.5, win='real')
				vis.plot ('errord', D_loss.item ())
				vis.plot ('errorg', G_loss.item ())
			#######################################
			############打印记录###################
			if i % 1== 0:
				print ('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					   % (epoch, args.max_epoch, i, len (train_loader),
						  D_loss.item (), G_loss.item (), D_x, D_G_z1, D_G_z2))
				########添加画图参数########
				G_losses.append (G_loss.item ())
				D_losses.append (D_loss.item ())
				with torch.no_grad ():
					noises = torch.randn (args.batch_size, args.nz, 1, 1).to (device)
					fake = netG (noises).detach ().cpu ()
				import torchvision.utils as vutils
				img_list.append (vutils.make_grid (fake, padding=2, normalize=True))
	


		#######################################
		############保存模型###################
	
		if (epoch + 1) % args.save_every == 0:
			import torchvision as tv
			# 保存模型、图片
			tv.utils.save_image (fake.data [:64], '%s/%s.png' % (args.save_data, epoch), normalize=True,range=(-1, 1))
			torch.save (netD.state_dict (), 'checkpoints/netd_%s.pth' % epoch)
			torch.save (netG.state_dict (), 'checkpoints/netg_%s.pth' % epoch)
			print('完成%s的模型保存'%epoch)

	#######################################
	############画图###################
	
	if args.plt:
		import matplotlib.pyplot as plt
		import numpy as np
		import torchvision.utils as vutils
		
		plt.figure (figsize=(10, 5))
		plt.title ("GAN")
		plt.plot (G_losses, label="G")
		plt.plot (D_losses, label="D")
		plt.xlabel ("迭代次数")
		plt.ylabel ("损失")
		plt.legend ()
		plt.show ()
		
		# 从数据集加载
		real_batch = next (iter (train_dataset))
		
		# 画出真图
		plt.figure (figsize=(15, 10))
		plt.subplot (1, 2, 1)
		plt.axis ("off")
		plt.title ("真图")
		plt.imshow (np.transpose (
			vutils.make_grid (real_batch [0].to (device) [:64], padding=5, normalize=True).cpu (),
			(1, 2, 0)))
		
		# 画出假图
		plt.subplot (1, 2, 2)
		plt.axis ("off")
		plt.title ("假图")
		plt.imshow (np.transpose (img_list [-1], (1, 2, 0)))
		plt.show ()
				

		

		

@torch.no_grad()#禁用梯度计算
def test():
	#判断Gpu
	device = torch.device ('cuda') if args.cuda else torch.device ('cpu')
	#初始化网络
	netg, netd = netG (args).eval (), netD (args).eval ()
	#定义噪声
	noises = torch.randn (args.batch_size, args.nz, 1, 1).to (device)
	#载入网络
	netd.load_state_dict (torch.load ('checkpoints/netd_%s.pth'%args.max_epoch))
	netg.load_state_dict (torch.load ('checkpoints/netg_%s.pth'%args.max_epoch))
	#设备化
	netd.to (device)
	netg.to (device)
	# 生成图片，并计算图片在判别器的分数
	fake_img = netg (noises)
	scores = netd (fake_img).detach ()
	
	# 挑选最好的某几张
	indexs = scores.topk (5) [1]
	result = []
	for i in indexs:
		result.append (fake_img.data [i])
	
	# 保存图片
	import torchvision as tv
	tv.utils.save_image (torch.stack (result), 5, normalize=True, range=(-1, 1))











	
	###################可视化类##################################
import visdom
import time
import torchvision as tv
import numpy as np


class Visualizer ():
	"""
	封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
	调用原生的visdom接口
	"""
	
	def __init__ (self, env='default', **kwargs):
		import visdom
		self.vis = visdom.Visdom (env=env, use_incoming_socket=False, **kwargs)
		
		# 画的第几个数，相当于横座标
		# 保存（’loss',23） 即loss的第23个点
		self.index = {}
		self.log_text = ''
	
	def reinit (self, env='default', **kwargs):
		"""
		修改visdom的配置
		"""
		self.vis = visdom.Visdom (env=env, use_incoming_socket=False, **kwargs)
		return self
	
	def plot_many (self, d):
		"""
		一次plot多个
		@params d: dict (name,value) i.e. ('loss',0.11)
		"""
		for k, v in d.items ():
			self.plot (k, v)
	
	def img_many (self, d):
		for k, v in d.items ():
			self.img (k, v)
	
	def plot (self, name, y):
		"""
		self.plot('loss',1.00)
		"""
		x = self.index.get (name, 0)
		self.vis.line (Y=np.array ([y]), X=np.array ([x]),
					   win=(name),
					   opts=dict (title=name),
					   update=None if x == 0 else 'append'
					   )
		self.index [name] = x + 1
	
	def img (self, name, img_):
		"""
		self.img('input_img',t.Tensor(64,64))
		"""
		
		if len (img_.size ()) < 3:
			img_ = img_.cpu ().unsqueeze (0)
		self.vis.image (img_.cpu (),
						win=(name),
						opts=dict (title=name)
						)
	
	def img_grid_many (self, d):
		for k, v in d.items ():
			self.img_grid (k, v)
	
	def img_grid (self, name, input_3d):
		"""
		一个batch的图片转成一个网格图，i.e. input（36，64，64）
		会变成 6*6 的网格图，每个格子大小64*64
		"""
		self.img (name, tv.utils.make_grid (
			input_3d.cpu () [0].unsqueeze (1).clamp (max=1, min=0)))
	
	def log (self, info, win='log_text'):
		"""
		self.log({'loss':1,'lr':0.0001})
		"""
		
		self.log_text += ('[{time}] {info} <br>'.format (
			time=time.strftime ('%m%d_%H%M%S'),
			info=info))
		self.vis.text (self.log_text, win=win)
	
	def __getattr__ (self, name):
		return getattr (self.vis, name)
	
	
	
if __name__ == '__main__':
	if args.test:
		test()
	else:
		train()
```
* 输入：
* 训练:
```python
python train.py --cuda --plt
```
* 测试:
```python
python train.py --cuda --test
```
    * `--cuda`表示用GPU
    *  `--plt`表示启动画图功能
    *  `--vis` 表示使用visdom可视化
    *  具体可看参数列表:
    *  ![](/回顾-GAN/q.png)

** 输出：**
![](https://blog.mviai.com/images/回顾-GAN/r.png)
![](https://blog.mviai.com/images/回顾-GAN/r1.png)
![](https://blog.mviai.com/images/回顾-GAN/r2.png)

** 注：为什么结果这样，为了迎合网络，简单我把图片改的不成样 具体可以看参数设置**
