---
title: 回顾-instaGAN
date: 2019-2-21 13:40:56
tags:
    - instaGAN

categories: 
    - 框架
    - pytorch
    - GAN
---

# 1.InstaGAN简介：
[InstaGAN](https://openreview.net/forum?id=ryxwJhC9YX)这种 GAN 结合了实例信息（如目标分割掩码），提高了多实例转换的能力。在保持实例置换不变性的同时，该 GAN 对图像和相应的实例属性集进行转换。为此，研究人员引入了一个语境保留损失函数，鼓励网络学习目标实例之外的恒等函数。此外，他们还提出了一种序列 mini-batch 推理/训练技术，这种技术借助有限的 GPU 内存处理多个实例，增强了该网络在多实例任务中的泛化能力。
*********
是Cycle GAN的改进版本。虽然Cycle GAN替换不能改变形状，但它可以改变形状。但老实说，这种印象是准确度和分辨率仍然很低
***********

# 2. instaGANGAN网络结构:
如图:
![](https://blog.mviai.com/images/instaGAN/network.png)

* 首先看图（左边）
	* 这是全局图，可以看到依然是cycleGAN模式，最大特点就是原图与mask进行合并（cat）输入和输出
*（右边）
	* 大体上我们可以把G看做![](https://blog.mviai.com/images/instaGAN/g.png)，d看做![](https://blog.mviai.com/images/instaGAN/d.png)
	* 看到原图与mask合并后，还加了一个add模块 ，即为了获得属性，我们对所有集合元素的不变性求和，然后将其与等方差的恒等映射连接起来。
	* 中间f，表示用通过f函数提取特征。
# 3.instaGAN损失函数
其实就是把加一个约束，保证masks之外的信息保持不变（下式中的L_{ctx}项）。整个loss如下：
![](https://blog.mviai.com/images/instaGAN/loss.png)

对比下cycleGAN的损失函数
![](https://blog.mviai.com/images/instaGAN/closs.png)

然后在分开看下instaGAN的loss：
![](https://blog.mviai.com/images/instaGAN/inloss.png)

* 解释下：
	* LSGAN损失函数时最基本的GAN网络损失函数
    * cycle-consistency loss cyclegan网络的损失函数，为保证图像翻译过程中映射关系一一对应
    * identity mapping loss 也是内容损失函数保证图像翻译前后内容不变
    * context preserving loss ，也是保证在实例翻译时图像的背景内容不发生变化。
	
# 4 创新点
1、提出了一个多属性图像翻译的网络结构，且属性之间的顺序是任意的
2、提出了context preserving loss来鼓励网络去学习一个目标示例的之外的一个恒等映射
3、提出一种sequential mini-batch的方法顺序生成mini-batchs的属性，而不是在整个数据集上做一次。
	* sequential mini-batch其实就是渐进迭代的方法把mask分开，一个个生成就好了。但是这里要注意，通常小的mask放到后面效果会更好。因为迭代进行的话，每次生成的图片都会被改变，后面如果是大mask，很容易把前面生成的小mask跟淹没掉。
所以，使用这种办法，一定程度上破坏了object mask之间的时序不变性，如果GPU显存足够，就不要用这种方法了。




