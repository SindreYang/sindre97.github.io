---
title: 回顾-pix2pix
tags:
  - pix2pix
categories: 
    - 框架
    - pytorch
    - GAN
toc: false

date: 2018-12-25 03:38:57
---

# 1.Pix2Pix 简介:
* 1.这是基于CGAN的pix2pix模型!
* 2.Pix2Pix与一般GAN不同的地方在于，其实现的目标是图像翻译，A-》B，比如，一张场景可以转换为RGB全彩图，也可以转化成素描，也可以转化为灰度图。
* 顾名思义，Pix2Pix指的是像素对像素的翻译，图像大小保持不变。G网络的输入是A图像，通过G网络生成的图像叫做FakeB，而真实的图像就是RealB。
* 论文: https://arxiv.org/abs/1611.07004
* 项目：https://github.com/phillipi/pix2pix

# 2.Pix2Pix 网络架构:
** <font color=red> 生成网络: </font>**
生成器G使用U-net实现:
![](https://blog.mviai.com/images/回顾-pix2pix/g.png)
* 1. 从图中可以看出u-net采用跳跃式链接的全卷积结构(有点像resnet的跨层链接)

U-net来源于VAE的encoder-decoder:
如图;
![](https://blog.mviai.com/images/回顾-pix2pix/u.png)
**注:U-Net通过卷积和反卷积实现的U形形状的网络结构；输入和输出都是3个维度**


** <font color=red> 判别网络: </font>**
判别器D使用马尔科夫性的判别器(PatchGAN)([论文](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf))
* 简单来说:PatchGAN可以理解为一种风格/纹理损失网络
![](https://blog.mviai.com/images/回顾-pix2pix/d.png)
** 注:根据kernelsize是1的进行将为，最终实现1个chanel的图像**

** <font color=red> 损失函数: </font>**
* 原始GAN(G不需要x):
![](https://blog.mviai.com/images/回顾-pix2pix/gan.png)
* CGAN(G需要x):
![](https://blog.mviai.com/images/回顾-pix2pix/cgan.png)
* 给GAN加个L1或L2,而Pix2pix使用的是L1架构，可以减少模糊程度:
![](https://blog.mviai.com/images/回顾-pix2pix/l1.png)
* 最终G损失:
![](https://blog.mviai.com/images/回顾-pix2pix/g_f.png)
** 注:最终的loss是针对一个chanel的图像的每个像素求MSE(均方差loss，fake的label为0，real的label为1)或BCE(二进制交叉熵)。**


** <font color=red> 训练方法: </font>**
* ![](https://blog.mviai.com/images/回顾-pix2pix/train.png)
* 训练大致过程如上图所示。图片 x 作为此cGAN的条件，需要输入到G和D中。G的输入是{x,z}（其中，x 是需要转换的图片，z 是随机噪声），输出是生成的图片G(x,z)。D则需要分辨出{x,G(x,z)}和{x,y}。


# 3.Pix2Pix 创新点:
* 1.一般的方法都是训练CNN去缩小输入跟输出的欧氏距离,论文在GAN的基础上提出一个通用的方法：pix2pix 来解决这一类问题。通过pix2pix来完成成对的图像转换
* 2.输入为图像而不是随机向量
* 3.成对输入为图像而不是随机向量
* 4.Patch判别器来降低计算量提升效果
* 5.L1损失函数的加入来保证输入和输出之间的一致性。


# 4. Pytorch实现:
