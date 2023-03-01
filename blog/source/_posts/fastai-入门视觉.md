---
title: fastai-入门视觉
date: 2018-12-15 16:04:01
tags:
    - fastai
categories: 
    - 框架
    - pytorch
    - fastai
    
---

<font color=red> 来自官方文档!!</font>

## 视觉库概览:
1.visionfastai库的模块包含定义数据集和训练计算机视觉任务模型的所有必要功能。它包含四个不同的子模块来实现该目标：
*************
2.vision.image包含Image对象的基本定义以及在后台使用的所有函数，以将转换应用于此类对象。
******
3.vision.transform 包含我们可用于数据扩充的所有变换。
********
4.vision.data包含定义ImageClassificationDataset以及实用功能，以轻松构建DataBunch计算机视觉问题。
**********
5.vision.learner 允许您使用预训练的CNN骨干构建和微调模型，或从头开始训练随机​​初始化的模型。
*********

```python

#首先，从fastai库导入您需要的所有内容。
from fastai.vision import*  #导入计算机视觉库
from fastai  import *        #导入常用库

#
# 首先，创建一个包含MNIST子集的数据文件夹
# ，data/mnist_sample使用这个小帮助程序，为您下载它：


path=untar_data(URLs.MNIST_SAMPLE)
#print(path)

#由于下载文件家包含标准文件夹train和valid文件夹，并且每个文件夹包含一个文件夹
# ，因此您可以DataBunch在一行中创建：

data=ImageDataBunch.from_folder(path)

# 复习：
# from_folder：从imagenet风格数据集创建
# path与train，valid，test子文件夹
# （或提供valid_pct）
# “ Imagenet风格 ”数据集看起来像这样（请注意，测试文件夹是可选的）：
# path\
#   train\
#     clas1\
#     clas2\
#     ...
#   valid\
#     clas1\
#     clas2\
#     ...
#   test\



#加载一个预训练模型（从vision.models）准备进行微调：

learn=create_cnn(data,models.resnet18,metrics=accuracy)


#完成所有工作 开始训练
learn.fit(1)
```
****************
*** ok  就这么暴力,完了!!!
让我想起了sklearn!***
***************

### 下面来回头看下:

```python

#######################################
######回   顾###############
##########################
#首先来看最重要数据长什么样
#print(data)

#可以通过获取相应的属性来访问该训练集和验证集
ds = data.train_ds
# print(ds)
#
```

```python
# 我们顺便看看vision.image，它定义了Image类。
# 我们的数据集将在索引时返回Image对象。
from fastai.vision import Image
import matplotlib.pyplot as plt
img,label=ds[0]
#print(img)

#换种方式显示图片

img.show(figsize=(2, 1), title='Little ')

#同时 你还可以改变它
#如旋转
img.rotate(35)
```

```python
#下面来看看数据增强方式
# vision.transform让我们进行数据扩充。
# 最简单的方法是从一组标准变换中进行选择，
# 其中默认值是为图片设计的

#print(help(get_transforms))
#创建你想要的列表

tfms=[rotate(degrees=(-20,20)),symmetric_warp(magnitude=(-0.3,0.3))]

print(tfms)

#可以使用apply tfms方法将这些变换应用于图像

fig,axes=plt.subplots(1,4,figsize=(8,2))
print(fig,axes)
for ax in axes:
    ds[0][0].apply_tfms(tfms).show(ax=ax)


# 您可以使用转换后的训练和验证数据加载器在单个步骤中
# 创建一个数据库，并传入一个元组（train_TFMS，valid_TFMS）

data=ImageDataBunch.from_folder(path,ds_tfms=(tfms,[]))
print(data)
```

```python
##############################################3
#下面来看训练过程
# 现在你已准备好训练一个模型。要创建模型，
# 只需将您的DataBunch和模型创建函数
# （例如vision.models或torchvision.models提供的函数）
# 传递给create_cnn，并调用fit：

learn=create_cnn(data,models.resnet18,metrics=accuracy,callback_fns = ShowGraph)

learn.fit(100)

#接下来我们看一下最不正确的图像，以及分类矩阵。


interp=ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))


interp.plot_confusion_matrix()

# 要简单地预测新图像的结果（类型为image，
# 例如用open image打开），
# 只需使用learn.predict。它返回类，它的索引
# 和每个类的概率。
img=learn.data.train_ds[0][0]
print(learn.predict(img))
```

** <font color=blue>要显示图像记得添加(我是在最后直接显示:</font>
```
plt.show()
```








