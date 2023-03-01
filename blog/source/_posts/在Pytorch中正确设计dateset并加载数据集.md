---
title: 在Pytorch中正确设计dateset并加载数据集
date: 2019-05-07 20:40:50
tags:
        - 加载数据集
categories:
 - 框架
 - pytorch
 - 数据加载
    
---


# 一、前言
在构建深度学习任务中，最重要的当然是如何设计我们的神经网络。

但在实际的训练过程中，如何正确编写、使用加载数据集的代码同样是不可缺少的一环，在不同的任务中不同数据格式的任务中，加载数据的代码难免会有差别。为了避免重复编写并且避免一些与算法无关的错误，我们有必要讨论一下如何正确加载数据集。

这里只讨论如何加载图像格式的数据集，对于文字或者其他的数据集不进行讨论。

# 二、正确加载数据集
加载数据集是深度学习训练过程中不可缺少的一环。一般地，在平常的时候，我们第一个想到的是将所有需要的数据聚成一堆一堆然后通过构建list去一一读取我们的数据：
![](http:https://blog.mviai.com/images/dateset并加载数据集/1.png)
假如我们编写了上述的图像加载数据集代码，在训练中我们就可以依靠<font color='#ff0000'>get_training_data()</font>这个函数来得到batch_size个数据，从而进行训练，乍看下去没什么问题，但是一旦我们的数据量超过1000：

* 将所有的图像数据直接加载到numpy数据中会占用大量的内存
* 由于需要对数据进行导入，每次训练的时候在数据读取阶段会占用大量的时间
* 只使用了单线程去读取，读取效率比较低下
* 拓展性很差，如果需要对数据进行一些预处理，只能采取一些不是特别优雅的做法

既然问题这么多，到底说回来，我们应该如何正确地加载数据集呢？

本文将会介绍如何根据Pytorch官方提供的数据加载模板，去编写自己的加载数据集类，从而实现高效稳定地加载我们的数据集。(Pytorch官方教程介绍)

# 三、Dataset类
<font color='#ff0000'>Dataset</font>类是Pytorch中图像数据集中最为重要的一个类，也是Pytorch中所有数据集加载类中应该继承的父类。其中父类中的两个私有成员函数必须被重载，否则将会触发错误提示：
```python
def getitem(self, index):
def len(self):
```
其中<font color='#ff0000'>  ` __len__  ` </font>应该返回数据集的大小，而<font color='#ff0000'>`__getitem__`</font>应该编写支持数据集索引的函数，例如通过<font color='#ff0000'>`dataset [i]`</font>可以得到数据集中的第<font color='#ff0000'> i+1</font>个数据。

![](https://blog.mviai.com/images/dateset并加载数据集/2.png)

上面所示的这个类，其实也就是起到了封装我们加载函数的作用(对象处理起来更加方便明朗么)，在继承了这个Dataset类之后，我们需要实现的核心功能便是<font color='#ff0000'>  `__getitem__() ` </font>函数，<font color='#ff0000'>  `__getitem__() ` </font>是Python中类的默认成员函数，我们通过实现这个成员函数实现可以通过索引来返回图像数据的功能。

那么怎么得到图像从而去返回呢？当然不会直接将图像数据加载到内存中，相反我们只需要得到图像的地址就足够了，然后在调用的时候通过不同的读取方式读取即可。

关于读取方式：https://oldpan.me/archives/pytorch-transforms-opencv-scikit-image

定义自己的数据集类
那么我们开始定义一个自己的数据集类吧。

首先继承上面的<font color='#ff0000'>` dataset ` </font>类。然后在<font color='#ff0000'>`__init__() ` </font>方法中得到图像的路径，然后将图像路径组成一个数组，这样在<font color='#ff0000'>`__getitim__() ` </font>中就可以直接读取：
```python
# 假设下面这个类是读取船只的数据类
class ShipDataset(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """
    def __init__(self, root, augment=None):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(root) if
            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        self.augment = augment   # 是否需要图像增强

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        return open_image(self.image_files[index])

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)
        
```
如果我们需要在读取数据的同时对图像进行增强的话，可以在<font color='#ff0000'>  `__getitem__(self, index)`</font>函数中设置图像增强的代码，例如：
```python
    def __getitem__(self, index):
        if self.augment:
            image = open_image(self.image_files[index])
            iamge = self.augment(iamge)  # 这里对图像进行了增强
            return image
        else:
            # 如果不进行增强，直接读取图像数据并返回
            # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
            return open_image(self.image_files[index])
            
```
当然，图像增强的方法可以使用Pytorch内置的图像增强方式，也可以使用自定义或者其他的图像增强库。这个很灵活，当然要记住一点，在Pytorch中得到的图像必须是<font color='#ff0000'>  `tensor`</font>，也就是说我们还需要再修改一下<font color='#ff0000'>`__getitem__(self, index)`</font>：
```python
    def __getitem__(self, index):
        if self.augment:
            image = open_image(self.image_files[index])
            iamge = self.augment(iamge)  # 这里对图像进行了增强
            return to_tensor(image)      # 将读取到的图像变成tensor再传出
        else:
            # 如果不进行增强，直接读取图像数据并返回
            # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
            return to_tensor(open_image(self.image_files[index]))
            ```
这样，一个基本的数据类就设计好了。

## DataLoader类
之前所说的<font color='#ff0000'>  `Dataset `</font>类是读入数据集数据并且对读入的数据进行了索引。但是光有这个功能是不够用的，在实际的加载数据集的过程中，我们的数据量往往都很大，对此我们还需要一下几个功能：

* 可以分批次读取：batch-size
* 可以对数据进行随机读取，可以对数据进行洗牌操作(shuffling)，打乱数据集内数据分布的顺序
* 可以并行加载数据(利用多核处理器加快载入数据的效率)

这时候就需要<font color='#ff0000'>  `Dataloader`</font>类了，<font color='#ff0000'> `Dataloader`</font>这个类并不需要我们自己设计代码，我们只需要利用<font color='#ff0000'> `DataLoader`</font>类读取我们设计好的<font color='#ff0000'>` ShipDataset`</font>即可：
```python
# 利用之前创建好的ShipDataset类去创建数据对象
ship_train_dataset = ShipDataset(data_path, augment=transform)
# 利用dataloader读取我们的数据对象，并设定batch-size和工作现场
ship_train_loader = DataLoader(ship_train_dataset, batch_size=16, num_workers=4, shuffle=False, **kwargs)

```

这时候通过<font color='#ff0000'> `ship_train_loader`</font>返回的数据就是按照batch-size来返回特定数量的训练数据的tensor，而且此时利用了多线程，读取数据的速度相比单线程快很多。

我们这样读取:
```python
for image in train_loader:

        image = image.to(device)  # 将tensor数据移动到device当中
        optimizer.zero_grad()
        output = model(image)     # model模型处理(n,c,h,w)格式的数据，n为batch-size
        ...
        
```
读取数据的基本模式就是这样，当然在实际中不可能这么简单，我们除了图像数据可能还有json、csv等文件需要我们去读取配合图像完成任务。但是原理基本都是一样的，具体复杂点的例子可以查看官方的例程介绍，这里就不赘述了。

# 创建自己的数据集
除了设计读取数据集的代码，我们实际的图像数据应该怎么去放置呢？

一般来说，我们自己制作的数据集一般包含三个部分：***train***、***val***和***test***，我们一般放在三个文件夹中，然后利用代码读取。这样是最舒服最方便的了。

但是因为某些原因，我们得到的数据集却不是这样放好的，比如只有一个文件夹，所有文件都放里头了。或者好几个trian的文件夹需要我们去合并。

当然，如果数据集很小的话(例如小于1000个)，那就无所谓了，直接打开文件夹移动就行，但是如果数据为10W以上级别。直接打开文件夹移动文件那电脑会直接卡死(内存32G，6核处理器依旧卡顿)。那么怎么去整体我们的数据，让代码可以顺利训练数据放去训练？

这里有两种方式。

## 1、自己写脚本移动这些文件
这里以Linux为例，linux下为.sh脚本文件，window则为bat文件。

将下面的脚本代码保存为<font color='#ff0000'> `mm.sh`</font>(随便起的)，自己修改<font color='#ff0000'> `path/from/ `</font>和<font color='#ff0000'> `path/to/ `</font>的地址，tail后面为移动文件的数量。
```
for file in $(ls path/from/ -p | grep -v / | tail -100)
do
mv $file path/to/
done
```
如果移动过程中遇到下面的问题，试着改编权限再来一次。
```
mv: cannot stat '03c5d57c0.jpg': No such file or directory
```

## 2、编写代码灵活读取train、val、以及test文件夹中的数据
之前所说的读取方式ShipDataset类仅仅支持一个文件夹的读取，但是我们得到的只是一个文件夹里面包含了我们采集的数据，但是这些数据有比较多(比如50G)，也不好进行移动分成三份(训练集、验证集和测试集)，这时我们需要自己设计编写代码去实现这些功能。

至于如何去编写，大家可以阅读fastai的源代码去理解一下基本思路(很好的思路，可以好好借鉴下)，fastai是一个包装了Pytorch的快速深度学习开发库：https://oldpan.me/archives/fastai-1-0-quick-study


本文转自：https://oldpan.me/archives/how-to-load-dataset-in-correctly-pytorch
