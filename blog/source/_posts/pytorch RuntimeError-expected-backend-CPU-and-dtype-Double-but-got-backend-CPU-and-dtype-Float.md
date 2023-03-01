---
title: pytorch expected backend CPU and dtype Double but got backend CPU and dtype Float
date: 2019-01-25 03:38:57
tags:
    - 爬坑
categoies:
    - 框架
    - pytorch
    - 爬坑
---

### 用pytorch搭建网络测试时，代码报错如下：
RuntimeError: expected backend CPU and dtype Double but got backend CPU and dtype Float

***
报错在transform.ToTensor ()：
***
如图：
![](http:/images.yx1024.top/爬坑/totensor.png)

报类型错误，默认x类型为float64，加上那句运行正确。


****
尝试过其他方法：


1：
如A fix would be to call .double() on your model (or .float() on the input)
https://github.com/pytorch/pytorch/issues/2138

2：

from_numpy().float()

3:
astype('float')

4:
您的输入和目标张量是DoubleTensors，但您的模型参数是FloatTensors。您必须转换输入或参数。

要将输入转换为float（推荐）：

inputs, labels = data
inputs = inputs.float()
labels = labels.float()
inputs, labels = Variable(inputs), Variable(labels)
要将模型转换为double：

model = ColorizerNet()
model.double()
我建议使用浮动而不是双打。它是PyTorch中的默认张量类型。在GPU上，浮点计算比双重计算快得多。
https://discuss.pytorch.org/t/problems-with-weight-array-of-floattensor-type-in-loss-function/381




