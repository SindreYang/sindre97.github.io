---
title:  fastai-安装
date: 2018-12-15 15:31:15
tags:
    - fastai
categories: 
    - 框架
    - pytorch
    - fastai
   
---


*** 来自官方文档!!!!***
************
***********
#### 前提:
**注意:Python:需要python3.6及其以上**

**fastai需要pytorch1.0以上**
********

*******
#### pytorch1.0可直接安装:
|平台|	|GPU|	|cpu|
|-|-|-|
|Linux|	|直接安装|	|直接安装|
|Mac|  |源码安装|	|直接安装|
|Windows||直接安装|	|直接安装|
<font color=red> 具体更新,请看官方:https://pytorch.org/</font>
***********

# cpu安装:

**`conda` 安装：**
```
 conda install -c pytorch pytorch-cpu torchvision
 conda install -c fastai fastai
 ```
 **`pip` 安装：**
 ```
 pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
 pip install fastai
```
*********

# Gpu安装:
<font color=red> 注意对应cuda版本,以下以9.2为例</font>
**`conda` 安装：**

```
conda install -c pytorch pytorch-nightly cuda92
conda install -c fastai fastai
 ```
 **`pip` 安装：**
 ```
 pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
 pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org
 pip install fastai
```
***********


<font color=red>下图给liunx用户安装cuda提醒:</font>

| CUDA工具包 || NVIDIA（Linux x86_64）显卡驱动 |
|-|-|
| CUDA 10.0 || > = 410.00 |
| CUDA 9.0|	|> = 384.81|
| CUDA 8.0| |> = 367.48|


