
---
title: RuntimeError output with shape [1,x,x] doesn't match the broadcast shape [3, x, x]
date: 2019-5-15 13:40:56
tags:
    - 爬坑

categories:
     - 框架
     - pytorch


---
在用pytorch导入图片数据时出现了上述错误，
报错程序：
```python
transform = transforms.Compose([
	transforms.ToTensor(), 
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 
```
报错原因：这是因图片里面有是灰度图像，只有一个通道，而上面的transforms.Normalize 却对三个通道都归一化了，这肯定会报错，所以只要像下面修改即可：
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))# 第一个是均值，第二个是标准差，
	
    ])
```
参考：https://blog.csdn.net/qq_31829611/article/details/90200694 
