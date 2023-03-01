---
title: The size of tensor a (5) must match the size of tensor b (3) at non-singleton dimension 3
date: 2018-12-25 03:38:57
tags:
    - 爬坑
categoies:
    - 框架
    - pytorch
    - 爬坑
---

******
打印所有操作的张量形状，然后您将更好地了解Tensor形状如何随每个操作而变化。之后，可以找出要更改的行以使尺寸匹配。

*********
主要在你的输入尺寸  输出尺寸  模型结构方面考虑 
*******
用.size（）排错
