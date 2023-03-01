---
title: '教程1:OpenCV代码转换为Web API'
tags:
  - cv
categories:
  - 框架
  - opencv
toc: false
date: 2020-03-08 23:59:03
---


# 目标:
1. 我们将创建一个Web API，允许用户调用OpenCV代码。
2. 快速建立一个简单的示例。
3. 只需要一个Web浏览器，即可在所有平台上运行。
4. 该项目将是免费的！将注册一个免费帐户，并使用一个开源框架。
5. 将从一个非常基本的示例开始，在该示例中，用户将输入图像url传递到后端。后端读取图像并返回其宽度和高度。

# 工具:
1. [pythonanywhere](http://pythonanywhere.com/)
-	PythonAnywhere不仅仅是一个托管平台。这是用于编写python代码的功能强大的IDE。它使您可以通过Web浏览器访问带有语法高亮显示的代码编辑器，Unix终端，访问日志文件。当然，您可以轻松地从github转移现有代码，也可以根据需要在vi中转移代码。它还附带安装了OpenCV！ 
2. [web2py](http://www.web2py.com/)
-	免费的开源全栈框架，用于快速开发快速，可伸缩，安全和可移植的数据库驱动的基于Web的应用程序。用Python编写和编程


# 步骤:
## 注册
1. 注册PythonAnywhere并安装web2py(初学者的帐户是免费的,记住您的用户名)
2. 注册并登录后，转到“ Web”选项卡，然后添加一个新的Web应用程序。
![image.png](https://blog.mviai.com/images/FtkXBQuiEWGkMdxevcEKmDn9CMtu)
3. 选择web2py作为您的python框架
![image.png](https://blog.mviai.com/images/Fn_6VoDczoUUyfrD8EXNl905_BN3)
4. 选择web2py的管理员密码。(记住这是管理员密码,不是登录密码)
![image.png](https://blog.mviai.com/images/Fnuaih8Gp2gs7Wk_U0c3xYhvkyd6)
![image.png](https://blog.mviai.com/images/FhFWGHLwvkJIMXPlh2F_SQ2IDYQE)

## 在web2py上创建Web应用
1. 打开一个新标签，然后转到位于web2py的管理界面(https://你的用户名.pythonanywhere.com/admin/default/index)

![image.png](https://blog.mviai.com/images/FiG0_2rlroCh3LksbPbVUJfzesrF)
![image.png](https://blog.mviai.com/images/Fn5vKEq3PIUUt0npOq7uhUuX62cw)

2. 登陆后,检查是否有如下文件:(出现sindre目录)
![image.png](https://blog.mviai.com/images/FitcbVJmwkj5ONrpzEp2SvuIQj9P)


## 将OpenCV代码添加到web2py

web2py目录结构:
- 模型包含应用程序的所有数据和规则
- 控制器包含用于处理数据的代码
- 视图则显示基础数据的某些状态

![image.png](https://blog.mviai.com/images/FvmAUQVa7r2anLef1CKARrVwICK0)


TODO: 输入图像URL-->返回图像的宽度和高度

在控制器 controllers / default.py最后添加以下代码
![image.png](https://blog.mviai.com/images/FrZI70EK0hcjb3ff631SR0d8JvX2)


```python
# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：learnopencv -> web
@IDE    ：CLion
@Author ：sindre
@Date   ：2020/3/8 下午7:41
==================================================
'''


__author__ = 'sindre'

import cv2
import numpy as np
import urllib2
import json

def image_dimensions():
    # 伪装成Mozilla，因为一些web服务器可能不喜欢python机器人
    hdr = {'User-Agent': 'Mozilla/5.0'}
    
    # 设定要求
    req = urllib2.Request(urllib2.request.vars.url, headers=hdr)

    try:

        # 获取URL的内容

        con = urllib2.urlopen( req )

        # 读取内容并将其转换为numpy数组

        im_array = np.asarray(bytearray(con.read()), dtype=np.uint8)

        #将numpy数组转换为图像。

        im =  cv2.imdecode(im_array, cv2.IMREAD_GRAYSCALE)

        # 获取图像的宽度和高度。

        height, width = im.shape

        #将宽度和高度封装在一个对象中，并返回经过编码的JSON

        return json.dumps({"width" : width, "height" : height})



    except urllib2.HTTPError as e:

        return e.fp.read()



```
保存后就可以测试了(注意,点 save file旁的图标才是保存)

## 测试
随机找个图片地址
如:https://home.yx1024.top/images/1.png

curl -F url=https://home.yx1024.top/images/1.png http://你的用户名.pythonanywhere.com/你的应用名/default/image_dimensions




