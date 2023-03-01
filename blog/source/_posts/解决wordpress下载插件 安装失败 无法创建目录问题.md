---
title: 解决wordpress下载插件 安装失败 无法创建目录问题
tags:
  - wordpress
url: 87.html
id: 87
categories:
  - 网站
  - wordpress
date: 2018-04-05 21:03:58
---

\[warning\]

本文只针对liunx
----------

mac可以简单直接更改文件权限，windows 直接获取管理员权限 用户名组名为 www-data(大家可能不太一样）而此时wordpress用户组为root，这样就不能创建目录了，具体原因大家可以查阅linux相关知识。 我们在default目录下 输入ls -l wordpress (wordpress目录具体地址)  就可以看到用户组了，下面是未修改的用户和用户组，都是root ![](https://blog.mviai.com/images/wp-content/uploads/2018/04/20170128222146285-300x200.png) 如果不知道自己用户组是www-data还是www ，网上教程都是www，于是自己直接在外面创建文件，然后赋予权限 ，最后ls -l 展示下 发现如图 ![](https://blog.mviai.com/images/wp-content/uploads/2018/04/2018-04-07-14-01-41屏幕截图-300x86.png) 自己全是www-data 说明自己是www-data用户组   然后修改wordpress目录下整体权限（我是直接把wordprss解压到www下的） ![](https://blog.mviai.com/images/wp-content/uploads/2018/04/2018-04-07-14-02-34屏幕截图-300x183.png) 然后大功告成 \[/warning\]
