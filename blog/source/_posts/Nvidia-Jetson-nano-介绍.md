---
title: Jetson nano 介绍
tags:
  - jetson nano
categories:
  - 工具
  - 嵌入式
  - nvidia
toc: false
date: 2019-06-10 10:52:43
---

1、官网介绍
https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/

![](https://blog.mviai.com/images/n1/1.webp)


良心价，只要99美元。
各模型跑分。


![](https://blog.mviai.com/images/n1/2.webp)


2、相关参数

![](https://blog.mviai.com/images/n1/3.webp)



跟x1对比功耗超低；

![](https://blog.mviai.com/images/n1/4.webp)
3、软件支持
相关内容可以到官网下载
https://developer.nvidia.com/embedded/downloads

![](https://blog.mviai.com/images/n1/5.webp)


有些资源估计的科学下载才行。
国内微雪作为销售商，已经整理了资料更方便下载，百度云盘下载快很多。
http://www.waveshare.net/wiki/Jetson_Nano_Developer_Kit


![](https://blog.mviai.com/images/n1/7.webp)


系统是Ubuntu定制版，有5g多。
5、开箱图片

![](https://blog.mviai.com/images/n1/8.webp)


![](https://blog.mviai.com/images/n1/9.webp)



做工精良，良心机。

6、配件及软件准备
Jetson nano包装只有板子。
6.1.电源
小米充电器，5V 2A，电源线；要保证2A。有5V 2Adc口电源线亦可，但是需要设置跳线屏蔽usb供电口。

![](https://blog.mviai.com/images/n1/pz.png)
![](https://blog.mviai.com/images/n1/pz.gif)
6.2 TF卡
要刷系统到TF卡，最小要求16G。
sd卡插在如图：
![](https://blog.mviai.com/images/n1/sd.png)
6.3 格式化软件
https://www.sdcard.org/downloads/formatter/
6.4 烧录软件
Nvidia官方推荐使用Etcher
![](https://blog.mviai.com/images/n1/10.webp)


支持各种平台。
6.5 显示配件
键盘鼠标、hdmi接口显示器等；
7、烧录系统
通过微雪的百度云分享下载完成后，用sdformater格式化sd卡，然后用ether烧录系统镜像。
![](https://blog.mviai.com/images/n1/11.webp)
![](https://blog.mviai.com/images/n1/12.webp)
![](https://blog.mviai.com/images/n1/13.webp)
烧录要12分钟，12g的系统镜像。


8、启动系统


![](https://blog.mviai.com/images/n1/14.webp)



![](https://blog.mviai.com/images/n1/15.webp)






设置完初始信息即可进入系统，跟Ubuntu操作体验一样。

9、启动系统查看工作状态，温度
```
sudo tegrastats
root@jetson-desktop:~# sudo tegrastats
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [1%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/125
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [1%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35.5C thermal@28.5C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/125
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [0%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35.5C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/111
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [2%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35.5C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/104
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [3%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27C PMIC@100C GPU@29C AO@34.5C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/99
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [0%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@28C PMIC@100C GPU@29C AO@35C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/104
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [2%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@24.5C CPU@28C PMIC@100C GPU@28.5C AO@35C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/101
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [0%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35.5C thermal@28.5C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/104
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [1%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@24.5C CPU@27.5C PMIC@100C GPU@28.5C AO@35C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/101
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [1%@102,1%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35.5C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/104
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [4%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@25C CPU@27.5C PMIC@100C GPU@29C AO@35C thermal@28.25C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 125/105
RAM 724/3965MB (lfb 81x4MB) IRAM 0/252kB(lfb 252kB) CPU [1%@102,0%@102,0%@102,0%@102] EMC_FREQ 4%@204 GR3D_FREQ 0%@153 APE 25 PLL@24.5C CPU@27C PMIC@100C GPU@28.5C AO@35C thermal@28C POM_5V_IN 877/877 POM_5V_GPU 0/0 POM_5V_CPU 83/104

```
开箱介绍完成。

参考：https://www.jianshu.com/p/c9a7635f315c
