---
title: '第一部分:GitKraken破解及教程'
tags:
  - GItKraken
categories:
  - 工具
  - 日常工具
  
toc: false
date: 2020-03-01 19:35:41
---

# 1.GItKraken介绍
gitKraken，这款工具操作比较方便，UI也是我喜欢的风格，对没有太多git使用经验的新手比较友好，学习成本相对较低。尤其喜欢的一点就是它的分支和提交非常清晰。
![image.png](https://blog.mviai.com/images/Fs9HOjVYzYWRDsUM1XmDcoq7LKc7)
**优点:**
1.            可以在不切换分支的情况下，操作其他的分支
2.     	   多平台(Windows,liunx,mac)


**缺点:**
1.	       启动慢,占用资源多
2.    	   收费


# 2.下载
[官网下载](https://www.gitkraken.com/download)


# 3.汉化
虽然都能认识,但是第一次接触,汉化亲切些;

## 下载
```
git clone https://github.com/k-skye/gitkraken-chinese.git

```

## 原理

通过修改软件目录下english语言对应的一个json文件内容来完成汉化目的

## 操作步骤
**(注意备份英文文件,备份为strings.json.bak,习惯了操作就改回来)**
1. 将项目中的 `strings.json` 替换到 GitKraken 语言目录下的 `strings.json`.  
(实际目录可能会不一样,但文件名一定是strings.json)
  
   - Windows: `%程序安装目录%\gitkraken\app-x.x.x\resources\app\src\strings.json` (x.x.x 是你的GitKraken版本)
   - Mac: `/Applications/GitKraken.app/Contents/Resources/app/src/strings.json`
   - Linux: `/usr/share/gitkraken/resources/app.asar.unpacked/src` (感谢lyydhy 10.31补充 Gitkraken是deepin 通过deb 安装的)
     
2. 重启GitKraken.


# 4.破解
1.安装完成后(sudo dpkg -i xx.deb),打开软件
2.打开网址[破解器](https://github.com/KillWolfVlad/GitKraken-AUR),查看是否版本对应,否则查看问题回答最多(里面有解决方案)
3.我下载的更新分支(github.com/BoGnY/GitCracken)的,它支持6.5
4.看说明:如下
-		GNU/Linux不要用snap安装
-	        macOS必须开启后在破解(实测liunx也需要)
- 	        node.js 版本需要12及其以上(sudo n 选择)
-  		yarn必须安装		
 		完成上述几条,执行
```c
yarn install
 yarn build
node dist/bin/gitcracken.js --help
		

```
 **  如遇到下载问题,首先查看是否默认换到淘宝源,如遇到报错,点击下载链接是否可以远程下载,如无法,打开yarn.lock文件修改地址(可以尝试本地地址 file://xxxxx) **

	
-	完成后开始破解
```c	
		yarn run gitcracken patcher
		yarn run gitcracken patcher --asar ~/Downloads/gitkraken/resources/app.asar(修改为你的安装地址)
		yarn run gitcracken patcher backup unpack patch

```

		
**注意:如果报错,用yarn run gitcracken patcher remove 删除,然后到sudo rm gitkraken/resources/app.asar.*(如果破解后启动不了,也用这个方法)**

### 提示:防止更新
	**更新后还要收费,修改hosts文件**
	vim etc/hosts
	然后在文件最后加入
```c
0.0.0.0 release.gitkraken.com#屏蔽官方地址

```

![image.png](https://blog.mviai.com/images/Fia7aodyxl3vw3G3WU151wVAl2dH)
