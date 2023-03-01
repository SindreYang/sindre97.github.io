---
title: '第二部分:GitKraken破解及教程'
tags:
  - GItKraken
categories:
  - 工具
  - 日常工具
  - GItKraken

toc: false
date: 2020-03-02 00:20:48
---

# 1. 首次打开程序
第一次打开GitKraken程序时， GitKraken会提示需要登陆，可以用github.com的账号登陆，或者用邮箱创建账号登陆,选择第一个可以直接与github关联.
![image.png](https://blog.mviai.com/images/FrcP9jLjMr_CgDGrsjsdJ8NlGb1v)

# 2. Authentication（授权）
我们一般用到比较多的情况是从服务器上clone一个已有的仓库。在clone服务器上的仓库前，首先需要设置/生成本地的加密解密密钥。

打开 GitKraken 的File => Preference => Authentication
![image.png](https://blog.mviai.com/images/FvjIJjj0Hk_to2a1r7iGY9u32HQE)


#  3. 修改用户名
为了方便项目中代码的管理，需要重新编辑用户名。

点击右上角的图像即可看到所示的下拉菜单，鼠标悬于Profile上，会出现一个Edit按钮。
File => Preference =>Profile
![image.png](https://blog.mviai.com/images/FgRQNecQrAXNKw7E1BWn2_RsjOJW)
![image.png](https://blog.mviai.com/images/Fj8t_E76fjLstmtCGRMT4YPJDkS_)

# 4. 初始化本地仓库
如果你需要在自己本地代码的目录下新建一个Git仓库，点击左上角的 File => Init Repo，点击 浏览 按钮选择需要初始化仓库的目录，然后点击 创建储存库 即会出现如下图所示的界面。
![image.png](https://blog.mviai.com/images/FkUktZPr7tiQnzxoupU1mhV6HjDE)
- 图中的.gitignore 和 License 都是可选的。.gitignore文件是git在提交时会读取的一个文件，现在可以不用管它的作用是什么（如果项目是一个python工程，我们可以选用预设好的python.gitignore）。当然如果有兴趣对此深入了解的话，建议去看一看progit这本书，这本书详细了介绍了Git。
![image.png](https://blog.mviai.com/images/Fvhp9AXiQtUPcoeBRPX6hQleo85a)

# 5.克隆服务器上的项目
首先，返回主界面，点击File => Clone Repo，选择 使用URL克隆，如下图：
![image.png](https://blog.mviai.com/images/FoQC1UYejtdMgOc81L1Th6DK2FXZ)
![image.png](https://blog.mviai.com/images/FkWdJIilK-vOsERSv3CjGs43hnZz)

# 6. 打开现有的Git仓库
点击左上角 File ==> open repo 
![image.png](https://blog.mviai.com/images/FlFcOyn1ibS4sN8cQS4XOqVWafKN)

# 7. Kraken界面
## 主界面
![image.png](https://blog.mviai.com/images/ForYMNJhqPEOmBfXUjHXvzEGAv84)


## 功能界面
![image.png](https://blog.mviai.com/images/Fn_PQHqWmEa7BsuVEXikecCeJl7W)
	-  最上面的 本地 下拉菜单中显示的是本地的分支。
	-  远程 下拉菜单中显示的是远程仓库的列表，点击其中一个远程仓库，就会显示该仓库中拥有的分支数（远程分支）。

**可以通过程序上方的  按钮将本地的分支上传到服务器。（非管理员无法删除服务器上的主分支）**

	- 标签 下拉菜单中显示的是本地的标签，需要推送到服务器才能分享标签。
	- 子模块  表示当前仓库的子模块

## 提交记录区域
![image.png](https://blog.mviai.com/images/FqStc-VtZh7x9omCdq9ZLedzp6Oz)
每一行都表示一个提交记录


## 文件改动区域
当文件修改后:
![image.png](https://blog.mviai.com/images/Fnv7vGLCSuCOXPDpnIXRe7ihHX0o)

未暂存相当于草稿,暂存后就可以提交了

## 顶部操作区域
![image.png](https://blog.mviai.com/images/FlhZdanFFZofwWUdoTxT_4vVIIqE)

相当于
1. Undo（回退一个提交记录）；
2. Redo（回到回退前的那个提交记录）；
3. Push（将本地的提交记录同步到服务器上）；
4. Pull（将服务器上的提交记录同步到本地）；
5. Branch（新建一个分支）；


# 8. 提交代码
修改了某个文件后，在程序右侧会出现已修改文件的列表
![image.png](https://blog.mviai.com/images/Fq9z9IPUqvRRxr5Kr2lg8qA5PEdd)
点击文件,就可以查看差异:
![image.png](https://blog.mviai.com/images/Fg309KGh3ZB5At1Z2qmBbr7hZa4R)

![image.png](https://blog.mviai.com/images/FlCV-IMb-SwJ9HvxSOEo_yQBT1lB)


**如果在未暂存区域(草稿):**
**还可以通过比较,选择性保留及删除**
点击  暂存 按钮将会暂存这一块修改的内容，保留绿色部分（即保留+2 ~ +4 行的内容，丢弃 -2 ~ -4 行的内容），

 点击 丢弃 将会丢弃掉改动的这一部分，保留红色的部分（保留-2 ~ -4 行的内容，丢弃 +2 ~ +4 行的内容）。
![image.png](https://blog.mviai.com/images/FtfRJ5sRpAFB78eYBISWuWmy78NC)

## 放弃本次文件的改动
有些情况下，由于更改代码造成了编译无法通过等错误时，想要放弃这次对文件的修改，将文件还原成上一次提交后的状态，一种简单的恢复文件的方法就是，在未暂存列表中找到这个文件，右键点击，选择丢弃改动就行.(如果要放弃全部更改,点击放弃按钮就行)
 
## 修改提交记录的描述信息
![image.png](https://blog.mviai.com/images/FlgOKf1sZsakOMhEZNsIzKb55Yrc)
![image.png](https://blog.mviai.com/images/Fq8R_ci9YYNkWwUMir0g1AsSYoWW)

## 查看文件的历史修改及其追责
![image.png](https://blog.mviai.com/images/FjSWsB6MC9hQt_4_S6WFfI53nQdt)
右键点击,选择历史记录或文件追责
-	历史记录(显示每次提交记录与前次提交记录的差异；)
-	![image.png](https://blog.mviai.com/images/FkPLaQHa3xI8gDDG2T-0i1FEKITf)
-	追责(显示该次提交记录完成的文件内容)
-	![image.png](https://blog.mviai.com/images/FqUMFYPXrveBgpnJTPMDLC3jquHP)


# 8. 本地分支和标签
## 在提交记录区中查看分支状态
![image.png](https://blog.mviai.com/images/FpxYma901gcOm1gb2P3juFhFWMEh)

## 创建本地（Local）分支
在GitKraken中央区域的提交记录处右键点击,选择中间的 在此处创建分支;
![image.png](https://blog.mviai.com/images/FhRkHe4QGm3dPa4LraULfcFO6pRY)
![image.png](https://blog.mviai.com/images/FnB61ZRHeCGormgZrC5nHXZPS07s)

##  切换本地（Local）分支
左侧有勾的表明该分支是当前所在的分支
![image.png](https://blog.mviai.com/images/FnB61ZRHeCGormgZrC5nHXZPS07s)

  *直接在本地分支列表中双击  可以切换至该分支*

*参考:https://www.cnblogs.com/brifuture/p/9052512.html*
