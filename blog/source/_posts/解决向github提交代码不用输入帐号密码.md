---
title: 解决向github提交代码不用输入帐号密码
date: 2019-05-04 17:49:34
tags:
    - github
categories:
    - 工具
    - 日常工具
    - github
---
方案一：

在你的用户目录下新建一个文本文件.git-credentials
```
Windows：C:/Users/username

Mac OS X： /Users/username

Linux： /home/username
#注意：鼠标右键新建文件重复命名是成功不了的，需要借助Sublime等IDE工具来创建文件。
```
.git-credentials在文件中输入以下内容：
```
https:{username}:{password}@github.com

#{username}和{password}是你的github的账号和密码
```
修改git配置
执行命令：
```
git config --global credential.helper store
```
上述命令会在.gitconfig文件(.gitconfig与.git-credentials在同目录下)末尾添加如下配置:
```
[user]
    name = alice
    email = alice@aol.com
[credential]
    helper = store
```

经过上述三步配置之后, 你push代码到github时, 便无需再输入用户名密码了

方案二：

在命令行输入命令:
```
git config --global credential.helper store
```
这一步会在用户目录下的.gitconfig文件最后添加：
```
[credential]
    helper = store
    
```
push 代码
 *   push你的代码 (git push), 这时会让你输入用户名和密码, 这一步输入的用户名密码会被记住, 下次再push代码时就不用输入用户名密码!这一步会在用户目录下生成文件.git-credential记录用户名密码的信息。

总结：
* 方案一与方案二都是创建.git-credential文件并写入用户信息，一个是手动创建，一个命令创建。
