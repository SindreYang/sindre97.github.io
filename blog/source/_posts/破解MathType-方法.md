---

title: 破解MathType-方法
date: 2018-12-24 03:38:57
tags:
    - 破解MathType
categoies:
    - 工具
    - 日常工具

---
# 流程：
1.首先打开运行，快捷键 win+R
2.输入regedit。
3.打开注册表
4.在HKEY_CURRENT_USER项右键 点击查找
5.输入Install Options
6.删除键值对

# 快捷方式：
新建记事本：
输入
```
@echo on

reg delete "HKEY_CURRENT_USER\Software\Install Options\Options7.2" /f

pause

```
注意： reg delete后面 路径每个电脑不同

保存 新建文档.txt(没有txt，请打开后缀）  重命名 为刷新mathtype.bat  每次到期 运行下就可以了
