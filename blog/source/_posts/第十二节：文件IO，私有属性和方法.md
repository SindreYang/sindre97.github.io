---
title: 第十二节：文件IO，私有属性和方法
tags:
  - 课后解答
url: 244.html
id: 244
categories:
 - 学习
 - 教学
date: 2018-05-23 20:08:42
---



一；文件io
------

### 基本操作：

文件的存储方式
-------

*   计算机中，文件是以二进制的方式保存的
*   文本文件就是可以使用文本编辑器查看，二进制文件无法使用文本编辑器查看，是提供给其他软件使用的，例如图片，音视频等

操作文件的套路
-------

1、打开文件open

*   open函数负责打开文件， 并返回文件对象
*   打开文件的方式有很多种，常用的如下：
    *   r 只读，默认模式，如果文件不存在，抛出异常
    *   w 只写，如果文件存在，则覆盖，不存在，则创建
    *   a 追加，如果文件存在，指针会放在文件的结尾，不存在，创建新文件并写入
    *   rb 以二进制读取内容
    *   wb 以二进制写入内容 开发中更多的时候会以只读，只写的方式来操作文件 2、读写文件read，write
*   read方法一次性读入并返回文件的所有内容，执行后，文件指针会移动到文件的末尾
*   readline方法可以一次读取一行内容
*   方法执行后，会把文件指针移动到下一行，准备再次读取，读取大文件时，使用此方法在while循环中，依次读取，节约内存 3、关闭文件close
*   close，如果忘记关闭文件，会造成系统资源消耗，且会影响到后续对文件的访问

**文件指针**

*   文件指针标记从哪个位置开始读取数据
*   第一次打开文件时，通常文件指针会指向文件的开始位置，当执行read后，文件指针移动到末尾
*   在同一个python文件中，如果执行了read，那么再次使用此方法时，时无法获得内容的，可以使用seek方法改变指针位置

文件/目录的常用管理操作
------------

在python中，使用代码实现文件目录操作，需要导入os模块 **文件操作** os.rename(源文件名，目标文件名) os.remove（文件名） **目录操作** os.listdir 目录列表，类似ls os.mkdir 创建目录，和linux一致 os.rmdir 删除目录 os.chdir 修改工作目录 os.getcwd 获取当前工作目录current work directory os.path.isdir（文件路径） 判断是否是目录 os.path.isfile 判断是否是文件

### python 文件操作seek() 和 telll()  

file.seek()方法格式： seek(offset,whence=0)   移动文件读取指针到制定位置 offset:开始的偏移量，也就是代表需要移动偏移的字节数。 whence： 给offset参数一个定义，表示要从哪个位置开始偏移；0代表从文件开头算起，1代表开始从当前位置开始算起，2代表从文件末尾开始算起。当有换行时，会被换行截断。 seek（）无返回值，故值为None   tell() : 文科文件的当前位置，即tell是获取文件指针位置。 readline(n):读入若干行，n代表读入的最长字节数。 readlines() :读入所有行的内容 read读入所有行的内容

### 上下文管理器

普通版： def A1(): f=open("out.txt","w") f.write("123") f.close() 威胁：如果调用异常，资源卡住，无法释放     升级版： def A1(): f=open("out.txt","w") try： f.write("123") except IOError： print（“error”） finally： f.close() 优雅版： def A1（）： with open（“out.txt”，“w”） as f： f.write（“123”）

### 优雅的With as语句

Python提供了With语句语法，来构建对资源创建与释放的语法糖。给Database添加两个魔法方法：

'''python

class  Database(object):


def \_\_enter\_\_(self):

self.connect()

return  self

def \_\_exit\_\_(self,  exc_type,  exc_val,  exc_tb):

self.close()
'''
然后修改handle_query函数如下：



def handle_query():

with Database()  as  db:

print  'handle ---',  db.query()
'''
在Database类实例的时候，使用with语句。一切正常work。比起装饰器的版本，虽然多写了一些字符，但是代码可读性变强了

io模块
----

StringIO顾名思义就是在内存中读写str。 要把str写入StringIO，我们需要先创建一个StringIO，然后，像文件一样写入即可：

    from io import StringIO
    f = StringIO()
    print(f.write('hello py1 '))  # 10
    print(f.write('hello py2 '))  # 10
    print(f.write('hello py3 '))  # 10
    print(f.getvalue()) # hello py1 hello py2 hello py3

要读取StringIO，可以用一个str初始化StringIO，然后，像读文件一样读取：

    from io import StringIO
    f = StringIO('Hello!\nHi!\nGoodbye!')
    print(f.read())

StringIO操作的只能是str，如果要操作二进制数据，就需要使用BytesIO

    from io import BytesIO
    f = BytesIO() 
    print(f.write('中文'.encode('utf-8')))
    # 请注意，写入的不是str，而是经过UTF-8编码的bytes
    
    print(f.getvalue()) # b'\xe4\xb8\xad\xe6\x96\x87'

* * *

BytesIO
=======

和StringIO类似，也可以用一个bytes初始化BytesIO，然后，像读文件一样读取：

    from io import StringIO
    f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
    f.read()
    b'\xe4\xb8\xad\xe6\x96\x87'

* * *

StringIO和BytesIO是在内存中操作str和bytes的方法，使得和读写文件具有一致的接口。  

os模块
----

`os.getcwd() 获取当前工作目录，即当前python脚本工作的目录路径`

`os.chdir(``"dirname"``)  改变当前脚本工作目录；相当于shell下cd`

`os.curdir  返回当前目录: (``'.'``)`

`os.pardir  获取当前目录的父目录字符串名：(``'..'``)`

`os.makedirs(``'dirname1/dirname2'``)    可生成多层递归目录`

`os.removedirs(``'dirname1'``)    若目录为空，则删除，并递归到上一级目录，如若也为空，则删除，依此类推`

`os.mkdir(``'dirname'``)    生成单级目录；相当于shell中mkdir dirname`

`os.rmdir(``'dirname'``)    删除单级空目录，若目录不为空则无法删除，报错；相当于shell中rmdir dirname`

`os.listdir(``'dirname'``)    列出指定目录下的所有文件和子目录，包括隐藏文件，并以列表方式打印`

`os.remove()  删除一个文件`

`os.rename(``"oldname"``,``"newname"``)  重命名文件``/``目录`

`os.stat(``'path/filename'``)  获取文件``/``目录信息`

`os.sep    输出操作系统特定的路径分隔符，win下为``"\\",Linux下为"``/``"`

`os.linesep    输出当前平台使用的行终止符，win下为``"\t\n"``,Linux下为``"\n"`

`os.pathsep    输出用于分割文件路径的字符串`

`os.name    输出字符串指示当前使用平台。win``-``>``'nt'``; Linux``-``>``'posix'`

`os.system(``"bash command"``)  运行shell命令，直接显示`

`os.environ  获取系统环境变量`

`os.path.abspath(path)  返回path规范化的绝对路径`

`os.path.split(path)  将path分割成目录和文件名二元组返回`

`os.path.dirname(path)  返回path的目录。其实就是os.path.split(path)的第一个元素`

`os.path.basename(path)  返回path最后的文件名。如何path以／或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素`

`os.path.exists(path)  如果path存在，返回``True``；如果path不存在，返回``False`

`os.path.isabs(path)  如果path是绝对路径，返回``True`

`os.path.isfile(path)  如果path是一个存在的文件，返回``True``。否则返回``False`

`os.path.isdir(path)  如果path是一个存在的目录，则返回``True``。否则返回``False`

`os.path.join(path1[, path2[, ...]])  将多个路径组合后返回，第一个绝对路径之前的参数将被忽略`

`os.path.getatime(path)  返回path所指向的文件或者目录的最后存取时间`

`os.path.getmtime(path)  返回path所指向的文件或者目录的最后修改时间`

4.sys模块

`sys.argv           命令行参数``List``，第一个元素是程序本身路径`

`sys.exit(n)        退出程序，正常退出时exit(``0``)`

`sys.version        获取Python解释程序的版本信息`

`sys.maxint         最大的``Int``值`

`sys.path           返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值`

`sys.platform       返回操作系统平台名称`

`sys.stdout.write(``'please:'``)`

`val ``=``sys.stdin.readline()[:``-``1``]`

shutil模块：https://docs.python.org/3/library/shutil.html?highlight=shutil#module-shutil

私有属性和方法
-------

如果有一个对象，当需要对其进行修改属性时，有2种方法： （1）对象名.属性名=数据---->直接修改 （2）对象名.方法名()----->间接修改 为了更好的保障属性安全，不能随意修改，一般处理方式为： （1）将属性定义为私有属性 （2）添加一个可以调用的方法，供调用，也就是间接调用属性 私有方法是不能直接调用的
