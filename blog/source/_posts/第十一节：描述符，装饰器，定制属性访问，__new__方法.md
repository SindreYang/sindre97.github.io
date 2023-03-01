---
title: 第十一节：描述符，装饰器，定制属性访问，__new__方法
tags:
  - 课后解答
url: 241.html
id: 241
categories:
  - 学习
  - 教学
date: 2018-05-21 18:57:23
---

\[warning\]

\_\_new\_\_方法：
--------------

**\_\_new\_\_ 方法是什么？** \_\_new\_\_方法接受的参数虽然也是和\_\_init\_\_一样，但\_\_init\_\_是在类实例创建之后调用，而 \_\_new\_\_方法正是创建这个类实例的方法。 **\_\_new\_\_ 的作用** 依照Python官方文档的说法，\_\_new\_\_方法主要是当你继承一些不可变的class时(比如int, str, tuple)， 提供给你一个自定义这些类的实例化过程的途径。还有就是实现自定义的metaclass。  

定制属性访问:
-------

`object.``__getattr__`（_self_，_name_）[](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "这个定义的永久性")

当默认的属性访问失败，并调用[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")（或者[`__getattribute__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattribute__ "对象.__ getattribute__")引发[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")，因为_名字_是不是一个实例的属性或分类的属性`self`;或[`__get__()`](https://docs.python.org/3/reference/datamodel.html#object.__get__ "对象.__ get__")的_名称_属性提升[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")）。此方法应该返回（计算）的属性值或引发[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")异常。 请注意，如果通过正常机制找到属性，[`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "对象.__ getattr__")则不会调用该属性。（这是[`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "对象.__ getattr__")和之间的故意不对称[`__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__ "对象.__ setattr__")）。这是出于效率原因而完成的，否则[`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "对象.__ getattr__")将无法访问实例的其他属性。请注意，至少在实例变量中，您可以通过在实例属性字典中不插入任何值（而是将它们插入另一个对象中）来伪造完全控制。请参阅[`__getattribute__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattribute__ "对象.__ getattribute__")下面的方法，以实际获得对属性访问的完全控制。

`object.``__getattribute__`（_self_，_name_）[](https://docs.python.org/3/reference/datamodel.html#object.__getattribute__ "这个定义的永久性")

无条件地调用以实现类的实例的属性访问。如果这个类还定义了[`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "对象.__ getattr__")，那么除非[`__getattribute__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattribute__ "对象.__ getattribute__")明确地调用它或引发一个，否则后者将不会被调用[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")。此方法应返回（计算）的属性值或引发[`AttributeError`](https://docs.python.org/3/library/exceptions.html#AttributeError "AttributeError的")异常。为了避免此方法中的无限递归，它的实现应始终调用具有相同名称的基类方法来访问它所需的任何属性，例如。`object.__getattribute__(self,name)`

注意

当通过语言语法或内置函数隐式调用查找特殊方法时，此方法仍可能被忽略。请参阅[特殊方法查找](https://docs.python.org/3/reference/datamodel.html#special-lookup)。

`object.``__setattr__`（_self_，_name_，_value_）[](https://docs.python.org/3/reference/datamodel.html#object.__setattr__ "这个定义的永久性")

在试图进行属性分配时调用。这被称为而不是正常机制（即将值存储在实例字典中）。_name_是属性名称，_value_是要分配给它的值。 如果[`__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__ "对象.__ setattr__")想分配给实例属性，它应该调用具有相同名称的基类方法，例如。`object.__setattr__(self,name,value)`

`object.``__delattr__`（_self_，_name_）[](https://docs.python.org/3/reference/datamodel.html#object.__delattr__ "这个定义的永久性")

像[`__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__ "对象.__ setattr__")但删除属性而不是赋值。这应该只在对对象有意义时才能实现。`delobj.name`

`object.``__dir__`（_self_）[](https://docs.python.org/3/reference/datamodel.html#object.__dir__ "这个定义的永久性")

[`dir()`](https://docs.python.org/3/library/functions.html#dir "DIR")在对象上调用时调用。必须返回一个序列。[`dir()`](https://docs.python.org/3/library/functions.html#dir "DIR")将返回的序列转换为列表并对其进行排序。

   

描述符：
----

一句话概括：描述符就是可重用的属性
-----------------

在这里我要告诉你：从根本上讲，描述符就是可以重复使用的属性。也就是说，描述符可以让你编写这样的代码：

1

2

3

4

f  =  Foo()

b  =  f.bar

f.bar  =  c

del  f.bar

而在解释器执行上述代码时，当发现你试图访问属性（b = f.bar）、对属性赋值（f.bar = c）或者删除一个实例变量的属性（del f.bar）时，就会去调用自定义的方法。    

装饰器:
----

01 什么是装饰器？
----------

装饰器可以让一个Python函数拥有原本没有的功能，也就是你可以通过装饰器，让一个平淡无奇的函数变的强大，变的漂亮。 举几个现实中的例子 1、你一个男的程序员，穿上女装，戴上假发，你就有了女人的外表（穿女装、戴假发的过程就是新的特效，你拥有了女人的外表，你原来的小jj还在，没有消失） 2、你新买的毛坯房，装修，买家具后变好看了（装修、家具就是新的特效） 3、孙悟空被放进炼丹炉装饰了一下，出来后，学会了火眼金睛，以前的本领都还在

02 为什么Python要引入装饰器？
-------------------

因为引入装饰器会便于开发，便于代码复用，可以把烂泥扶上墙， 装饰器可以让你一秒变女人且可以保住小JJ，当你某天后悔想重新变回男人，只要脱掉女装和假发即可（如果你变女人的时候，给小JJ做了手术（直接修改函数体的内容），想再变回男人可就痛苦了哦）

03 装饰器有利于解决哪些问题？
----------------

例子1： 扩展功能 比如你写了一段代码，当你执行 **孙悟空()** 就打印出它目前的技能

    # python3支持用中文做函数名，
    # 这里为了方便你理解，就用中文，实际情况为了兼容性，你可别用中文哦
    
    def 孙悟空():
      print('吃桃子')
    孙悟空()
    # 输出:
    # 吃桃子
    

现在你希望 **孙悟空**这个函数 打印出 **’有火眼金睛了’**，该怎么做呢？ 是的，你可以直接在函数里加一段 **print('有火眼金睛了')** 但是这样会破坏原来的代码，如果你的代码量很多很多的话，修改起来则是灾难， 不过别担心，你还可以用装饰器来装饰他，让他在原本基础上，扩展出新的功能 代码如下

    def 炼丹炉(func): # func就是‘孙悟空’这个函数
      def 变身(*args, **kwargs): #*args, **kwargs就是‘孙悟空’的参数列表，这里的‘孙悟空’函数没有传参数，我们写上也不影响，建议都写上  
          print('有火眼金睛了') # 加特效，增加新功能，比如孙悟空的进了炼丹炉后，有了火眼金睛技能  
          return func(*args, **kwargs) #保留原来的功能，原来孙悟空的技能，如吃桃子
      return 变身 # 炼丹成功，更强大的，有了火眼金睛技能的孙悟空出世
    
    @炼丹炉
    def 孙悟空():
      print('吃桃子')
    
    孙悟空()
    # 输出:
    # 有火眼金睛了
    # 吃桃子
    

例子2：扩展权限认证 比如你的代码，默认打开就播放动画片，代码如下

    def play():
      print('开始播放动画片 《喜洋洋和灰太狼》')
    
    play()
    # 输出
    # 开始播放动画片 《喜洋洋和灰太狼》
    

但是突然某天，你突然希望只允许1岁到10才可以看这个动画片，不希望程序员大叔看这个动画片怎么办？ 是的，你可以修改这个代码，加上年龄限制，但如果我们用装饰器的话，就更简单了，就可以不用破坏原来的代码，而且方便扩展到其他函数上

    userAge = 40
    
    def canYou(func):
      def decorator(*args, **kwargs):
          if userAge > 1 and userAge < 10:
              return func(*args, **kwargs)
          print('你的年龄不符合要求，不能看')
      return decorator
    
    @canYou
    def play():
      print('开始播放动画片 《喜洋洋和灰太狼》')
    
    play()
    # 输出
    # 你的年龄不符合要求，不能看
    # 你可以修改上面的 userAge 为9 试试
    

你看，是不是很简单，实际情况中，很多时候，你需要对一段代码加上权限认证，加上各种功能；但是又不想，或者不方便破坏原有代码，则可以用装饰器去扩展它

04 装饰器背后的实现原理是什么？
-----------------

原理 代码逆推后如下

    def 炼丹炉(func): 
      def 变身(*args, **kwargs):  
          print('有火眼金睛了') 
          return func(*args, **kwargs) 
      return 变身 
    
    def 孙悟空():  
      print('吃桃子')
    新_孙悟空 = 炼丹炉(孙悟空) #放入原料，原来的弱小的孙悟空，生成炼丹方案给 新_孙悟空 ，这里也可以把炼丹方案给 原来的‘孙悟空’，为了方便理解，给了新的孙悟空 
    
    新_孙悟空() # 执行炼丹程序，新的孙悟空出世
    

然后这段代码，写起来有点麻烦，Python官方出了一个快捷代码，也就是语法糖，用了语法糖就变成了下面这样

    def 炼丹炉(func): 
      def 变身(*args, **kwargs):  
          print('有火眼金睛了') 
          return func(*args, **kwargs) 
      return 变身 
    
    @炼丹炉  # 把下面的 ‘孙悟空’ 塞进炼丹炉，并把新的孙悟空复制给下面的函数
    def 孙悟空():  
      print('吃桃子')
    
    孙悟空() # 执行炼丹程序，新的孙悟空出世
    

可以一次性在一个函数上用多个装饰器吗？ 当然可以，下面我们给孙悟空，弄个金箍棒，让他学会72变，学会飞

    def 炼丹炉(func):
      def 变身(*args, **kwargs):
          print('有火眼金睛了')
          return func(*args, **kwargs)
      return 变身
    
    def 龙宫走一趟(func):
      def 你好(*args, **kwargs):
          print('有金箍棒了')
          return func(*args, **kwargs)
      return 你好
    
    def 拜师学艺(func):
      def 师傅(*args, **kwargs):
          print('学会飞、72变了')
          return func(*args, **kwargs)
      return 师傅
    
    @拜师学艺
    @龙宫走一趟
    @炼丹炉  
    def 孙悟空():
      print('吃桃子')
    
    孙悟空()
    # 输出
    # 学会飞、72变了
    # 有金箍棒了
    # 有火眼金睛了
    # 吃桃子
    

上面代码的等效于 **拜师学艺(龙宫走一趟(炼丹炉(孙悟空)))** 代码的执行顺序是 **先从内到外** **先执行 炼丹炉，然后是龙宫走一趟，最后是拜师学艺，\[/warning\]**

链接：https://www.zhihu.com/question/26930016/answer/360300235
