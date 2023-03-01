---
title: 小玩具-VIP视频解析界面
date: 2018-12-14 02:21:38
tags:
    - pyqt5
categories:
    - 界面
    - pyqt

---

## 需要下列工具:
1. QT-designer(用于快速构造界面)
2. QT-pyuic(用于把刚才界面转换成py文件,方便设计)
3. main.py空文件(用于储存页面逻辑)
4. url.txt(用于储存解析网址)
5. 背景图一张(可选)
****
### 第一步:
    直接打开QT-designer拖动组件设计出如下界面:
    
![](https://blog.mviai.com/images/小玩具-VIP视频解析界面/ui.png)
    
    
### 第二步:
    用Qt-pyuic (python文件里面的Scripts) 将设计好的转换成.py文件
    如在生成.ui文件下 运行命令
   ```
    pyuic5 -o 文件名.py 文件名.ui
    ```
    
 注: 如果找不到pyuic5 看是否将Scripts加入了环境变量
 
 
 然后可以看到生成对应的.py文件
 我是生成ui.py,如图:
```python
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.setEnabled(True)
        widget.resize(416, 667)
        widget.setFixedSize(widget.width(), widget.height())
        self.label = QtWidgets.QLabel(widget)
        self.label.setGeometry(QtCore.QRect(30, 10, 191, 17))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(widget)
        self.lineEdit.setGeometry(QtCore.QRect(0, 35, 301, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(widget)
        self.label_2.setGeometry(QtCore.QRect(0, 60, 391, 21))
        self.label_2.setObjectName("label_2")
        self.pushButton_run = QtWidgets.QPushButton(widget)
        self.pushButton_run.setGeometry(QtCore.QRect(310, 30, 89, 31))
        self.pushButton_run.setObjectName("pushButton_run")
        self.label_3 = QtWidgets.QLabel(widget)
        self.label_3.setGeometry(QtCore.QRect(90, 170, 211, 20))
        self.label_3.setObjectName("label_3")


        self.gridLayoutWidget = QtWidgets.QWidget(widget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 200, 401, 451))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        names=['优酷','土豆','爱奇艺','芒果','乐视','腾讯','搜狐','PPTV','360视','暴风影音','M1905','咪咕视频',
        '音悦台','哔哩哔哩','华数TV','网易公开课','新浪视频','范特西','M3U8','私有云','韩国DAUM'	,'品善网','开眼视频'
        ,'优米网','好看视频','美拍','2MM','凤凰视频','Naver','糖豆网','秒拍','快手','17173','梨视频','中国蓝','第一视频'
        ,'爱拍视频','汽车之家','ECHOMV','东方头条','今日头条','阳光宽频','西瓜视频','酷6视频','CCTV央视','27盘','91广场舞',
        '爆米花','火猫直播','酷狗MV','酷狗MV','QQ音乐MV','酷狗直播','酷狗LIVE','天天看看','激动网','斗鱼视频','斗鱼直播',
        '虎牙视频','虎牙直播','熊猫星颜','熊猫直播','战旗视频','战旗直播','龙珠视频','龙珠直播','来疯直播','触手视频','触手直播','花椒直播','花椒视频'
        ,'全民直播','全民视频','CC直播','CC视频','印客直播','YY神曲','YY回放','YY小','一直播','NOW直播' ]

        posittions=[(i,j)for  i in range(19) for j in range(5)]

        for posittions,name in zip(posittions,names):
            label=QtWidgets.QLabel(name)
            label.setStyleSheet("font:10pt;color:rgb(0,0, 255);font-weight:40px;")
            self.gridLayout.addWidget(label,*posittions)
        self.gridLayout.setContentsMargins(0, 1, 2, 0)
        self.gridLayout.setObjectName("gridLayout")



        self.pushButton_switch = QtWidgets.QPushButton(widget)
        self.pushButton_switch.setGeometry(QtCore.QRect(100, 110, 171, 31))
        self.pushButton_switch.setObjectName("pushButton_switch")

        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "VIP会员-新"))
        self.label.setText(_translate("widget", "视频网页最上面网址(链接):"))
        self.label_2.setText(_translate("widget", "复制VIP视频的地址到↑↑↑上栏中点击立即播放就行了"))
        self.pushButton_run.setText(_translate("widget", "开始播放"))
        self.pushButton_run.setStyleSheet("color:rgb(255,0,0)")
        self.label_3.setText(_translate("widget", "↓↓现支持以下免费播放↓↓"))
        self.pushButton_switch.setText(_translate("widget", "如果不能播放用力点我"))
        self.pushButton_switch.setStyleSheet("background: rgb(0,191,255);color:rgb(128,0,0)")
        self.pushButton_run.setStyleSheet ("background: rgb(0,191,255);color:rgb(128,0,0)")

```


##### 如果想看看界面,可以创建test.py,输入
```python
#继承窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_widget()
        self.ui.setupUi(self)
        
if __name__ =='__main__':
    app = QApplication (sys.argv)
    win1 = MainWindow ()# 创建实例
    win1.show ()
    app.exec_ ()

 ```
 
 
 
 ## 第三步:
 有了界面,还需要个播放窗口!
 为了简单,直接使用了浏览器当窗口!
 
 ![](https://blog.mviai.com/images/小玩具-VIP视频解析界面/20181214031720070.png)
 代码如下:
 ```python
 class ScreenWindow (QMainWindow):
    def __init__ (self,url):
        super (ScreenWindow, self).__init__ ()
        self.setWindowTitle('vip视频影院')
        self.setGeometry(5,30,1355,730)

        self.browser=QWebEngineView()
        self.browser.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)  # 支持视频播放
        self.browser.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        self.browser.settings().setAttribute(QWebEngineSettings.FullScreenSupportEnabled, True)
        # self.browser.page().fullScreenRequested.connect(self._fullScreenRequested)

        self.browser.load (QUrl (url))

        self.setCentralWidget(self.browser)
    #
    # def _fullScreenRequested(request):
    #     request.accept()
    #     w.showFullScreen()


    def screendisplay(self):
        if not self.isVisible ():
            self.show ()

 
 ```
 
 
 ** 注:中间那三行,为了加载flash插件,不然播放不了视频
 最后一行,为了让用户点击触发显示函数
 传入参数`url` 是为了让用户切换解析接口
 url来自下面↓
 **
 
 ### 最后一步:
 
 ```python
 class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_widget()
        self.ui.setupUi(self)
        url=self.switch()
        #print(url)


        self.setIcon()
        self.ui.pushButton_run.clicked.connect (self.TwoT)
        self.ui.pushButton_switch.clicked.connect (self.TwoT)

    def TwoT(self):
        url = self.switch ()
        self.win2 = ScreenWindow (url)
        self.win2.screendisplay()

    def setIcon (self):
        palette1 = QPalette ()

        # palette1.setColor(self.backgroundRole(), QColor(192,253,123))   # 设置背景颜色.scaled(self.width(), self.height()
        palette1.setBrush (self.backgroundRole (), QBrush (QPixmap ('d.png')))  # 设置背景图片
        self.setPalette (palette1)
        self.setAutoFillBackground(True) # 不设置也可以


        self.setWindowIcon (QIcon ('d.png'))


    def addurl(self):
        url = self.ui.lineEdit.text ()
        return url

    def switch(self):
        countmax = len (open ('url', 'r').readlines ())
        count=random.randint(0,countmax-1)
        if count >= countmax-1:
            count = random.randint(0,countmax-1)
        else:
            count += 1
        with open ('url', 'r') as f:
            vurl = f.readlines ()
            vurl = vurl [count].replace ('\n', '')
            qurl = vurl + self.addurl()
            #print(qurl)
        return qurl

if __name__ =='__main__':
    #argvs = sys.argv
    # 支援flash
    #argvs.append('--ppapi-flash-path=./pepflashplayer.dll')
    app = QApplication (sys.argv)
    win1 = MainWindow ()
    win1.show ()
    app.exec_ ()
```
     
 
 **注: 初始化函数`__init__`中 :继承窗口界面,两个触发函数
 `TWoT`:触发函数触发事件(运行切换函数,生成播放窗口)
 `setIcon`:设置界面背景
 `addurl`:读取用户输入url
 `switch`:随机取url文件里的解析口和用户输入url组成新的url
 
 URL文件:
 ```
 https://jx.lache.me/cc/?url=
Https://al.lache.me/vip/?url=
https://2wk.com/vip.php?url=
http://api.bbbbbb.me/jx/?url=
https://www.myxin.top/jx/api/?url=
http://www.syhbyl.tw/jx/api/?url=
https://vip.hackmg.com/jx/index.php?url=
http://jx.wslmf.com/?url=
http://api.52xmw.com/?url=
http://yun.baiyug.cn/vip/index.php?url=
https://jx.lache.me/cc/?url=
Https://al.lache.me/vip/?url=
https://jx.lache.me/cc/?url=
Https://al.lache.me/vip/?url=

```
 最后创建窗口,启动!
 ![](https://blog.mviai.com/images/小玩具-VIP视频解析界面/20181214032135019.png)
 ****
 ***注:无法观看,请下载flashplay (虽然已被淘汰)
 over***
 ***
 
 ***想要一键运行:
 创建run.bat文件
 输入:
```
python main.py
```
保存   就可以点击运行
***
 
## github:  [项目地址点我](https://github.com/sindre97/PyQt5_Toy)


