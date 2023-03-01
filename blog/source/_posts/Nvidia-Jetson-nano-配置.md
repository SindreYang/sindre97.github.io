---
title: Jetson nano 配置
tags:
  - jetson nano
categories:
  - 工具
  - 嵌入式
  - nvidia
toc: false
date: 2019-06-10 10:52:43
---

通过设置（system settings）-->network-->ipv4是查看dhcp分配ip为192.108.1.105；
jetson nano已经默认开启了openssh-server服务。
可以通过xshell直接连上，用教程一中设置的jetson登录；

![](https://blog.mviai.com/images/n1/n21.webp)

可见Ubuntu版本是18.04.2
为了方便操作，设置初始化root的密码；设置为jetson,具体操作如下；
```
jetson@jetson-desktop:~$ sudo passwd
[sudo] password for jetson: 
Enter new UNIX password: 
Retype new UNIX password: 
passwd: password updated successfully
jetson@jetson-desktop:~$ su
Password: 
root@jetson-desktop:/home/jetson# 


设置允许超级管理员远程访问
# vi /etc/ssh/sshd_config
找到并用#注释掉这行：PermitRootLogin prohibit-password

新建一行 添加：PermitRootLogin yes
```
重启服务
```
# service ssh restart
```
设置固定ip
```
/etc/network/interfaces
# interfaces(5) file used by ifup(8) and ifdown(8)
# Include files from /etc/network/interfaces.d:
source-directory /etc/network/interfaces.d

auto eth0
iface eth0 inet static
address 192.168.1.105
netmask 255.255.255.0
gateway 192.168.1.1



/etc/systemd/resolved.conf
[Resolve]
DNS==8.8.8.8 223.5.5.5
#FallbackDNS=8.8.8.8
#Domains=
#LLMNR=no
#MulticastDNS=no
#DNSSEC=no
#Cache=yes
#DNSStubListener=yes
```

重启再连接；联网正常；
```
root@jetson-desktop:~# ping www.yangxin.com
PING www.a.shifen.com (183.232.231.174) 56(84) bytes of data.
64 bytes from 183.232.231.174 (183.232.231.174): icmp_seq=1 ttl=57 time=5.70 ms
64 bytes from 183.232.231.174 (183.232.231.174): icmp_seq=2 ttl=57 time=5.61 ms
^C
--- www.a.shifen.com ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1002ms
rtt min/avg/max/mdev = 5.617/5.661/5.706/0.087 ms
```

2、默认组件
Jetson nano的镜像已经自带了JetPack，cuda，cudnn，opencv组件和sample，这些例子安装路径如下所示
```
TensorRT    /usr/src/tensorrt/samples/
CUDA    /usr/local/cuda-/samples/
cuDNN   /usr/src/cudnn_samples_v7/
Multimedia API  /usr/src/tegra_multimedia_api/
VisionWorks /usr/share/visionworks/sources/samples/ /usr/share/visionworks-tracking/sources/samples/ /usr/share/visionworks-sfm/sources/samples/
OpenCV  /usr/share/OpenCV/samples/
```
2.0获取超级

```
sudo su
```

2.1核对CUDA
```
root@jetson-desktop:/# nvcc -V
-bash: nvcc: command not found
```
加入路径
```
gedit  ~/.bashrc
```
文件最后加入
```

export CUBA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH

```


执行使生效
```
source ~/.bashrc
```
再次执行,显示cuda10.0
```
root@jetson-desktop:/# source ~/.bashrc
root@jetson-desktop:/# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sun_Sep_30_21:09:22_CDT_2018
Cuda compilation tools, release 10.0, V10.0.166
root@jetson-desktop:/# 
```
2.2核对opencv

```
root@jetson-desktop:/# pkg-config opencv --modversion
3.3.1
```

显示opencv当前版本是3.3.1
2.3核对cuDNN
```
root@jetson-desktop:/# cd /usr/src/cudnn_samples_v7/mnistCUDNN/
root@jetson-desktop:/usr/src/cudnn_samples_v7/mnistCUDNN# make
/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -IFreeImage/include  -m64    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -o fp16_dev.o -c fp16_dev.cu
g++ -I/usr/local/cuda/include -IFreeImage/include   -o fp16_emu.o -c fp16_emu.cpp
g++ -I/usr/local/cuda/include -IFreeImage/include   -o mnistCUDNN.o -c mnistCUDNN.cpp
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -o mnistCUDNN fp16_dev.o fp16_emu.o mnistCUDNN.o -I/usr/local/cuda/include -IFreeImage/include  -LFreeImage/lib/linux/aarch64 -LFreeImage/lib/linux -lcudart -lcublas -lcudnn -lfreeimage -lstdc++ -lm
FreeImage/lib/linux/aarch64/libfreeimage.a(strenc.o): In function `StrIOEncInit':
strenc.c:(.text+0x1294): warning: the use of `tmpnam' is dangerous, better use `mkstemp'


执行sample
root@jetson-desktop:/usr/src/cudnn_samples_v7/mnistCUDNN# chmod a+x mnistCUDNN
root@jetson-desktop:/usr/src/cudnn_samples_v7/mnistCUDNN# ./mnistCUDNN 
cudnnGetVersion() : 7301 , CUDNN_VERSION from cudnn.h : 7301 (7.3.1)
Host compiler version : GCC 7.3.0
There are 1 CUDA capable devices on your machine :
device 0 : sms  1  Capabilities 5.3, SmClock 921.6 Mhz, MemSize (Mb) 3964, MemClock 12.8 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.341979 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.395625 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 5.210573 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 5.213230 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 14.978802 time requiring 57600 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.135000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.170885 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.282708 time requiring 28800 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 1.206094 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 5.214895 time requiring 203008 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!
```

测试通过；
2.4核对python
```
root@jetson-desktop:/# python3
Python 3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

```




默认已经安装python2.7 和python3.6.7版本
3、增加python相关内容
3.1安装pip3
```
root@jetson-desktop:/# apt-get install python3-pip
正在读取软件包列表... 完成
正在分析软件包的依赖关系树       


root@jetson-desktop:/# pip3 -V
pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)

升级下版本到19.1;
root@jetson-desktop:~# python3 -m pip install --upgrade pip
Collecting pip
  Downloading https://files.pythonhosted.org/packages/f9/fb/863012b13912709c13cf5cfdbfb304fa6c727659d6290438e1a88df9d848/pip-19.1-py2.py3-none-any.whl (1.4MB)
    100% |████████████████████████████████| 1.4MB 279kB/s 
Installing collected packages: pip
  Found existing installation: pip 9.0.1
    Not uninstalling pip at /usr/lib/python3/dist-packages, outside environment /usr
Successfully installed pip-19.1

root@jetson-desktop:~# pip3 -V
pip 19.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)

```
3.2安装python-OpenCV
```
root@jetson-desktop:/# sudo apt-get install python3-opencv
正在读取软件包列表... 完成
正在分析软件包的依赖关系树       
正在读取状态信息... 完成       
下列软件包是自动安装的并且现在不需要了：
  apt-clone archdetect-deb busybox-static cryptsetup-bin dpkg-repack gir1.2-timezonemap-1.0 gir1.2-xkl-1.0 grub-common kde-window-manager kinit kio kpackagetool5 kwayland-data kwin-common kwin-data kwin-x11 libdebian-installer4
  libkdecorations2-5v5 libkdecorations2private5v5 libkf5activities5 libkf5attica5 libkf5completion-data libkf5completion5 libkf5declarative-data libkf5declarative5 libkf5doctools5 libkf5globalaccel-data libkf5globalaccel5
  libkf5globalaccelprivate5 libkf5idletime5 libkf5jobwidgets-data libkf5jobwidgets5 libkf5kcmutils-data libkf5kcmutils5 libkf5kiocore5 libkf5kiontlm5 libkf5kiowidgets5 libkf5newstuff-data libkf5newstuff5 libkf5newstuffcore5
  libkf5package-data libkf5package5 libkf5plasma5 libkf5quickaddons5 libkf5solid5 libkf5solid5-data libkf5sonnet5-data libkf5sonnetcore5 libkf5sonnetui5 libkf5textwidgets-data libkf5textwidgets5 libkf5waylandclient5 libkf5waylandserver5
  libkf5xmlgui-bin libkf5xmlgui-data libkf5xmlgui5 libkscreenlocker5 libkwin4-effect-builtins1 libkwineffects11 libkwinglutils11 libkwinxrenderutils11 libqgsttools-p1 libqt5designer5 libqt5help5 libqt5multimedia5 libqt5multimedia5-plugins
  libqt5multimediaquick-p5 libqt5multimediawidgets5 libqt5opengl5 libqt5positioning5 libqt5printsupport5 libqt5qml5 libqt5quick5 libqt5quickwidgets5 libqt5sensors5 libqt5sql5 libqt5test5 libqt5webchannel5 libqt5webkit5 libxcb-composite0
  libxcb-cursor0 libxcb-damage0 os-prober python3-dbus.mainloop.pyqt5 python3-icu python3-pam python3-pyqt5 python3-pyqt5.qtsvg python3-pyqt5.qtwebkit python3-sip qml-module-org-kde-kquickcontrolsaddons qml-module-qtmultimedia
  qml-module-qtquick2 rdate
使用'sudo apt autoremove'来卸载它(它们)。
将会同时安装下列软件：
  gdal-data libaec0 libarmadillo8 libarpack2 libcharls1 libdap25 libdapclient6v5 libepsilon1 libfreexl1 libfyba0 libgdal20 libgdcm2.8 libgeos-3.6.2 libgeos-c1v5 libgeotiff2 libgif7 libgl2ps1.4 libhdf4-0-alt libhdf5-100 libhdf5-openmpi-100
  libjsoncpp1 libkmlbase1 libkmldom1 libkmlengine1 liblept5 libminizip1 libnetcdf-c++4 libnetcdf13 libodbc1 libogdi3.2 libopencv-calib3d3.2 libopencv-contrib3.2 libopencv-core3.2 libopencv-features2d3.2 libopencv-flann3.2
  libopencv-highgui3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 libopencv-ml3.2 libopencv-objdetect3.2 libopencv-photo3.2 libopencv-shape3.2 libopencv-stitching3.2 libopencv-superres3.2 libopencv-video3.2 libopencv-videoio3.2
  libopencv-videostab3.2 libopencv-viz3.2 libpq5 libproj12 libqhull7 libsocket++1 libspatialite7 libsuperlu5 libsz2 libtcl8.6 libtesseract4 libtk8.6 liburiparser1 libvtk6.3 libxerces-c3.2 odbcinst odbcinst1debian2 proj-bin proj-data
  python3-numpy
建议安装：
  geotiff-bin gdal-bin libgeotiff-epsg libhdf4-doc libhdf4-alt-dev hdf4-tools libmyodbc odbc-postgresql tdsodbc unixodbc-bin ogdi-bin tcl8.6 tk8.6 mpi-default-bin vtk6-doc vtk6-examples gfortran python-numpy-doc python3-nose
  python3-numpy-dbg
下列【新】软件包将被安装：
  gdal-data libaec0 libarmadillo8 libarpack2 libcharls1 libdap25 libdapclient6v5 libepsilon1 libfreexl1 libfyba0 libgdal20 libgdcm2.8 libgeos-3.6.2 libgeos-c1v5 libgeotiff2 libgif7 libgl2ps1.4 libhdf4-0-alt libhdf5-100 libhdf5-openmpi-100
  libjsoncpp1 libkmlbase1 libkmldom1 libkmlengine1 liblept5 libminizip1 libnetcdf-c++4 libnetcdf13 libodbc1 libogdi3.2 libopencv-calib3d3.2 libopencv-contrib3.2 libopencv-core3.2 libopencv-features2d3.2 libopencv-flann3.2
  libopencv-highgui3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 libopencv-ml3.2 libopencv-objdetect3.2 libopencv-photo3.2 libopencv-shape3.2 libopencv-stitching3.2 libopencv-superres3.2 libopencv-video3.2 libopencv-videoio3.2
  libopencv-videostab3.2 libopencv-viz3.2 libpq5 libproj12 libqhull7 libsocket++1 libspatialite7 libsuperlu5 libsz2 libtcl8.6 libtesseract4 libtk8.6 liburiparser1 libvtk6.3 libxerces-c3.2 odbcinst odbcinst1debian2 proj-bin proj-data
  python3-numpy python3-opencv
升级了 0 个软件包，新安装了 67 个软件包，要卸载 0 个软件包，有 315 个软件包未被升级。
需要下载 55.8 MB 的归档。
解压缩后会消耗 265 MB 的额外空间。
您希望继续执行吗？ [Y/n] y
```

测试下看看
```
root@jetson-desktop:~# python3
Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> print(cv2.__version__)
3.2.0
>>> 

```
默认3.2版本，和系统本身自带的不统一; 
python2.7版本自带的opencv
```
root@jetson-desktop:~# python
Python 2.7.15rc1 (default, Nov 12 2018, 14:31:15) 
[GCC 7.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>>  print(cv2.__version__)
  File "<stdin>", line 1
    print(cv2.__version__)
    ^
IndentationError: unexpected indent
>>> print(cv2.__version__)
3.3.1
>>> 
```

python3能安装的OpenCV的版本太低;稍后再解决;
4，更新系统


执行命令
```
apt-get update && apt-get upgrade -y
```
```
效果片段如下：下载内容较多，可能时间会很长。
升级了 305 个软件包，新安装了 0 个软件包，要卸载 0 个软件包，有 10 个软件包未被升级。
需要下载 396 MB/396 MB 的归档。
解压缩后会消耗 69.5 MB 的额外空间。
命中:13 http://ports.ubuntu.com/ubuntu-ports bionic InRelease                                             
获取:1 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 login arm64 1:4.5-1ubuntu2 [301 kB]
命中:14 http://ports.ubuntu.com/ubuntu-ports bionic-updates InRelease                                     
命中:15 http://ports.ubuntu.com/ubuntu-ports bionic-backports InRelease                                   
命中:16 http://ports.ubuntu.com/ubuntu-ports bionic-security InRelease                                    
正在读取软件包列表... 完成 20%]                                              80.2 kB/s 1小时 22分 13秒    
获取:2 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libgomp1 arm64 8.3.0-6ubuntu1~18.04 [69.6 kB]                                                                                                                             
获取:3 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libitm1 arm64 8.3.0-6ubuntu1~18.04 [24.3 kB]                                                                                                                              
获取:4 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 gcc-8-base arm64 8.3.0-6ubuntu1~18.04 [18.7 kB]                                                                                                                           
获取:5 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libgcc1 arm64 1:8.3.0-6ubuntu1~18.04 [34.4 kB]                                                                                                                            
获取:6 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 liblsan0 arm64 8.3.0-6ubuntu1~18.04 [121 kB]                                                                                                                              
获取:7 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libtsan0 arm64 8.3.0-6ubuntu1~18.04 [269 kB]                                                                                                                              
获取:8 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libcc1-0 arm64 8.3.0-6ubuntu1~18.04 [46.4 kB]                                                                                                                             
获取:9 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libatomic1 arm64 8.3.0-6ubuntu1~18.04 [9,164 B]                                                                                                                           
获取:10 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libstdc++6 arm64 8.3.0-6ubuntu1~18.04 [372 kB]                                                                                                                           
获取:11 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libnss-systemd arm64 237-3ubuntu10.21 [90.1 kB]                                                                                                                          
获取:12 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libudev1 arm64 237-3ubuntu10.21 [45.2 kB]                                                                                                                                
获取:13 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 udev arm64 237-3ubuntu10.21 [1,050 kB]                                                                                                                                   
获取:14 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libnss-myhostname arm64 237-3ubuntu10.21 [30.2 kB]                                                                                                                       
获取:15 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libpam-systemd arm64 237-3ubuntu10.21 [92.8 kB]                                                                                                                          
获取:17 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libsystemd0 arm64 237-3ubuntu10.21 [171 kB]                                                                                                                              
获取:18 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libpam0g arm64 1.1.8-3.6ubuntu2.18.04.1 [51.1 kB]                                                                                                                        
获取:16 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 systemd arm64 237-3ubuntu10.21 [2,551 kB]                                                                                                                                
获取:19 http://ports.ubuntu.com/ubuntu-ports bionic-updates/main arm64 libpam-modules-bin arm64 1.1.8-3.6ubuntu2.18.04.1 [32.7 kB]
```
默认的更新源国内访问实在太慢了。
/etc/apt/sources.list文件
```
gudit /etc/apt/sources.list 
```
里面的链接全部替换成清华的源
```
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe

然后重新执行;
获取:84 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/main arm64 Packages [1,024 B]                                                                                                                                         
获取:85 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/main Translation-en [448 B]                                                                                                                                           
获取:86 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/universe arm64 Packages [3,468 B]                                                                                                                                     
获取:87 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/universe Translation-en [1,604 B]                                                                                                                                     
获取:88 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/universe arm64 DEP-11 Metadata [7,156 B]                                                                                                                              
获取:89 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/universe DEP-11 48x48 Icons [29 B]                                                                                                                                    
获取:90 http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports bionic-backports/universe DEP-11 64x64 Icons [29 B]                                                                                                                                    
已下载 45.5 MB，耗时 16秒 (2,900 kB/s)    
```
速度超快！
5，关闭桌面系统
先看下内存
```
root@jetson-desktop:~# free
              total        used        free      shared  buff/cache   available
Mem:        4059712      572136     2529292       19044      958284     3304544
Swap:             0           0           0
root@jetson-desktop:~# 
```

关闭桌面
```
root@jetson-desktop:~# sudo systemctl set-default multi-user.target
Removed /etc/systemd/system/default.target.
Created symlink /etc/systemd/system/default.target → /lib/systemd/system/multi-user.target.
root@jetson-desktop:~# reboot
```
再看内存
```
root@jetson-desktop:~# free
              total        used        free      shared  buff/cache   available
Mem:        4059712      321340     3511972       17616      226400     3573020
Swap:             0           0           0
root@jetson-desktop:~#
```
参考链接：https://www.jianshu.com/p/1fac6cdedd0d