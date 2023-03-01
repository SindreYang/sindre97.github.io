---
title: KDD CUP99数据集预处理
date: 2019-05-04 17:30:17
tags:
    - kdd99
catagories:
    - 框架
    - 数据处理
    - kdd99
---
-----
***       KDD CUP99数据集预处理  ***

-----
 

1、数据集下载：http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

2、KDDCup99网络入侵检测数据集介绍：

https://blog.csdn.net/com_stu_zhang/article/details/6987632

https://www.cnblogs.com/gongyanc/p/6703532.html

3、Weka进阶——基于KDD99数据集的入侵检测分析：

https://blog.csdn.net/jbfsdzpp/article/details/44099849

4、符号型特征数值化

采用one-hot方法进行数值化：https://blog.csdn.net/qq_28617019/article/details/79717184

 

5、KDD CUP99数据集预处理

（1）字符型特征转换为数值型特征（即符号型特征数值化）
Python3对KDD CUP99数据集预处理代码实现（仅实现字符型特征转为数值型特征）
# kdd99数据集预处理
 将kdd99符号型数据转化为数值型数据

```python
# coding:utf-8

import numpy as np
import pandas as pd
import csv
import time

global label_list  # label_list为全局变量

list_big=[] #储存大数据
# 定义kdd99数据预处理函数
def preHandel_data ():
	source_file = 'kddcup.data_10_percent_corrected'
	handled_file = 'kddcup.data_10_percent_corrected.csv'
	data_file = open (handled_file, 'w', newline='')  # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
	with open (source_file, 'r') as data_source:
		csv_reader = csv.reader (data_source)
		csv_writer = csv.writer (data_file)
		count = 0  # 记录数据的行数，初始化为0
		for row in csv_reader:
			temp_line = np.array (row)  # 将每行数据存入temp_line数组里
			list_big.append (int(temp_line [4]))
			list_big.append (int(temp_line [5]))
			temp_line [1] = handleProtocol (row)  # 将源文件行中3种协议类型转换成数字标识
			temp_line [2] = handleService (row)  # 将源文件行中70种网络服务类型转换成数字标识
			temp_line [3] = handleFlag (row)  # 将源文件行中11种网络连接状态转换成数字标识
			temp_line [4] = handlenorm (int(row[4]))
			temp_line [5] = handlenorm (int(row[5]))
			temp_line [41] = handleLabel (row)  # 将源文件行中23种攻击类型转换成数字标识
			csv_writer.writerow (temp_line)
			count += 1
			# 输出每行数据中所修改后的状态
			#print (count, 'status:', temp_line [1], temp_line [2], temp_line [3], temp_line [41])
		data_file.close ()


# 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index (x, y):
	return [i for i in range (len (y)) if y [i] == x]


# 定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol (input):
	protocol_list = ['tcp', 'udp', 'icmp']
	if input [1] in protocol_list:
		return find_index (input [1], protocol_list) [0]


# 定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService (input):
	service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
					'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
					'hostnames',
					'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
					'ldap',
					'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
					'nntp',
					'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
					'shell',
					'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
					'urh_i', 'urp_i',
					'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
	if input [2] in service_list:
		return find_index (input [2], service_list) [0]


# 定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag (input):
	flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
	if input [3] in flag_list:
		return find_index (input [3], flag_list) [0]


# 定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
def handleLabel (input):
	# label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
	# 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
	# 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
	# 'spy.', 'rootkit.']
	global label_list  # 在函数内部使用全局变量并修改它
	if input [41] in label_list:
		return find_index (input [41], label_list) [0]
	else:
		label_list.append (input [41])
		return find_index (input [41], label_list) [0]



#定义将源文件行中从源主机到目标主机的数据的字节数归一化到（0,255)
def handlenorm(input):
	max_data=max(list_big)
	min_data=min(list_big)
	results=255/max_data*(input-min_data)
	
	return  results


if __name__ == '__main__':
	start_time = time.clock ()
	global label_list  # 声明一个全局变量的列表并初始化为空
	label_list = []
	preHandel_data ()
	end_time = time.clock ()
	print ("Running time:", (end_time - start_time))  # 输出程序运行时间
    
```
      该代码仅对10%的训练集（kddcup.data_10_percent_corrected）进行处理



引用：
https://blog.csdn.net/asialee_bird/article/details/80491256


6、KDD CUP99数据集按行转换成图片
# 将其转换为图片  
```python
import csv

import numpy as np

data =np.random.random((4,4,3))
data=[]
def Toimage():
	source_file = 'kddcup.data_10_percent_corrected.csv'
	with open (source_file, 'r') as data_source:
		csv_reader = csv.reader (data_source)
		count = 0  记录数据的行数，初始化为0
		for row in csv_reader:

			new_row=[float(x) for x in row]
			image_data=np.array(new_row)

			data = np.array (image_data).reshape (6, 7)
			imageio.imwrite ('image/{}.jpg'.format(count), data)
			count+=1


'''

#Toimage()
#print(np.array(data))
#
# new_list = [i for i in range(9)]
# data=np.array(data).reshape(4,4,3)
# print(data)

from multiprocessing import Process
from multiprocessing import Manager
import cv2

def Toimage ():
	source_file = 'kddcup.data_10_percent_corrected.csv'
	with open (source_file, 'r') as data_source:
		csv_reader = csv.reader (data_source)
		count = 0  # 记录数据的行数，初始化为0
		for row in csv_reader:
			new_row = [float (x) for x in row]
			image_data = np.array (new_row)
			data = np.array (image_data).reshape (6, 7)
			cv2.imwrite ('image/{}.jpg'.format (count), data)
			cv2.waitKey ()
			cv2.destroyAllWindows ()
			count += 1



if __name__ == '__main__':  # 进程间默认不能共用内存
	manager = Manager ()
	dic = manager.dict ()  # 这是一个特殊的字典
	

	p = Process (target=Toimage, args=(dic))
	p.start ()
	p.join ()
	
	
	print ('end')
```
