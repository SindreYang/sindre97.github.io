---
title: '教程2:斑点检测(Blob检测)'
tags:
  - cv
categories:
  - 框架
  - opencv
toc: false
date: 2020-03-09 00:52:33
---

# 什么是Blob？
Blob是图像中一组共享的像素，它们具有某些共同的属性（例如灰度值）。在上图中，深色连接区域是斑点，斑点检测的目的是识别并标记这些区域。
![image.png](https://blog.mviai.com/images/FpYRqbsn3CaJ-vf4IKRfKkChgVAW)

# 如何检测？

**顾名思义，SimpleBlobDetector基于以下描述的相当简单的算法。该算法由参数控制（下面以粗体显示），并具有以下步骤。**
1. **阈值处理**：通过以minThreshold开始的阈值对源图像进行阈值处理，将源图像转换为多个二进制图像  。这些阈值以thresholdStep递增，  直到  maxThreshold。因此，第一个阈值为  minThreshold， 第二个阈值为  minThreshold  +  thresholdStep，第三个阈 值为  minThreshold  + 2 x thresholdStep，依此类推。
2. **分组**： 在每个二进制图像中，连接的白色像素被分组在一起。让我们称这些二进制blob。
3.** 合并**   ：计算二进制图像中二进制斑点的中心，并合并比minDistBetweenBlob 更近的斑点  。
4. **中心和半径计算**： 计算并返回新合并的Blob的中心和半径。


# 过滤方法?
- **按颜色**：首先，您需要设置  filterByColor =1。设置  blobColor = 0以选择较暗的blob，将  blobColor =  255设置为较浅的blob。 
- **按大小**：   可以通过设置参数filterByArea = 1以及minArea   和maxArea的适当值来基于大小过滤blob 。例如，设置  minArea   = 100将滤除所有少于100个像素的斑点。
- 按形状： 现在形状具有三个不同的参数。
	- 圆度：  这只是测量斑点距圆的距离。例如，正六边形的圆度比正方形大。要按圆度过滤，请设置 filterByCircularity  =1。然后为minCircularity  和maxCircularity设置适当的值。 圆度定义为
		-	\ frac {4 * \ pi * Area} {perimeter * perimeter}
		- 这意味着圆的圆度为1，正方形的圆度为0.785，依此类推。

- 凸性： 凸度定义为（斑点的面积/凸包的面积）。现在，形状的凸包是完全封闭该形状的最紧密的凸形。由凸滤波器，首先设置filterByConvexity  = 1 ，然后设置0≤  minConvexity ≤1  和maxConvexity（≤1） 图为凹形与凸形
![image.png](https://blog.mviai.com/images/FiddIYcg0MJx1eJ9NVoEqVqG8Gkz)

- 惯性比： 不要让它吓到你。数学家经常使用容易混淆的单词来描述非常简单的事物。您只需要知道这可以衡量形状的伸长程度。例如，对于一个圆，该值是1，对于椭圆它是0和1之间，而对于线是0。要通过过滤器惯量比，设置  filterByInertia = 1 ， 并设置0≤  minInertiaRatio  ≤1  和  maxInertiaRatio  （≤ 1） 适当。 
![image.png](https://blog.mviai.com/images/FmI9PxggIyccKESVr5kkeRpaxHj9)


# SimpleBlobDetector
```python


# 导入相关的包
import cv2
import numpy as np

# 读取图片
im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)

# 设置SimpleBlobDetector参数
params = cv2.SimpleBlobDetector_Params()

# 更改阈值
params.minThreshold = 10
params.maxThreshold = 200


# 基于大小过滤
params.filterByArea = True
params.minArea = 1500

# 基于圆度过滤
params.filterByCircularity = True
params.minCircularity = 0.1

# 按凸性过滤
params.filterByConvexity = True
params.minConvexity = 0.87
    
# 按惯性过滤
params.filterByInertia = True
params.minInertiaRatio = 0.01

# 通过参数创建检测器
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# 检测 blobs.
keypoints = detector.detect(im)

# 将检测到的斑点绘制为红色圆圈
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# 确保圆的大小对应blobs的大小

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示 blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)



```

![image.png](https://blog.mviai.com/images/FmX70X4ma5-XzZTlRTTp23duG8DX)



```c++
// 导入相关包
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

	//读取图片
	Mat im = imread( "../blob.jpg", IMREAD_GRAYSCALE );

	// 设置SimpleBlobDetector参数
	SimpleBlobDetector::Params params;

	// 设置阈值
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// 按大小过滤
	params.filterByArea = true;
	params.minArea = 1500;

	// 按圆度过滤
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// 按凸性过滤
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// 按惯性过滤
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;


	// 创建保存特征点变量
	vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // 如果使用 OpenCV 2

	// 设置通过参数创建检测器
	SimpleBlobDetector detector(params);

	// 检测 blobs
	detector.detect( im, keypoints);
#else 

	// 设置通过参数创建检测器
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

	// 检测 blobs
	detector->detect( im, keypoints);
#endif


    //# 将检测到的斑点绘制为红色圆圈
    //# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    //# 确保圆的大小对应blobs的大小

	Mat im_with_keypoints;
	drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// 显示 blobs
	imshow("keypoints", im_with_keypoints );
	waitKey(0);

}



```


**注:希望能调好参数,把所有的都识别到**