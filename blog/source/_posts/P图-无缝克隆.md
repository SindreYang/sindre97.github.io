---
title: P图(无缝克隆)
tags:
  - cv
categories:
  - 框架
  - opencv
toc: false
date: 2020-03-09 01:58:44
---

# 无缝克隆是什么
两张图片合在一起,一般有以下三种方式
-	新手方式
-	![image.png](https://blog.mviai.com/images/Fs9r1gVdpyhIPnBve931nF2gPvbb)
-	Photoshop(遮罩)
-	![image.png](https://blog.mviai.com/images/FnwLXFbvhoNJfYZOpkMRLqslvIzb)
-	opencv方式
-	![image.png]https://blog.mviai.com/images/FprK2rkiGOSWO25lKuShhVluN77_)


# 原理
OpenCV中的无缝克隆是SIGGRAPH 2003有影响力的论文的实现，该论文名为Patrick Poez，Michel Gangnet和Andrew Blake，名称为[“ Poisson Image Editing”](http://www.irisa.fr/vista/Papers/2003_siggraph_perez.pdf)。

现在我们知道，如果使用精心创建的遮罩将源图像（飞机）的强度（RGB值）与目标图像（天空）混合，我们将获得如图3所示的结果。本文的中心思想是使用图像梯度而不是图像强度可以产生更真实的结果。无缝克隆后，蒙版区域中结果图像的强度与蒙版区域中源区域的强度不同。相反，结果图像在遮罩区域中的梯度与源区域在遮罩区域中的梯度大约相同。另外，在被遮蔽区域的边界处的结果图像的强度与目的地图像（天空）的强度相同。

作者表明，这是通过求解泊松方程来完成的，因此可以解决论文的标题-泊松图像编辑。该论文的理论和实现细节实际上非常酷.



# 主要函数
```python
output = cv2.seamlessClone(src, dst, mask, center, flags)
```

```c++
seamlessClone(Mat src, Mat dst, Mat mask, Point center, Mat output, int flags)

```

src	        将被复制到目标图像的源图像。在我们的示例中是飞机。
dst	        源映像将被克隆到的目标映像。在我们的示例中，它是天空图像。
mask	要克隆的对象周围的粗糙蒙版。这应该是源图像的大小。如果您很懒，请将其设置为全白图像！
center 	源图像中心在目标图像中的位置。
flags	当前起作用的两个标志是NORMAL_CLONE和MIXED_CLONE。正常克隆和无缝克隆。
output	输出结果图像。


# 基础版
```python
# 基本

# 导入相关的包
import cv2
import numpy as np 

# 读入图片
src = cv2.imread("images/airplane.jpg")
dst = cv2.imread("images/sky.jpg")


#在飞机周围创建一个粗糙的mask。
src_mask = np.zeros(src.shape, src.dtype)
#将遮罩定义为封闭的多边形
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
#函数cv :: fillPoly填充由多个多边形轮廓所界定的区域。该函数可以填充。复杂区域，例如，带有孔的区域，具有自相交的轮廓（它们的某些部分）等等
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# 飞机中心的位置
center = (800,100)

# 完成克隆
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# 保存结果
cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output);



```
```C++
//# 基本
//
//# 导入相关的包

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // 读入图片
    Mat src = imread("../images/airplane.jpg");
    Mat dst = imread("../images/sky.jpg");
    

    // 在飞机周围创建一个粗糙的mask。
    Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
    
    // 将遮罩定义为封闭的多边形
    Point poly[1][7];
    poly[0][0] = Point(4, 80);
    poly[0][1] = Point(30, 54);
    poly[0][2] = Point(151,63);
    poly[0][3] = Point(254,37);
    poly[0][4] = Point(298,90);
    poly[0][5] = Point(272,134);
    poly[0][6] = Point(43,122);
    
    const Point* polygons[1] = { poly[0] };
    int num_points[] = { 7 };
    
    // 通过填充多边形来创建蒙版n
    fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));
    
    // 飞机中心的位置
    Point center(800,100);
    
    // 将src无缝克隆到dst中，然后将结果放入输出中
    Mat output;
    seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);
    
    // 保存结果
    imwrite("../images/opencv-seamless-cloning-example.jpg", output);
    
}


```

![image.png](https://blog.mviai.com/images/Fts1LG2qSBpltPS2Tr6h27YVPG0Q)


# 高级版
```python

import cv2
import numpy as np

#读取图像
im = cv2.imread("images/wood-texture.jpg")
obj= cv2.imread("images/iloveyouticket.jpg")

# 创建全白图像
mask = 255 * np.ones(obj.shape, obj.dtype)

# dst中src中心的位置
width, height, channels = im.shape
center = (int(height/2), int(width/2))

#将src无缝克隆到dst中，然后将结果放入输出中
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

# 写入结果
cv2.imwrite("images/opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)


```

```c++

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // 读取图像：src图像将被克隆到dst中
    Mat src = imread("images/iloveyouticket.jpg");
    Mat dst = imread("images/wood-texture.jpg");
    

    // 创建全白mask
    Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());
    
    
    // dst中src中心的位置
    Point center(dst.cols/2,dst.rows/2);
    
    // 将src无缝克隆到dst中，然后将结果放入输出中
    Mat normal_clone;
    Mat mixed_clone;
    
    seamlessClone(src, dst, src_mask, center, normal_clone, NORMAL_CLONE);
    seamlessClone(src, dst, src_mask, center, mixed_clone, MIXED_CLONE);
    
    // 写入图像
    imwrite("images/opencv-normal-clone-example.jpg", normal_clone);
    imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone);
    
    
}


```
![image.png](https://blog.mviai.com/images/FgCBXytyYenRRRGBK70Sz1RTUk6Z)
![image.png](https://blog.mviai.com/images/FgzxGclKjyu_Cq3m436y2fWbrZ7I)
![image.png](https://blog.mviai.com/images/FqHI0kWMsDBdLYrfsI_R4r6ERFvs)

# 大师版

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图片
A = cv2.imread("images/man.jpg")
B = cv2.imread("images/woman.jpg")

# 转换成浮点数
A = np.float32(A) / 255.0
B = np.float32(B) / 255.0

#
# 在A中的人脸周围创建一个粗糙的mask
mask = np.zeros(A.shape,A.dtype)
polygon = np.array([[164,226], [209,225], [238,188], [252,133], [248,75], [240,29], [192,15], [150,15], [100,70], [106,133], [123,194] ], np.int32)
cv2.fillPoly(mask, [polygon], (255, 255, 255))
# 将mask转换浮点数
mask = np.float32(mask) / 255.0

# 浮点数<1.0 就乘以获取男女mask的加权平均值
mask = mask * 0.7 # 0.7 for man, 0.3 for woman

# 调整大小为 2^（金字塔中的级别）的倍数，因此在本例中为32
A = cv2.resize(A,(384,352))

# B的mask的大小应与A相同，以便以后进行乘法和加法运算
B = cv2.resize(B,(A.shape[1],A.shape[0]))
mask = cv2.resize(mask,(A.shape[1],A.shape[0]))

# 从原始图像开始（金字塔底）
guassianA = A.copy()
guassianB = B.copy()
guassianMask = mask.copy()
cv2.imshow('guass',guassianMask)

# 两种图像的拉普拉斯金字塔相结合
combinedLaplacianPyramids = []

# 金字塔中的级别数，尝试使用不同的值，请注意图像大小
maxIterations = 5

for i in range(maxIterations):

	# 计算两个图像的拉普拉斯金字塔
	laplacianA = cv2.subtract(guassianA, cv2.pyrUp(cv2.pyrDown(guassianA)))
	laplacianB = cv2.subtract(guassianB, cv2.pyrUp(cv2.pyrDown(guassianB)))

	# 结合两个拉普拉斯金字塔，将加权平均与mask金字塔金字塔相结合
	combinedLaplacian = guassianMask * laplacianA + (1.0 - guassianMask) * laplacianB

	# 在合并的拉普拉斯金字塔列表的开头添加CombinedLaplacian
	combinedLaplacianPyramids.insert(0,combinedLaplacian)

	# 更新高斯金字塔以进行下一次迭代
	guassianA = cv2.pyrDown(guassianA)
	guassianB = cv2.pyrDown(guassianB)
	guassianMask = cv2.pyrDown(guassianMask)

# 添加拉普拉斯金字塔的最后一个组合（金字塔的最高层）
lastCombined = guassianMask * guassianA + (1.0 - guassianMask) * guassianB
combinedLaplacianPyramids.insert(0,lastCombined)

# 重建影像
blendedImage = combinedLaplacianPyramids[0]
for i in range(1,len(combinedLaplacianPyramids)):
    # upSample并添加到下一个级别
    blendedImage = cv2.pyrUp(blendedImage)
    blendedImage = cv2.add(blendedImage, combinedLaplacianPyramids[i])

cv2.imshow('Blended',blendedImage)

# 直接融合两个图像进行比较
directCombination = mask * A + (1.0 - mask) * B
cv2.imshow('Direct combination',directCombination)

cv2.waitKey(0)

```

```c++


#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void getLaplacianPyramid(Mat& guassianPyramid, Mat& laplacianPyramid){
    // 从高斯金字塔计算拉普拉斯金字塔
    Mat downSampled;
    pyrDown(guassianPyramid,downSampled);

    // 上采样下采样
    Mat blurred;
    pyrUp(downSampled,blurred);

    subtract(guassianPyramid, blurred, laplacianPyramid);

}

void combineImages(Mat& A, Mat& B, Mat& mask, Mat& destination){
    
    destination = Mat::zeros(A.rows, A.cols, CV_32FC3);
    
    // 目的是求A和B的加权总和，分别具有权重掩码和1-掩码
    for(int y = 0; y < A.rows; y++)
    {
        for(int x = 0; x < A.cols; x++)
        {   
            Vec3f a = A.at<Vec3f>(Point(x,y));
            Vec3f b = B.at<Vec3f>(Point(x,y));
            Vec3f m = mask.at<Vec3f>(Point(x,y));
            
            float b_ = a[0]*m[0]+(1-m[0])*b[0];
            float g_ = a[1]*m[1]+(1-m[1])*b[1];
            float r_ = a[2]*m[2]+(1-m[2])*b[2];

            destination.at<Vec3f>(y,x)[0] = b_;
            destination.at<Vec3f>(y,x)[1] = g_;
            destination.at<Vec3f>(y,x)[2] = r_;
        }
    }
}

int main( int argc, char** argv )
{
    // 读取图片
    Mat A = imread("images/man.jpg");
    Mat B = imread("images/woman.jpg");

    // 转换成浮点数
    A.convertTo(A, CV_32FC3, 1/255.0);
    B.convertTo(B, CV_32FC3, 1/255.0);

    //在A的人脸周围创建一个粗糙的mask。
    Mat mask = Mat::zeros(A.rows, A.cols, CV_8UC3);

    // 创建飞机的边缘点
    Point points[11];
    points[0] = Point(164,226);
    points[1] = Point(209,225);
    points[2] = Point(238,188);
    points[3] = Point(252,133);
    points[4] = Point(248,75);
    points[5] = Point(240,29);
    points[6] = Point(192,15);
    points[7] = Point(150,15);
    points[8] = Point(100,70);
    points[9] = Point(106,133);
    points[10] = Point(123,194);

    const Point* polygon[1] = {points}; //构成点数组
    int npt[] = {11}; // 点数组的长度
    
    //填充由点形成的多边形
    fillPoly(mask, polygon, npt, 1, Scalar(255, 255, 255));

    // 转换成浮点数
    mask.convertTo(mask, CV_32FC3, 1/255.0);

    // 用 浮点数<1.0 乘以获取男女面部的加权平均值
    mask = mask * 0.7;

    // 调整大小为2^（金字塔中的级别）的倍数，因此在本例中为32
    resize(A, A, Size(384,352));

    //B mask的大小应与A相同，以便以后进行乘法和加法运算
    resize(B, B, A.size());
    resize(mask, mask, A.size());

    //从原始图像开始（金字塔底）
    Mat guassianA = A.clone();
    Mat guassianB = B.clone();
    Mat guassianMask = mask.clone();

    // 金字塔中的级别数，尝试使用不同的值。注意图像尺寸
    int maxIterations = 2;

    // 两种图像的组合拉普拉斯金字塔
    vector<Mat> combinedLaplacianPyramids;

    for (int i = 0; i < maxIterations; i++){
        // 计算A的拉普拉斯金字塔
        Mat laplacianA;
        getLaplacianPyramid(guassianA,laplacianA);

        // 计算B的拉普拉斯金字塔
        Mat laplacianB;
        getLaplacianPyramid(guassianB,laplacianB);

        // 结合拉普拉斯金字塔
        Mat combinedLaplacian;
        combineImages(laplacianA, laplacianB, guassianMask, combinedLaplacian);
 
        //在合并的拉普拉斯金字塔列表的开头插入combinedLaplacian
        combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),combinedLaplacian);

        // 更新高斯金字塔以进行下一次迭代
        pyrDown(guassianA,guassianA);
        pyrDown(guassianB,guassianB);
        pyrDown(guassianMask,guassianMask);

    }

    //合并最后一个guassian（拉普拉斯金字塔的顶层与guassian的金字塔相同）
    Mat lastCombined;
    combineImages(guassianA, guassianB, guassianMask, lastCombined);

    // 在合并的拉普拉斯金字塔列表的开头插入lastCombined
    combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),lastCombined);

    // 重建影像
    Mat blendedImage = combinedLaplacianPyramids[0];

    for (int i = 1; i < combinedLaplacianPyramids.size(); i++){
        // upSample并添加到下一个级别
        pyrUp(blendedImage,blendedImage);
        add(blendedImage, combinedLaplacianPyramids[i],blendedImage);
    }

    // 将混合图像放回原始位置的天空图像
    imshow("blended",blendedImage);

    // 将混合图像放回原始位置的天空图像
    Mat directCombined;
    combineImages(A, B, mask, directCombined);
    imshow("directCombined",directCombined);
    waitKey(0);


    return 0;
}


```
![image.png](https://blog.mviai.com/images/FmB86oMyDHvbEWkBhIR11rOqea_E)