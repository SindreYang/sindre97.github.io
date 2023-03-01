---
title: php中添加html中的css，js，图片，视频等外部文件方法
tags:
  - wordpress
url: 129.html
id: 129
categories:
  - 网站
  - wordpress
date: 2018-04-08 16:15:10
---

php中添加html中的css，js，图片，视频等外部文件方法
-------------------------------

**方法一：**
--------

1、将 xx.html 修改为 page-xx.php 上传到你当前使用的主题目录中； 2、在WordPress后台创建别名为 xx 的页面后发布，大功告成。 **注意：创建的页面别名一定要与page-后面一样。或者在步骤1中将xx改为你的页面ID。**

**方法二：**
--------

1、将 xx.html 修改为page-template-xx.php 然后再该文件头部添加：

    <?php
    /*
    Template Name: xx 页面模板
    */
    ?>
    

然后上传到你当前使用的主题目录中； 2、在WordPress后台创建别名为 xx 的页面并选择页面模板为第一步中创建的 xx页面模板，然后发布，大功告成。 **如果你添加的页面是首页，可以在 设置 > 阅读 \> 首页显示 中设置将此页面设置为首页。**

**关于CSS、JS、图片等外部文件**
--------------------

如果你的页面有引用CSS、JS以及图片，例如：sample.css、sample.js、sample.jpg，可以将这些文件一并复制到主题目录下，然后引用地址改为：

    <link href="<?php echo get_theme_file_uri( 'sample.css' ); ?>" rel="stylesheet" type="text/css" />
    <script src="<?php echo get_theme_file_uri( 'sample.js' ); ?>" type="text/javascript"></script>
    <img src="<?php echo get_theme_file_uri( 'sample.jpg' ); ?>" />
    

如果有视频等，同上面方法。

* * *

**如果你想了解更多：**
-------------

2017.11.04 新增： 上面介绍了将WordPress转换为Page（页面）的方法，下面介绍转换为首页、分类、标签、文章等页面的方法：

1.  front-page.php**：这个文件是首页，**如果没有则使用上面方法二中在后台设置为首页的页面；
2.  home.php：文档归档页面，通常1中都没有使用这个显示首页；
3.  index.php：1、2都没有使用这个显示首页；
4.  single.php：文章模板文件；
5.  404.php：404页面文件；
6.  page.php：页面模板文件，支持 page-$id （即页面 ID）或 page-$slug （即页面别名）；
7.  category.php：分类归档模板文件，支持 category-$id 或category-$slug ；
8.  tag.php：标签归档模板文件，支持 category-$id 或category-$slug ；
9.  author.php：作者归档模板文件，支持 category-$id 或 category-$slug；
10.  date.php：日期归档模板文件；
11.  archive.php：如果主题没有7-10之中的任一文件，那么都会用此模板文件显示对应内容，当此模板文件也不存在时，则使用index.php显示，支持 archive-$id 或archive-$slug。

header.php、footer.php、sidebar.php等文件一般都是“页面部分”模板文件，即：页眉、页脚、边栏。 如果你能看懂英文，详细可参考官方文档： [WordPress模板文件等级介绍（官方）](https://link.zhihu.com/?target=https%3A//developer.wordpress.org/themes/basics/template-hierarchy/) [WordPress获取主题目录里的文件和目录](https://link.zhihu.com/?target=https%3A//developer.wordpress.org/themes/basics/linking-theme-files-directories/)

[@马向阳](https://www.zhihu.com/question/20129430/answer/40653740)
