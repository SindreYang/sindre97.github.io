---
title: WordPress页面，分页函数调用，主题函数调用，模板函数调用
tags:
  - wordpress
url: 137.html
id: 137
categories:
  - 网站
  - wordpress
date: 2018-04-09 02:30:26

---

*   WordPress页面，分页函数调用，主题函数调用，模板函数调用
    --------------------------------
    

*   [1描述](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.8F.8F.E8.BF.B0)
*   [2与WP_Query的互动](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E4.B8.8EWP_Query.E7.9A.84.E4.BA.92.E5.8A.A8)
*   [3用法](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E7.94.A8.E6.B3.95)
*   [4方法和属性](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.B9.E6.B3.95.E5.92.8C.E5.B1.9E.E6.80.A7)
    *   [4.1属性](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E5.B1.9E.E6.80.A7)
    *   [4.2方法](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.B9.E6.B3.95)
*   [5参数](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E5.8F.82.E6.95.B0)
    *   [5.1作者](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E4.BD.9C.E8.80.85)
    *   [5.2分类目录](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E5.88.86.E7.B1.BB.E7.9B.AE.E5.BD.95)
    *   [5.3标签](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.A0.87.E7.AD.BE)
    *   [5.4自定义分类法](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E8.87.AA.E5.AE.9A.E4.B9.89.E5.88.86.E7.B1.BB.E6.B3.95)
    *   [5.5搜索](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.90.9C.E7.B4.A2)
    *   [5.6文章＆页面](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.87.E7.AB.A0_.26_.E9.A1.B5.E9.9D.A2)
    *   [5.7文章类型](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.87.E7.AB.A0.E7.B1.BB.E5.9E.8B)
    *   [5.8文章状态](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.87.E7.AB.A0.E7.8A.B6.E6.80.81)
    *   [5.9分页](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E5.88.86.E9.A1.B5)
    *   [5.10排序](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.8E.92.E5.BA.8F)
    *   [5.11置顶文章](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E7.BD.AE.E9.A1.B6.E6.96.87.E7.AB.A0)
    *   [5.12时间](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.97.B6.E9.97.B4)
    *   [5.13自定义字段](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E8.87.AA.E5.AE.9A.E4.B9.89.E5.AD.97.E6.AE.B5)
    *   [5.14权限](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.9D.83.E9.99.90)
    *   [5.15缓存](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E7.BC.93.E5.AD.98)
    *   [5.16返回字段](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E8.BF.94.E5.9B.9E.E5.AD.97.E6.AE.B5)

[官网法典](https://codex.wordpress.org/zh-cn:Class_Reference/WP_Query#.E6.96.87.E7.AB.A0_.26_.E9.A1.B5.E9.9D.A2)
