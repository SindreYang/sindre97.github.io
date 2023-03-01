
``` bash
# 前提
npm install 

# 同步语雀的文档到本地
yuque-hexo sync

#  清理语雀文档
yuque-hexo clean

# 生成文章 会在source文件夹下生成 文章名.md
hexo new "文章名"

# 清理生成的静态文件 也就是清理掉public文件夹以其下所有内容
hexo clean

# 重新编译生成静态文件 也就是生成public文件夹
hexo g

# 本地服务启动 主要用于本地调试博客
hexo server

# 上传博客到github，也就是将public文件夹下的所有内容上传到github
hexo d

```
