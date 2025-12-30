#!/bin/bash

# 下载最新数据
wget -O qlib_bin.tar.gz https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz

# 创建目标目录
mkdir -p ~/.qlib/qlib_data/cn_data

# 解压文件
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1

# 删除临时文件
rm -f qlib_bin.tar.gz

echo "Qlib数据更新完成: $(date)"