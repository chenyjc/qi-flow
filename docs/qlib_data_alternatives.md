# Qlib 数据下载替代方案

当 GitHub Release 更新不及时时，可以使用以下替代方案获取最新数据。

## 方案一：使用 Dolt 直接克隆数据

Dolt 数据库可能比 GitHub Release 更及时。

### 安装 Dolt

```bash
# Linux
sudo curl -L https://github.com/dolthub/dolt/releases/latest/download/dolt-linux-amd64 -o /usr/local/bin/dolt
sudo chmod +x /usr/local/bin/dolt

# Mac
brew install dolt
```

### 克隆数据

```bash
dolt clone chenditc/investment_data
```

### 导出为 Qlib 格式

需要 Docker：

```bash
docker run -v /<output_dir>:/output -it --rm chenditc/investment_data bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

### 配置要求

- 磁盘空间：5-10GB
- 内存：8-16GB（导出时）
- 时间：10-30 分钟克隆，5-15 分钟导出

## 方案二：使用 Docker 自己运行更新

需要 Tushare token（免费注册：https://tushare.pro/）。

```bash
export TUSHARE=<Token>

docker run -v /<output_dir>:/output -it --rm chenditc/investment_data bash daily_update.sh && bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

## 方案三：使用 Akshare 补充最新数据

项目已集成 Akshare，可以：
- 用 Qlib 历史数据做训练
- 用 Akshare 获取最近几天的实时数据

## 相关链接

- GitHub: https://github.com/chenditc/investment_data
- DoltHub: https://www.dolthub.com/repositories/chenditc/investment_data
- Tushare: https://tushare.pro/
- 中文介绍: 量化系列2 - 众包数据集