import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
import pandas as pd
import numpy as np
import os
import tarfile
import urllib.request
from tqdm import tqdm
import datetime
import sqlite3
import logging

# 配置日志
logger = logging.getLogger(__name__)

class QlibService:
    def __init__(self):
        import os
        # 使用绝对路径
        self.provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        print(f"Qlib数据路径: {self.provider_uri}")
        print(f"路径是否存在: {os.path.exists(self.provider_uri)}")
        self.init_qlib()
        self.init_stock_db()
    
    def init_qlib(self):
        """初始化Qlib"""
        try:
            # 检查Qlib是否已初始化
            if not hasattr(qlib, '_initialized') or not qlib._initialized:
                qlib.init(provider_uri=self.provider_uri, region=REG_CN)
            
            # 配置Qlib workflow使用SQLite后端
            sqlite_uri = "sqlite:///mlflow.db"
            R.set_uri(sqlite_uri)
        except Exception as e:
            print(f"Qlib初始化失败: {e}")
            # 尝试使用绝对路径
            try:
                import os
                provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
                qlib.init(provider_uri=provider_uri, region=REG_CN)
                print(f"使用绝对路径初始化Qlib成功: {provider_uri}")
            except Exception as e2:
                print(f"使用绝对路径初始化Qlib也失败: {e2}")
    
    def release_qlib(self):
        """释放Qlib对数据目录的占用"""
        try:
            # 清除Qlib的缓存和连接
            if hasattr(qlib, '_initialized'):
                qlib._initialized = False
            # 清除数据缓存
            from qlib.data import D
            if hasattr(D, '_data'):
                D._data = None
            print("Qlib资源已释放")
        except Exception as e:
            print(f"释放Qlib资源失败: {e}")
    
    def init_stock_db(self):
        """初始化股票信息数据库"""
        DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 创建股票信息表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def update_stock_db(self):
        """更新股票信息数据库"""
        try:
            import akshare as ak
            # 获取A股所有股票信息
            stock_info_df = ak.stock_info_a_code_name()
            
            DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # 获取当前时间
            updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 清空旧数据并插入新数据
            cursor.execute('DELETE FROM stock_info')
            
            # 插入新数据
            for _, row in stock_info_df.iterrows():
                code = row['code']
                name = row['name']
                cursor.execute('INSERT INTO stock_info (code, name, updated_at) VALUES (?, ?, ?)', 
                              (code, name, updated_at))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"更新股票数据库失败: {e}")
            return False
    
    def get_stock_names_from_db(self, stock_keys):
        """从数据库获取股票名称"""
        stock_names = {}
        try:
            DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            for stock in stock_keys:
                # 尝试不同格式的代码
                code_formats = [stock]
                if stock.startswith('SH') or stock.startswith('SZ'):
                    code_formats.append(stock[2:])
                else:
                    code_formats.append(f"SH{stock}")
                    code_formats.append(f"SZ{stock}")
                
                for code in code_formats:
                    cursor.execute('SELECT name FROM stock_info WHERE code = ?', (code,))
                    result = cursor.fetchone()
                    if result:
                        stock_names[stock] = result[0]
                        break
            
            conn.close()
            return stock_names, True, None
        except Exception as e:
            return {}, False, e
    
    def get_stock_names(self, stock_keys):
        """获取股票代码到名称的映射"""
        stock_names = {}
        
        # 1. 首先尝试从数据库获取
        stock_names, db_success, db_error = self.get_stock_names_from_db(stock_keys)
        
        if db_success and stock_names:
            return stock_names, True, None
        
        # 2. 如果数据库获取失败或没有数据，尝试从akshare获取
        try:
            import akshare as ak
            # 尝试获取股票名称数据
            stock_info_df = ak.stock_info_a_code_name()
            
            # 创建股票代码到名称的映射，支持带前缀和不带前缀的代码
            for _, row in stock_info_df.iterrows():
                code = row['code']
                name = row['name']
                # 保存不带前缀的代码映射
                stock_names[code] = name
                # 保存带SH前缀的代码映射
                stock_names[f"SH{code}"] = name
                # 保存带SZ前缀的代码映射
                stock_names[f"SZ{code}"] = name
            
            # 更新数据库
            self.update_stock_db()
            
            return stock_names, True, None
        except Exception as e:
            # 3. 如果所有方法都失败，使用股票代码作为名称
            for stock in stock_keys:
                stock_names[stock] = stock
                # 为不同格式的代码创建映射
                if stock.startswith('SH') or stock.startswith('SZ'):
                    # 同时添加不带前缀的映射
                    stock_names[stock[2:]] = stock[2:]
                else:
                    # 同时添加带前缀的映射
                    stock_names[f"SH{stock}"] = stock
                    stock_names[f"SZ{stock}"] = stock
            return stock_names, False, e
    
    def download_data_with_progress(self, progress_callback=None):
        """下载Qlib数据（备份旧数据后替换，多线程加速，带进度回调）"""
        try:
            import shutil
            import concurrent.futures
            import threading
            
            # 数据URL
            url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
            
            # 目标目录
            target_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data")
            
            # 备份目录
            backup_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data_backup")
            
            # 临时文件路径（使用绝对路径）
            temp_file = os.path.join(os.path.expanduser("~/.qlib"), "qlib_bin.tar.gz")
            
            # 创建临时目录
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            # 进度锁和状态
            progress_lock = threading.Lock()
            downloaded_bytes = [0]
            last_reported_progress = [0]
            
            def report_progress(progress, message):
                if progress_callback:
                    # 避免重复报告相同进度
                    with progress_lock:
                        if progress > last_reported_progress[0] or progress == 100:
                            last_reported_progress[0] = progress
                            progress_callback(progress, message)
            
            # 多线程下载函数（分块读取以实时报告进度）
            def download_chunk(url, start, end, temp_file_part, chunk_index):
                headers = {'Range': f'bytes={start}-{end}'}
                req = urllib.request.Request(url, headers=headers)
                chunk_size = 1024 * 1024  # 1MB per read
                
                with urllib.request.urlopen(req, timeout=300) as response:
                    with open(temp_file_part, 'wb') as f:
                        while True:
                            data = response.read(chunk_size)
                            if not data:
                                break
                            f.write(data)
                            with progress_lock:
                                downloaded_bytes[0] += len(data)
                                progress = int(downloaded_bytes[0] / file_size * 80)
                                # 每5%报告一次进度
                                if progress >= last_reported_progress[0] + 5 or progress >= 80:
                                    report_progress(progress, f"下载中 {downloaded_bytes[0]/1024/1024:.1f}MB / {file_size/1024/1024:.1f}MB")
            
            report_progress(0, "开始下载Qlib数据...")
            
            # 获取文件大小
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as response:
                file_size = int(response.headers.get('Content-Length', 0))
            
            downloaded = False
            if file_size > 0:
                report_progress(5, f"文件大小: {file_size / 1024 / 1024:.2f} MB")
                
                try:
                    # 使用多线程下载（8线程）
                    num_threads = 8
                    chunk_size = file_size // num_threads
                    
                    report_progress(10, f"使用 {num_threads} 线程并行下载...")
                    
                    # 创建临时文件片段
                    temp_parts = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = []
                        for i in range(num_threads):
                            start = i * chunk_size
                            end = start + chunk_size - 1 if i < num_threads - 1 else file_size - 1
                            temp_part = temp_file + f'.part{i}'
                            temp_parts.append(temp_part)
                            futures.append(executor.submit(download_chunk, url, start, end, temp_part, i))
                        
                        # 等待所有下载完成
                        concurrent.futures.wait(futures)
                    
                    # 合并文件片段
                    report_progress(80, "合并下载片段...")
                    with open(temp_file, 'wb') as outfile:
                        for i, temp_part in enumerate(temp_parts):
                            with open(temp_part, 'rb') as infile:
                                outfile.write(infile.read())
                            os.remove(temp_part)
                            report_progress(80 + int((i + 1) / len(temp_parts) * 5), f"合并片段 {i+1}/{len(temp_parts)}")
                    
                    downloaded = True
                    report_progress(85, "下载完成")
                except Exception as e:
                    print(f"多线程下载失败: {str(e)}")
                    # 清理临时文件片段
                    for temp_part in temp_parts:
                        if os.path.exists(temp_part):
                            os.remove(temp_part)
            
            # 如果多线程下载失败，使用单线程下载
            if not downloaded:
                report_progress(20, "使用单线程下载...")
                
                def single_thread_progress(count, block_size, total_size):
                    downloaded = count * block_size
                    progress = int(downloaded / total_size * 80)
                    report_progress(progress, f"下载中 {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
                
                urllib.request.urlretrieve(url, temp_file, reporthook=single_thread_progress)
                report_progress(85, "下载完成")
            
            # 释放Qlib对数据目录的占用
            report_progress(88, "释放资源...")
            self.release_qlib()
            
            # 强制垃圾回收释放资源
            import gc
            gc.collect()
            
            report_progress(90, "开始备份旧数据...")
            
            # 删除上次的备份（如果存在）
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            # 备份当前数据（如果存在）- 使用复制后删除的方式
            if os.path.exists(target_dir):
                report_progress(91, "复制旧数据到备份目录...")
                shutil.copytree(target_dir, backup_dir)
                report_progress(92, "删除旧数据目录...")
                # 尝试多次删除，因为可能有文件锁定
                for attempt in range(3):
                    try:
                        shutil.rmtree(target_dir)
                        break
                    except Exception as e:
                        if attempt < 2:
                            gc.collect()
                            import time
                            time.sleep(1)
                        else:
                            raise e
            
            report_progress(93, "开始解压新数据...")
            
            # 创建目标目录
            os.makedirs(target_dir, exist_ok=True)
            
            # 解压文件
            with tarfile.open(temp_file, "r:gz") as tar:
                members = tar.getmembers()
                total_members = len(members)
                # 过滤出需要解压的文件
                valid_members = [m for m in members if '/' in m.name]
                
                for i, member in enumerate(valid_members):
                    member.name = member.name.split('/', 1)[1]
                    tar.extract(member, target_dir)
                    # 每10个文件报告一次进度
                    if i % 10 == 0 or i == len(valid_members) - 1:
                        progress = 93 + int((i + 1) / len(valid_members) * 5)
                        report_progress(progress, f"解压中 {i+1}/{len(valid_members)} 文件")
            
            # 删除临时文件
            os.remove(temp_file)
            
            report_progress(98, "重新初始化Qlib...")
            
            # 重新初始化Qlib
            self.init_qlib()
            
            report_progress(100, "完成")
            
            return {"success": True, "message": "Qlib数据已更新完成！旧数据已备份。"}
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            # 如果更新失败，尝试恢复备份
            try:
                if os.path.exists(backup_dir) and not os.path.exists(target_dir):
                    shutil.move(backup_dir, target_dir)
            except Exception as restore_error:
                print(f"恢复备份失败: {restore_error}")
            return {"success": False, "message": f"数据加载失败: {str(e)}"}
    
    def download_data(self):
        """下载Qlib数据（备份旧数据后替换，多线程加速）"""
        return self.download_data_with_progress(None)
    
    def train_model_with_progress(self, progress_callback, market, benchmark, train_start_date, train_end_date, 
                    valid_start_date, valid_end_date, test_start_date, test_end_date, 
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree):
        """训练模型（带进度回调）"""
        try:
            progress_callback(0, "开始训练模型...")
            
            progress_callback(10, "配置数据处理参数...")
            
            # 数据处理配置
            data_handler_config = {
                "start_time": train_start_date,
                "end_time": test_end_date,
                "fit_start_time": train_start_date,
                "fit_end_time": valid_end_date,
                "instruments": market,
            }
            
            progress_callback(20, "构建任务配置...")
            
            # 构建任务配置
            task = {
                "model": {
                    "class": model_type,
                    "module_path": "qlib.contrib.model.gbdt",
                    "kwargs": {
                        "loss": "mse",
                        "colsample_bytree": colsample_bytree,
                        "learning_rate": lr,
                        "subsample": subsample,
                        "lambda_l1": 205.6999,
                        "lambda_l2": 580.9768,
                        "max_depth": max_depth,
                        "num_leaves": num_leaves,
                        "num_threads": 20,
                    },
                },
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158",
                            "module_path": "qlib.contrib.data.handler",
                            "kwargs": data_handler_config,
                        },
                        "segments": {
                            "train": (train_start_date, train_end_date),
                            "valid": (valid_start_date, valid_end_date),
                            "test": (test_start_date, test_end_date),
                        },
                    },
                },
            }
            
            progress_callback(30, "初始化模型...")
            
            # 初始化模型
            model = init_instance_by_config(task["model"])
            
            progress_callback(40, "初始化数据集...")
            
            # 初始化数据集
            dataset = init_instance_by_config(task["dataset"])
            
            progress_callback(50, "开始模型训练...")
            
            # 训练模型
            with R.start(experiment_name="train_model"):
                # 保存训练参数
                R.log_params(
                    market=market,
                    benchmark=benchmark,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    valid_start_date=valid_start_date,
                    valid_end_date=valid_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_end_date,
                    model_type=model_type,
                    lr=lr,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    **flatten_dict(task)
                )
                
                progress_callback(60, "训练模型中...")
                model.fit(dataset)
                
                progress_callback(80, "保存模型...")
                R.save_objects(trained_model=model)
                rid = R.get_recorder().id
            
            progress_callback(90, "训练完成，生成记录ID...")
            
            return {"success": True, "recorder_id": rid, "message": "模型训练完成！"}
        except Exception as e:
            progress_callback(-1, f"模型训练失败: {str(e)}")
            return {"success": False, "message": f"模型训练失败: {str(e)}"}
    
    def train_model(self, market, benchmark, train_start_date, train_end_date, 
                    valid_start_date, valid_end_date, test_start_date, test_end_date, 
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree):
        """训练模型"""
        return self.train_model_with_progress(None, market, benchmark, train_start_date, train_end_date, 
                    valid_start_date, valid_end_date, test_start_date, test_end_date, 
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree)
    
    def backtest_model(self, recorder_id, market, benchmark, start_date, end_date, 
                      initial_account, topk, n_drop, strategy_type):
        """执行回测"""
        try:
            # 获取指定ID的记录器
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")
            
            # 加载模型
            model = recorder.load_object("trained_model")
            
            # 加载数据集（需要重新构建）
            data_handler_config = {
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": start_date,
                "fit_end_time": end_date,
                "instruments": market,
            }
            
            dataset = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "test": (start_date, end_date),
                    },
                },
            }
            
            dataset = init_instance_by_config(dataset)
            
            # 回测配置
            port_analysis_config = {
                "executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "generate_portfolio_metrics": True,
                    },
                },
                "strategy": {
                    "class": strategy_type,
                    "module_path": "qlib.contrib.strategy.signal_strategy",
                    "kwargs": {
                        "model": model,
                        "dataset": dataset,
                        "topk": topk,
                        "n_drop": n_drop,
                    },
                },
                "backtest": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "account": initial_account,
                    "benchmark": benchmark,
                    "exchange_kwargs": {
                        "freq": "day",
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
            }
            
            # 执行回测和分析
            with R.start(experiment_name="backtest_analysis"):
                recorder = R.get_recorder()
                ba_rid = recorder.id
                
                # 保存训练记录ID和参数到回测记录中
                recorder.log_params(
                    # 训练记录ID
                    train_recorder_id=str(recorder_id),
                    
                    # 回测参数
                    backtest_market=market,
                    backtest_benchmark=benchmark,
                    backtest_start_date=start_date,
                    backtest_end_date=end_date,
                    initial_account=initial_account,
                    topk=topk,
                    n_drop=n_drop,
                    strategy_type=strategy_type
                )
                
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()
                
                par = PortAnaRecord(recorder, port_analysis_config, "day")
                par.generate()
            
            return {"success": True, "recorder_id": ba_rid, "message": "回测和分析完成！"}
        except Exception as e:
            return {"success": False, "message": f"回测和分析失败: {str(e)}"}
    
    def get_recorders(self, experiment_name="train_model"):
        """获取训练记录"""
        try:
            exp = R.get_exp(experiment_name=experiment_name)
            recorders = exp.list_recorders(rtype="list")
            
            recorder_list = []
            for rec in recorders:
                rec_info = rec.info
                rec_id = rec.id
                recorder_list.append({
                    "id": rec_id,
                    "start_time": rec_info.get('start_time', '未知'),
                    "params": rec.list_params()
                })
            
            return {"success": True, "recorders": recorder_list}
        except Exception as e:
            return {"success": False, "message": f"获取实验记录失败: {str(e)}"}
    
    def get_backtest_recorders(self):
        """获取回测记录"""
        try:
            backtest_exp = R.get_exp(experiment_name="backtest_analysis")
            backtest_recorders = backtest_exp.list_recorders(rtype="list")
            
            recorder_list = []
            for rec in backtest_recorders:
                rec_info = rec.info
                rec_id = rec.id
                recorder_list.append({
                    "id": rec_id,
                    "start_time": rec_info.get('start_time', '未知'),
                    "params": rec.list_params()
                })
            
            return {"success": True, "recorders": recorder_list}
        except Exception as e:
            return {"success": False, "message": f"获取回测记录失败: {str(e)}"}
    
    def get_backtest_result(self, recorder_id):
        """获取回测结果"""
        try:
            # 获取回测记录
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="backtest_analysis")
            
            # 加载结果数据
            report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
            positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
            analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

            print(f"Loaded positions: {positions is not None}, type: {type(positions)}")
            if positions:
                print(f"Positions keys: {len(positions.keys()) if hasattr(positions, 'keys') else 'N/A'}")
                if hasattr(positions, 'keys'):
                    print(f"First few dates: {list(positions.keys())[:3]}")
            
            # 计算累计收益率
            report_normal_df['cumulative_return'] = (1 + report_normal_df['return']).cumprod()
            report_normal_df['cumulative_bench'] = (1 + report_normal_df['bench']).cumprod()
            
            total_return = (report_normal_df['cumulative_return'].iloc[-1] - 1) * 100
            bench_return = (report_normal_df['cumulative_bench'].iloc[-1] - 1) * 100
            excess_return_total = total_return - bench_return
            
            # 获取其他分析指标
            annualized_return = analysis_df.loc[('excess_return_with_cost', 'annualized_return'), 'risk']
            information_ratio = analysis_df.loc[('excess_return_with_cost', 'information_ratio'), 'risk']
            max_drawdown = analysis_df.loc[('excess_return_with_cost', 'max_drawdown'), 'risk']
            
            # 关键指标
            key_metrics = {
                "total_return": round(total_return, 2),
                "bench_return": round(bench_return, 2),
                "excess_return": round(excess_return_total, 2),
                "annualized_return": round(annualized_return, 4),
                "information_ratio": round(information_ratio, 3),
                "max_drawdown": round(max_drawdown, 4)
            }
            
            # 累计收益数据
            cumulative_data = {
                "dates": report_normal_df.index.strftime('%Y-%m-%d').tolist(),
                "strategy": report_normal_df['cumulative_return'].tolist(),
                "benchmark": report_normal_df['cumulative_bench'].tolist()
            }
            
            # 日收益数据
            daily_data = {
                "dates": report_normal_df.index.strftime('%Y-%m-%d').tolist(),
                "strategy": report_normal_df['return'].tolist(),
                "benchmark": report_normal_df['bench'].tolist()
            }
            
            # 持仓数据
            positions_data = []
            all_positions_data = {}  # 存储所有日期的持仓
            if positions:
                # 获取所有日期
                all_dates = list(positions.keys())

                # 预先获取所有涉及的股票代码
                all_stock_keys = set()
                for date in all_dates:
                    pos = positions[date]
                    if hasattr(pos, 'position'):
                        pos_dict = pos.position
                    elif isinstance(pos, dict):
                        pos_dict = pos
                    else:
                        pos_dict = {}
                    all_stock_keys.update([stock for stock in pos_dict.keys() if stock not in ['cash', 'now_account_value']])

                # 获取所有股票名称
                stock_names, success, error = self.get_stock_names(list(all_stock_keys))

                # 遍历所有日期的持仓，存储到 all_positions_data
                for date in all_dates:
                    pos = positions[date]
                    if hasattr(pos, 'position'):
                        pos_dict = pos.position
                    elif isinstance(pos, dict):
                        pos_dict = pos
                    else:
                        pos_dict = {}

                    stock_keys = [stock for stock in pos_dict.keys() if stock not in ['cash', 'now_account_value']]
                    date_positions = []
                    for stock in stock_keys:
                        info = pos_dict[stock]
                        if isinstance(info, dict):
                            stock_name = stock_names.get(stock, stock)
                            current_price = float(info.get('price', 0))
                            amount = float(info.get('amount', 0))
                            count_day = int(info.get('count_day', 0))

                            # 计算成本价和盈亏 - 使用与最终持仓相同的逻辑
                            avg_price = current_price
                            profit = 0.0
                            profit_rate = 0.0

                            try:
                                # 计算开仓日期
                                open_date = date - pd.DateOffset(days=count_day)
                                start_date = open_date - pd.DateOffset(days=2)
                                end_date = open_date + pd.DateOffset(days=2)

                                print(f"[DEBUG] Stock {stock}, date={date}, count_day={count_day}, open_date={open_date}")

                                price_data = D.features(
                                    instruments=[stock],
                                    fields=['$close'],
                                    start_time=start_date.strftime('%Y-%m-%d'),
                                    end_time=end_date.strftime('%Y-%m-%d')
                                )

                                print(f"[DEBUG] price_data empty={price_data.empty}, shape={price_data.shape if not price_data.empty else 0}")

                                if not price_data.empty:
                                    # 查找最接近开仓日期的价格
                                    date_diffs = abs(price_data.index.get_level_values('datetime') - open_date)
                                    min_diff_idx = date_diffs.argmin()
                                    avg_price = float(price_data.iloc[min_diff_idx]['$close'])
                                    print(f"[DEBUG] Found avg_price={avg_price}, current_price={current_price}")
                                else:
                                    print(f"[DEBUG] No price data found for {stock} around {open_date}")
                            except Exception as e:
                                print(f"[DEBUG] Error getting price for {stock}: {e}")

                            # 计算盈利金额和收益率 - 与最终持仓相同的公式
                            if avg_price > 0 and current_price > 0:
                                profit = (current_price - avg_price) * amount
                                profit_rate = (current_price - avg_price) / avg_price * 100

                            date_positions.append({
                                'stock_code': stock,
                                'stock_name': stock_name,
                                'weight': float(info.get('weight', 0)),
                                'hold_days': count_day,
                                'amount': amount,
                                'cost_price': avg_price,
                                'current_price': current_price,
                                'hold_value': amount * current_price,
                                'profit': round(profit, 2),
                                'profit_rate': round(profit_rate, 2)
                            })
                    all_positions_data[date.strftime('%Y-%m-%d')] = date_positions
                    print(f"Date {date.strftime('%Y-%m-%d')}: {len(date_positions)} positions")

                print(f"Total all_positions_data: {len(all_positions_data)} dates")

                # 获取最后一个交易日
                last_date = all_dates[-1]
                last_position = positions[last_date]
                
                # 提取股票持仓数据
                if hasattr(last_position, 'position'):
                    pos_dict = last_position.position
                elif isinstance(last_position, dict):
                    pos_dict = last_position
                else:
                    pos_dict = {}
                
                # 提取股票数据
                stock_keys = [stock for stock in pos_dict.keys() if stock not in ['cash', 'now_account_value']]
                
                # 获取股票名称
                stock_names, success, error = self.get_stock_names(stock_keys)
                
                # 遍历持仓数据
                for stock in stock_keys:
                    info = pos_dict[stock]
                    if isinstance(info, dict):
                        # 获取股票名称
                        stock_name = stock_names.get(stock, stock)
                        
                        # 计算盈利情况
                        current_price = float(info.get('price', 0))
                        amount = float(info.get('amount', 0))  # 持仓数量
                        count_day = int(info.get('count_day', 0))  # 持仓天数
                        
                        # 计算开仓日的价格作为成本价
                        avg_price = current_price
                        try:
                            # 获取开仓日期
                            open_date = last_date - pd.DateOffset(days=count_day)
                            
                            # 使用QLib的D.features()获取开仓日的价格
                            start_date = open_date - pd.DateOffset(days=2)
                            end_date = open_date + pd.DateOffset(days=2)
                            
                            price_data = D.features(
                                instruments=[stock],
                                fields=['$close'],
                                start_time=start_date.strftime('%Y-%m-%d'),
                                end_time=end_date.strftime('%Y-%m-%d')
                            )
                            
                            if not price_data.empty:
                                # 查找最接近开仓日期的价格
                                date_diffs = abs(price_data.index.get_level_values('datetime') - open_date)
                                min_diff_idx = date_diffs.argmin()
                                avg_price = float(price_data.iloc[min_diff_idx]['$close'])
                        except Exception as e:
                            pass
                        
                        # 计算盈利金额和收益率
                        profit = (current_price - avg_price) * amount if avg_price > 0 else 0
                        profit_rate = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
                        
                        # 添加到positions_data
                        positions_data.append({
                            'stock_code': stock,
                            'stock_name': stock_name,
                            'weight': float(info.get('weight', 0)),
                            'hold_days': count_day,
                            'amount': amount,
                            'cost_price': avg_price,
                            'current_price': current_price,
                            'hold_value': amount * current_price,
                            'profit': profit,
                            'profit_rate': profit_rate
                        })
            
            # 获取回测配置参数
            backtest_config = recorder.list_params()

            return {
                "success": True,
                "key_metrics": key_metrics,
                "cumulative_data": cumulative_data,
                "daily_data": daily_data,
                "positions": positions_data,
                "all_positions": all_positions_data,
                "last_date": last_date.strftime('%Y-%m-%d') if positions else None,
                "config": {
                    "market": backtest_config.get("backtest_market", "csi300"),
                    "benchmark": backtest_config.get("backtest_benchmark", "SH000300"),
                    "start_date": backtest_config.get("backtest_start_date", ""),
                    "end_date": backtest_config.get("backtest_end_date", ""),
                    "initial_account": backtest_config.get("initial_account", 1000000),
                    "topk": backtest_config.get("topk", 10),
                    "n_drop": backtest_config.get("n_drop", 1),
                    "strategy_type": backtest_config.get("strategy_type", "TopkDropoutStrategy")
                }
            }
        except Exception as e:
            return {"success": False, "message": f"获取回测结果失败: {str(e)}"}
    
    def preview_data(self, market="csi300"):
        """预览数据"""
        try:
            # 使用固定的股票代码和日期范围
            fixed_stocks = ['SH600000', 'SH600036', 'SH601318', 'SH601398', 'SH601857']

            # 使用固定的日期范围（最近30天）
            import datetime
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')

            # 定义要获取的特征
            fields = ['$open', '$close', '$high', '$low', '$volume', '$amount']

            # 尝试获取数据
            data = None
            selected_stock = None

            for stock_code in fixed_stocks:
                try:
                    print(f"尝试获取股票数据: {stock_code}")
                    data = D.features(
                        instruments=[stock_code],
                        fields=fields,
                        start_time=start_date,
                        end_time=end_date
                    )
                    if data is not None and len(data) > 0:
                        selected_stock = stock_code
                        print(f"成功获取股票数据: {stock_code}, 共 {len(data)} 条记录")
                        break
                except Exception as e:
                    print(f"获取股票 {stock_code} 数据失败: {str(e)}")
                    continue

            if data is None or len(data) == 0:
                return {"success": False, "message": "无法获取股票数据，请检查数据是否正确下载"}

            # 转换数据格式
            data_dict = {
                "dates": [],
                "stocks": [],
                "data": []
            }

            # 提取日期
            try:
                dates = data.index.get_level_values('datetime').unique()
                dates_list = [date.strftime('%Y-%m-%d') for date in dates]
                dates_list = dates_list[-10:][::-1]
                data_dict["dates"] = dates_list
            except Exception as e:
                data_dict["dates"] = [str(i) for i in range(10)][::-1]

            # 提取股票
            data_dict["stocks"] = [selected_stock]

            # 提取数据
            try:
                open_list = data['$open'].tolist() if '$open' in data.columns else []
                close_list = data['$close'].tolist() if '$close' in data.columns else []
                high_list = data['$high'].tolist() if '$high' in data.columns else []
                low_list = data['$low'].tolist() if '$low' in data.columns else []
                volume_list = data['$volume'].tolist() if '$volume' in data.columns else []
                amount_list = data['$amount'].tolist() if '$amount' in data.columns else []
                
                stock_data_dict = {
                    "stock": selected_stock,
                    "open": open_list[-10:][::-1],
                    "close": close_list[-10:][::-1],
                    "high": high_list[-10:][::-1],
                    "low": low_list[-10:][::-1],
                    "volume": volume_list[-10:][::-1],
                    "amount": amount_list[-10:][::-1]
                }
                data_dict["data"].append(stock_data_dict)
            except Exception as e:
                print(f"提取数据失败: {str(e)}")
                return {"success": False, "message": f"提取数据失败: {str(e)}"}

            if not data_dict["data"]:
                return {"success": False, "message": "无法提取数据"}

            return {
                "success": True,
                "market": market,
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": 1,
                "data": data_dict
            }
        except Exception as e:
            print(f"预览数据失败: {str(e)}")
            return {"success": False, "message": f"获取数据失败: {str(e)}"}

    def delete_train_recorder(self, recorder_id):
        """删除训练记录"""
        try:
            # 获取实验
            exp = R.get_exp(experiment_name="train_model")
            # 删除指定的记录器
            exp.delete_recorder(recorder_id)
            return {"success": True, "message": f"训练记录 {recorder_id} 已删除"}
        except Exception as e:
            return {"success": False, "message": f"删除训练记录失败: {str(e)}"}

    def delete_backtest_recorder(self, recorder_id):
        """删除回测记录"""
        try:
            # 获取回测实验
            exp = R.get_exp(experiment_name="backtest_analysis")
            # 删除指定的记录器
            exp.delete_recorder(recorder_id)
            return {"success": True, "message": f"回测记录 {recorder_id} 已删除"}
        except Exception as e:
            return {"success": False, "message": f"删除回测记录失败: {str(e)}"}
