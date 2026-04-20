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
    
    def init_qlib(self, force=False):
        """初始化Qlib"""
        try:
            if force:
                qlib._initialized = False

            if not hasattr(qlib, '_initialized') or not qlib._initialized:
                qlib.init(provider_uri=self.provider_uri, region=REG_CN)

            sqlite_uri = "sqlite:///mlflow.db"
            R.set_uri(sqlite_uri)
        except Exception as e:
            print(f"Qlib初始化失败: {e}")
            try:
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
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (code, market)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def update_stock_db(self):
        """更新股票信息数据库（获取各指数成分股）"""
        try:
            import akshare as ak
            
            DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('DELETE FROM stock_info')
            
            index_map = {
                "csi300": "000300",
                "csi500": "000905",
                "csi800": "000906",
                "csi1000": "000852"
            }
            
            total_count = 0
            for market, index_code in index_map.items():
                try:
                    index_stocks = ak.index_stock_cons(symbol=index_code)
                    for _, row in index_stocks.iterrows():
                        code = str(row.get('品种代码', row.get('code', '')))
                        name = row.get('品种名称', row.get('name', ''))
                        if code:
                            cursor.execute('INSERT OR REPLACE INTO stock_info (code, name, market, updated_at) VALUES (?, ?, ?, ?)',
                                          (code, name, market, updated_at))
                            total_count += 1
                    print(f"{market}: 获取 {len(index_stocks)} 只股票")
                except Exception as e:
                    print(f"获取 {market} 成分股失败: {e}")
            
            conn.commit()
            conn.close()
            return {"success": True, "message": f"已更新 {total_count} 只股票信息"}
        except Exception as e:
            print(f"更新股票数据库失败: {e}")
            return {"success": False, "message": f"更新失败: {str(e)}"}
    
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
    
    def check_data_release(self):
        """检查 Qlib 数据发布日期"""
        try:
            import json
            api_url = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
            
            req = urllib.request.Request(api_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                release_info = json.loads(response.read().decode('utf-8'))
                release_date = release_info.get('published_at', '')
                if release_date:
                    release_date = release_date.split('T')[0]
                return {
                    "success": True,
                    "release_date": release_date,
                    "message": f"数据发布日期: {release_date}"
                }
        except Exception as e:
            print(f"获取发布日期失败: {e}")
            return {"success": False, "message": f"获取发布日期失败: {str(e)}"}
    
    def download_data_with_progress(self, progress_callback=None):
        """下载Qlib数据（Python实现，带进度回调）"""
        try:
            import shutil
            import tarfile
            import json
            
            url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
            api_url = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
            target_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data")
            backup_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data_backup")
            temp_file = os.path.join(os.path.expanduser("~/.qlib"), "qlib_bin.tar.gz")
            
            def report_progress(progress, message):
                print(f"[下载进度] {progress}%: {message}")
                if progress_callback:
                    progress_callback(progress, message)
            
            report_progress(0, "检查数据发布日期...")
            
            # 获取最新 release 信息
            release_date = None
            try:
                req = urllib.request.Request(api_url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    release_info = json.loads(response.read().decode('utf-8'))
                    release_date = release_info.get('published_at', '')
                    if release_date:
                        release_date = release_date.split('T')[0]
                        report_progress(5, f"数据发布日期: {release_date}")
            except Exception as e:
                print(f"获取发布日期失败: {e}")
                report_progress(5, "无法获取发布日期")
            
            report_progress(10, "开始下载Qlib数据...")
            
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            # 获取文件大小
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as response:
                file_size = int(response.headers.get('Content-Length', 0))
            
            if file_size > 0:
                report_progress(15, f"文件大小: {file_size / 1024 / 1024:.2f} MB")
            
            report_progress(20, "开始下载...")
            
            # 下载文件
            downloaded = 0
            chunk_size = 1024 * 1024
            
            with urllib.request.urlopen(url, timeout=300) as response:
                with open(temp_file, 'wb') as f:
                    while True:
                        data = response.read(chunk_size)
                        if not data:
                            break
                        f.write(data)
                        downloaded += len(data)
                        progress = int(20 + downloaded / file_size * 60)
                        report_progress(progress, f"下载中 {downloaded/1024/1024:.1f}MB / {file_size/1024/1024:.1f}MB")
            
            report_progress(80, "下载完成")
            
            # 备份旧数据
            if os.path.exists(target_dir):
                report_progress(85, "备份旧数据...")
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.move(target_dir, backup_dir)
            
            report_progress(88, "开始解压...")
            
            # 解压文件
            os.makedirs(target_dir, exist_ok=True)
            with tarfile.open(temp_file, "r:gz") as tar:
                members = tar.getmembers()
                valid_members = [m for m in members if '/' in m.name]
                total = len(valid_members)
                for i, member in enumerate(valid_members):
                    member.name = member.name.split('/', 1)[1]
                    tar.extract(member, target_dir)
                    if i % 100 == 0 or i == total - 1:
                        progress = int(88 + (i + 1) / total * 10)
                        report_progress(progress, f"解压中 {i+1}/{total}")
            
            # 删除临时文件
            os.remove(temp_file)
            
            report_progress(100, "数据下载完成")
            
            self.init_qlib(force=True)
            return {"success": True, "message": "Qlib数据已更新完成！", "release_date": release_date}
            
        except Exception as e:
            print(f"数据下载失败: {str(e)}")
            return {"success": False, "message": f"数据下载失败: {str(e)}"}
    
    def download_data(self):
        """下载Qlib数据（备份旧数据后替换，多线程加速）"""
        return self.download_data_with_progress(None)
    
    def train_model_with_progress(self, progress_callback, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed=42, num_threads=1):
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
                        "num_threads": num_threads,
                        "random_state": seed,
                        "deterministic": True,
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
            
            # 生成记录名称
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            train_recorder_name = f"{current_time}_{market}_{train_start_date.replace('-', '')}_to_{train_end_date.replace('-', '')}"
            
            # 训练模型 - 使用 recorder_name 参数指定名称
            with R.start(experiment_name="train_model", recorder_name=train_recorder_name):
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
                    fit_start_time=train_start_date,
                    fit_end_time=valid_end_date,
                    model_type=model_type,
                    lr=lr,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    **flatten_dict(task)
                )
                
                # 获取 recorder id
                recorder = R.get_recorder()
                rid = recorder.id
                
                progress_callback(60, "训练模型中...")
                model.fit(dataset)

                progress_callback(75, "生成预测结果...")
                # 在训练记录中生成预测信号和标签
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                progress_callback(80, "计算评估指标...")
                # 加载预测和标签数据用于评估
                pred = recorder.load_object("pred.pkl")
                label = recorder.load_object("label.pkl")

                # 合并预测和标签
                pred_label = pd.concat([pred, label], axis=1)

                # 计算IC指标
                from qlib.workflow.record_temp import calc_ic, calc_long_short_prec, calc_long_short_return
                ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])

                # 计算长短期精度
                long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], label.iloc[:, 0], is_alpha=True)

                # 计算长短期收益
                long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])

                # 计算关键指标
                metrics = {
                    "IC": float(ic.mean()),
                    "ICIR": float(ic.mean() / ic.std()),
                    "Rank_IC": float(ric.mean()),
                    "Rank_ICIR": float(ric.mean() / ric.std()),
                    "Long_precision": float(long_pre.mean()),
                    "Short_precision": float(short_pre.mean()),
                    "Long_Short_Avg_Return": float(long_short_r.mean()),
                    "Long_Short_Avg_Sharpe": float(long_short_r.mean() / long_short_r.std()) if long_short_r.std() != 0 else 0,
                }

                # 保存指标
                R.log_metrics(**metrics)

                # 生成分组收益数据用于可视化
                pred_label_df = pred_label.copy()
                pred_label_df.columns = ['score', 'label']

                # 计算分组收益 (5组)
                N = 5
                pred_label_drop = pred_label_df.dropna(subset=['score'])
                pred_label_sorted = pred_label_drop.sort_values('score', ascending=False)

                group_returns = {}
                for i in range(N):
                    group_name = f"Group{i+1}"
                    group_data = pred_label_sorted.groupby(level='datetime', group_keys=False).apply(
                        lambda x: x.iloc[len(x)//N*i:len(x)//N*(i+1)]['label'].mean()
                    )
                    group_returns[group_name] = group_data.dropna().tolist()

                # 多空收益
                long_short = pd.Series(group_returns['Group1']) - pd.Series(group_returns['Group5'])
                long_avg = pd.Series(group_returns['Group1']) - pred_label_df.groupby(level='datetime')['label'].mean()

                group_returns['long_short'] = long_short.dropna().tolist()
                group_returns['long_average'] = long_avg.dropna().tolist()
                group_returns['dates'] = pred_label_df.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()

                # 计算测试集收益（按组）
                test_returns = {}
                test_pred_label = pred_label_df[
                    (pred_label_df.index.get_level_values('datetime') >= test_start_date) &
                    (pred_label_df.index.get_level_values('datetime') <= test_end_date)
                ].copy()

                if len(test_pred_label) > 0:
                    test_pred_label_sorted = test_pred_label.sort_values('score', ascending=False)

                    for i in range(N):
                        group_name = f"Group{i+1}"
                        group_data = test_pred_label_sorted.groupby(level='datetime', group_keys=False).apply(
                            lambda x: x.iloc[len(x)//N*i:len(x)//N*(i+1)]['label'].mean()
                        )
                        test_returns[group_name] = group_data.dropna().tolist()

                    # 测试集多空收益
                    test_long_short = pd.Series(test_returns['Group1']) - pd.Series(test_returns['Group5'])
                    test_returns['long_short'] = test_long_short.dropna().tolist()
                    test_returns['dates'] = test_pred_label.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()
                else:
                    # 如果没有测试集数据，使用空数据
                    for i in range(N):
                        test_returns[f"Group{i+1}"] = []
                    test_returns['long_short'] = []
                    test_returns['dates'] = []

                # 计算IC序列
                ic_data = {
                    'dates': ic.index.strftime('%Y-%m-%d').tolist(),
                    'ic': ic.tolist(),
                    'rank_ic': ric.tolist()
                }

                # 保存可视化数据
                viz_data = {
                    'metrics': metrics,
                    'group_returns': group_returns,
                    'test_returns': test_returns,
                    'ic_data': ic_data
                }
                R.save_objects(viz_data=viz_data)

                progress_callback(85, "保存模型...")

                # 获取 recorder 以便后续加载
                recorder = R.get_recorder()
                rid = recorder.id
                print(f"保存模型到 recorder: {rid}")

                # 保存模型 - 使用 recorder 直接保存
                import os
                import pickle
                import tempfile

                # 创建临时文件保存模型
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    pickle.dump(model, tmp)
                    tmp_path = tmp.name

                # 使用 recorder 保存模型文件
                recorder.save_objects(trained_model=model)

                # 同时保存为本地文件作为备份
                mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                model_backup_dir = os.path.join(mlruns_dir, "model_backups")
                os.makedirs(model_backup_dir, exist_ok=True)
                backup_path = os.path.join(model_backup_dir, f"{rid}_trained_model.pkl")
                with open(backup_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"模型备份保存到: {backup_path}")

                # 删除临时文件
                os.unlink(tmp_path)

                print(f"模型保存完成")

            progress_callback(90, "训练完成，生成记录ID...")

            return {"success": True, "recorder_id": rid, "message": "模型训练完成！"}
        except Exception as e:
            progress_callback(-1, f"模型训练失败: {str(e)}")
            return {"success": False, "message": f"模型训练失败: {str(e)}"}

    def train_model(self, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed=42, num_threads=1):
        """训练模型"""
        return self.train_model_with_progress(None, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed, num_threads)
    
    def backtest_model(self, recorder_id, market, benchmark, start_date, end_date,
                      initial_account, topk, n_drop, hold_days, stop_loss, strategy_type):
        """执行回测"""
        try:
            # 获取指定ID的记录器
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")

            # 检查记录器中的对象列表
            try:
                # 尝试列出记录器中的对象
                artifacts = recorder.list_artifacts()
                print(f"记录器 {recorder_id} 中的对象: {artifacts}")
            except Exception as e:
                print(f"无法列出对象: {e}")

            # 加载模型
            try:
                model = recorder.load_object("trained_model")
                print(f"成功加载模型 from recorder {recorder_id}")
            except Exception as e:
                error_msg = str(e)
                print(f"加载模型失败: {error_msg}")
                # 尝试其他可能的名称
                try:
                    model = recorder.load_object("model")
                    print(f"成功从 'model' 加载模型")
                except:
                    # 如果都不行，尝试从pickle文件直接加载
                    import os
                    import glob
                    mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                    print(f"搜索mlruns目录: {mlruns_dir}")

                    # 首先检查备份目录
                    model_backup_dir = os.path.join(mlruns_dir, "model_backups")
                    backup_path = os.path.join(model_backup_dir, f"{recorder_id}_trained_model.pkl")

                    model_loaded = False
                    if os.path.exists(backup_path):
                        try:
                            import pickle
                            with open(backup_path, 'rb') as f:
                                model = pickle.load(f)
                            print(f"成功从备份文件加载模型: {backup_path}")
                            model_loaded = True
                        except Exception as e2:
                            print(f"从备份文件加载失败: {e2}")

                    if not model_loaded:
                        print(f"备份文件不存在或加载失败: {backup_path}")

                        # 尝试找到模型文件
                        model_paths = []
                        for root, dirs, files in os.walk(mlruns_dir):
                            if recorder_id in root:
                                for file in files:
                                    if file.endswith('.pkl') or file == 'trained_model':
                                        model_paths.append(os.path.join(root, file))

                        print(f"找到的模型文件: {model_paths}")

                        if not model_paths:
                            return {"success": False, "message": f"回测和分析失败: 无法找到训练好的模型。请确保模型训练成功完成。错误: {error_msg}"}

                        # 尝试从找到的路径加载
                        for model_path in model_paths:
                            try:
                                import pickle
                                with open(model_path, 'rb') as f:
                                    model = pickle.load(f)
                                print(f"成功从 {model_path} 加载模型")
                                model_loaded = True
                                break
                            except Exception as e2:
                                print(f"从 {model_path} 加载失败: {e2}")
                                continue

                        if not model_loaded:
                            return {"success": False, "message": f"回测和分析失败: 无法加载训练好的模型。错误: {error_msg}"}

            # 读取训练时的 fit 时间参数（用于特征标准化，防止数据泄露）
            train_params = recorder.list_params()
            fit_start_time = train_params.get('fit_start_time', start_date)
            fit_end_time = train_params.get('fit_end_time', end_date)

            # 加载数据集（需要重新构建）
            data_handler_config = {
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": fit_start_time,  # 使用训练时的统计时间
                "fit_end_time": fit_end_time,      # 使用训练时的统计时间
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
                        "hold_days": hold_days,
                        "stop_loss": stop_loss / 100.0,  # 转换为小数
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
            
            # 生成回测记录名称: 当前日期时间_市场_起止日期
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_recorder_name = f"{current_time}_{market}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}"
            
            # 执行回测和分析 - 使用 recorder_name 参数指定名称
            with R.start(experiment_name="backtest_analysis", recorder_name=backtest_recorder_name):
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
                    hold_days=hold_days,
                    stop_loss=stop_loss,
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
                # 获取 recorder 名称，如果没有则使用 id
                rec_name = rec_info.get('name', rec_id)
                recorder_list.append({
                    "id": rec_id,
                    "name": rec_name,
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
                # 获取 recorder 名称，如果没有则使用 id
                rec_name = rec_info.get('name', rec_id)
                recorder_list.append({
                    "id": rec_id,
                    "name": rec_name,
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
            
            total_return = float((report_normal_df['cumulative_return'].iloc[-1] - 1) * 100)
            bench_return = float((report_normal_df['cumulative_bench'].iloc[-1] - 1) * 100)
            excess_return_total = float(total_return - bench_return)
            
            # 获取其他分析指标
            annualized_return = float(analysis_df.loc[('excess_return_with_cost', 'annualized_return'), 'risk'])
            information_ratio = float(analysis_df.loc[('excess_return_with_cost', 'information_ratio'), 'risk'])
            max_drawdown = float(analysis_df.loc[('excess_return_with_cost', 'max_drawdown'), 'risk'])
            
            # 关键指标
            key_metrics = {
                "total_return": float(round(total_return, 2)),
                "bench_return": float(round(bench_return, 2)),
                "excess_return": float(round(excess_return_total, 2)),
                "annualized_return": float(round(annualized_return, 4)),
                "information_ratio": float(round(information_ratio, 3)),
                "max_drawdown": float(round(max_drawdown, 4))
            }
            
            # 累计收益数据
            cumulative_data = {
                "dates": report_normal_df.index.strftime('%Y-%m-%d').tolist(),
                "strategy": [float(x) for x in report_normal_df['cumulative_return'].tolist()],
                "benchmark": [float(x) for x in report_normal_df['cumulative_bench'].tolist()]
            }
            
            # 日收益数据
            daily_data = {
                "dates": report_normal_df.index.strftime('%Y-%m-%d').tolist(),
                "strategy": [float(x) for x in report_normal_df['return'].tolist()],
                "benchmark": [float(x) for x in report_normal_df['bench'].tolist()]
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
                                profit = float((current_price - avg_price) * amount)
                                profit_rate = float((current_price - avg_price) / avg_price * 100)

                            date_positions.append({
                                'stock_code': stock,
                                'stock_name': stock_name,
                                'weight': float(info.get('weight', 0)),
                                'hold_days': int(count_day),
                                'amount': float(amount),
                                'cost_price': float(avg_price),
                                'current_price': float(current_price),
                                'hold_value': float(amount * current_price),
                                'profit': float(round(profit, 2)),
                                'profit_rate': float(round(profit_rate, 2))
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
                        profit = float((current_price - avg_price) * amount) if avg_price > 0 else 0.0
                        profit_rate = float((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
                        
                        # 添加到positions_data
                        positions_data.append({
                            'stock_code': stock,
                            'stock_name': stock_name,
                            'weight': float(info.get('weight', 0)),
                            'hold_days': int(count_day),
                            'amount': float(amount),
                            'cost_price': float(avg_price),
                            'current_price': float(current_price),
                            'hold_value': float(amount * current_price),
                            'profit': float(profit),
                            'profit_rate': float(profit_rate)
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
                    "hold_days": backtest_config.get("hold_days", 3),
                    "stop_loss": backtest_config.get("stop_loss", 5.0),
                    "strategy_type": backtest_config.get("strategy_type", "TopkDropoutStrategy")
                }
            }
        except Exception as e:
            return {"success": False, "message": f"获取回测结果失败: {str(e)}"}
    
    def preview_data(self, market="csi300"):
        """预览数据"""
        try:
            DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            query_market = market if market != "all" else "csi300"
            cursor.execute('SELECT code FROM stock_info WHERE market = ? LIMIT 5', (query_market,))
            rows = cursor.fetchall()
            conn.close()
            
            fixed_stocks = []
            for row in rows:
                code = row[0]
                fixed_stocks.append(f"SH{code}" if code.startswith('6') else f"SZ{code}")
            
            if not fixed_stocks:
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

    def get_train_result(self, recorder_id):
        """获取训练结果（模型评估可视化数据）"""
        try:
            # 获取训练记录
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")

            # 获取参数
            params = recorder.list_params()

            # 辅助函数：安全处理数值（处理NaN和Inf）
            def safe_value(value):
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    if pd.isna(value) or np.isnan(value) or np.isinf(value):
                        return None
                    return float(value)
                return value

            def safe_list(values):
                return [safe_value(v) for v in values]

            # 尝试加载可视化数据
            viz_data = None
            try:
                viz_data = recorder.load_object("viz_data")
            except:
                pass

            # 尝试加载预测和标签数据（如果没有viz_data）
            if viz_data is None:
                try:
                    pred = recorder.load_object("pred.pkl")
                    label = recorder.load_object("label.pkl")

                    # 计算IC指标
                    from qlib.workflow.record_temp import calc_ic, calc_long_short_prec, calc_long_short_return
                    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
                    long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], label.iloc[:, 0], is_alpha=True)

                    ic_mean = ic.mean()
                    ic_std = ic.std()
                    ric_mean = ric.mean()
                    ric_std = ric.std()

                    metrics = {
                        "IC": safe_value(ic_mean),
                        "ICIR": safe_value(ic_mean / ic_std) if ic_std and ic_std != 0 else None,
                        "Rank_IC": safe_value(ric_mean),
                        "Rank_ICIR": safe_value(ric_mean / ric_std) if ric_std and ric_std != 0 else None,
                        "Long_precision": safe_value(long_pre.mean()),
                        "Short_precision": safe_value(short_pre.mean()),
                    }

                    # 生成分组收益数据
                    pred_label = pd.concat([pred, label], axis=1)
                    pred_label.columns = ['score', 'label']

                    N = 5
                    pred_label_drop = pred_label.dropna(subset=['score'])
                    pred_label_sorted = pred_label_drop.sort_values('score', ascending=False)

                    group_returns = {}
                    for i in range(N):
                        group_name = f"Group{i+1}"
                        group_data = pred_label_sorted.groupby(level='datetime', group_keys=False).apply(
                            lambda x: x.iloc[len(x)//N*i:len(x)//N*(i+1)]['label'].mean()
                        )
                        group_returns[group_name] = safe_list(group_data.dropna().tolist())

                    long_short = pd.Series(group_returns['Group1']) - pd.Series(group_returns['Group5'])
                    group_returns['long_short'] = safe_list(long_short.dropna().tolist())
                    group_returns['dates'] = pred_label.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()

                    # 尝试从参数获取测试集日期范围，如果不存在则使用全部数据
                    test_start = params.get('test_start_date', '')
                    test_end = params.get('test_end_date', '')
                    test_returns = {}

                    if test_start and test_end:
                        try:
                            test_pred_label = pred_label[
                                (pred_label.index.get_level_values('datetime') >= test_start) &
                                (pred_label.index.get_level_values('datetime') <= test_end)
                            ].copy()

                            if len(test_pred_label) > 0:
                                test_pred_label_sorted = test_pred_label.sort_values('score', ascending=False)

                                for i in range(N):
                                    group_name = f"Group{i+1}"
                                    group_data = test_pred_label_sorted.groupby(level='datetime', group_keys=False).apply(
                                        lambda x: x.iloc[len(x)//N*i:len(x)//N*(i+1)]['label'].mean()
                                    )
                                    test_returns[group_name] = safe_list(group_data.dropna().tolist())

                                test_long_short = pd.Series(test_returns['Group1']) - pd.Series(test_returns['Group5'])
                                test_returns['long_short'] = safe_list(test_long_short.dropna().tolist())
                                test_returns['dates'] = test_pred_label.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()
                        except Exception as e:
                            print(f"计算测试集收益失败: {e}")

                    ic_data = {
                        'dates': ic.index.strftime('%Y-%m-%d').tolist(),
                        'ic': safe_list(ic.tolist()),
                        'rank_ic': safe_list(ric.tolist())
                    }

                    viz_data = {
                        'metrics': metrics,
                        'group_returns': group_returns,
                        'test_returns': test_returns,
                        'ic_data': ic_data
                    }
                except Exception as e:
                    return {"success": False, "message": f"无法加载训练结果数据: {str(e)}"}
            else:
                # 如果viz_data存在，也需要处理其中的NaN值
                # 处理metrics
                if 'metrics' in viz_data:
                    viz_data['metrics'] = {k: safe_value(v) for k, v in viz_data['metrics'].items()}

                # 处理group_returns
                if 'group_returns' in viz_data:
                    for key in viz_data['group_returns']:
                        if key != 'dates':
                            viz_data['group_returns'][key] = safe_list(viz_data['group_returns'][key])

                # 处理test_returns（测试集收益）
                if 'test_returns' in viz_data:
                    for key in viz_data['test_returns']:
                        if key != 'dates':
                            viz_data['test_returns'][key] = safe_list(viz_data['test_returns'][key])

                # 处理ic_data
                if 'ic_data' in viz_data:
                    if 'ic' in viz_data['ic_data']:
                        viz_data['ic_data']['ic'] = safe_list(viz_data['ic_data']['ic'])
                    if 'rank_ic' in viz_data['ic_data']:
                        viz_data['ic_data']['rank_ic'] = safe_list(viz_data['ic_data']['rank_ic'])

            return {
                "success": True,
                "recorder_id": recorder_id,
                "params": params,
                "metrics": viz_data.get('metrics', {}) if viz_data else {},
                "group_returns": viz_data.get('group_returns', {}) if viz_data else {},
                "test_returns": viz_data.get('test_returns', {}) if viz_data else {},
                "ic_data": viz_data.get('ic_data', {}) if viz_data else {}
            }
        except Exception as e:
            return {"success": False, "message": f"获取训练结果失败: {str(e)}"}

    def delete_train_recorder(self, recorder_id):
        """删除训练记录"""
        try:
            # 获取实验
            exp = R.get_exp(experiment_name="train_model")
            # 删除指定的记录器
            exp.delete_recorder(recorder_id)
            
            # 清理mlruns目录中的对应文件
            self._clean_mlruns_files(recorder_id)
            
            return {"success": True, "message": f"训练记录 {recorder_id} 已删除"}
        except Exception as e:
            return {"success": False, "message": f"删除训练记录失败: {str(e)}"}

    def delete_all_train_recorders(self):
        """删除所有训练记录"""
        try:
            # 获取实验
            exp = R.get_exp(experiment_name="train_model")
            # 获取所有记录器
            recorders = exp.list_recorders(rtype="list")
            deleted_count = 0
            for recorder in recorders:
                try:
                    exp.delete_recorder(recorder.id)
                    # 清理mlruns目录中的对应文件
                    self._clean_mlruns_files(recorder.id)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除记录 {recorder.id} 失败: {e}")
            return {"success": True, "message": f"已删除 {deleted_count} 条训练记录"}
        except Exception as e:
            return {"success": False, "message": f"删除所有训练记录失败: {str(e)}"}

    def delete_backtest_recorder(self, recorder_id):
        """删除回测记录"""
        try:
            # 获取回测实验
            exp = R.get_exp(experiment_name="backtest_analysis")
            # 删除指定的记录器
            exp.delete_recorder(recorder_id)
            
            # 清理mlruns目录中的对应文件
            self._clean_mlruns_files(recorder_id)
            
            return {"success": True, "message": f"回测记录 {recorder_id} 已删除"}
        except Exception as e:
            return {"success": False, "message": f"删除回测记录失败: {str(e)}"}

    def _clean_mlruns_files(self, recorder_id):
        """清理mlruns目录中的文件"""
        try:
            import shutil
            # 遍历mlruns目录，查找包含recorder_id的目录
            mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
            if os.path.exists(mlruns_dir):
                for root, dirs, files in os.walk(mlruns_dir):
                    for dir_name in dirs:
                        if recorder_id in dir_name:
                            dir_path = os.path.join(root, dir_name)
                            if os.path.isdir(dir_path):
                                shutil.rmtree(dir_path)
                                print(f"已删除文件: {dir_path}")
        except Exception as e:
            print(f"清理mlruns文件失败: {e}")
