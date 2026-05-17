import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
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
    _qlib_initialized = False

    def __init__(self):
        import os
        self.provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        print(f"Qlib数据路径: {self.provider_uri}")
        print(f"路径是否存在: {os.path.exists(self.provider_uri)}")
        self.init_qlib()
        self.init_stock_db()

    def init_qlib(self, force=False):
        try:
            if force or not QlibService._qlib_initialized:
                qlib.init(provider_uri=self.provider_uri, region=REG_CN)
                QlibService._qlib_initialized = True

            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mlruns_dir = os.path.join(backend_dir, "mlruns")
            os.makedirs(mlruns_dir, exist_ok=True)
            sqlite_uri = f"sqlite:///{os.path.join(mlruns_dir, 'mlflow.db')}"
            R.set_uri(sqlite_uri)
            print(f"MLflow tracking URI: {sqlite_uri}")
        except Exception as e:
            print(f"Qlib初始化失败: {e}")
            try:
                provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
                qlib.init(provider_uri=provider_uri, region=REG_CN)
                QlibService._qlib_initialized = True
                print(f"使用绝对路径初始化Qlib成功: {provider_uri}")
                
                backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mlruns_dir = os.path.join(backend_dir, "mlruns")
                os.makedirs(mlruns_dir, exist_ok=True)
                sqlite_uri = f"sqlite:///{os.path.join(mlruns_dir, 'mlflow.db')}"
                R.set_uri(sqlite_uri)
            except Exception as e2:
                print(f"使用绝对路径初始化Qlib也失败: {e2}")

    def release_qlib(self):
        try:
            QlibService._qlib_initialized = False
            from qlib.data import D
            if hasattr(D, '_data'):
                D._data = None
            print("Qlib资源已释放")
        except Exception as e:
            print(f"释放Qlib资源失败: {e}")
    
    def init_stock_db(self):
        """初始化股票信息数据库，并确保表结构与当前版本一致"""
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

        # 兼容旧版本表结构：若缺少 market 列则迁移
        cursor.execute("PRAGMA table_info(stock_info)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if 'market' not in existing_columns:
            logger.info("stock_info 表缺少 market 列，执行迁移...")
            cursor.execute("ALTER TABLE stock_info ADD COLUMN market TEXT NOT NULL DEFAULT ''")
            # 重建主键约束需重建表
            cursor.execute('''
            CREATE TABLE stock_info_new (
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                market TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (code, market)
            )
            ''')
            cursor.execute('''
            INSERT INTO stock_info_new (code, name, market, updated_at)
            SELECT code, name, market, updated_at FROM stock_info
            ''')
            cursor.execute("DROP TABLE stock_info")
            cursor.execute("ALTER TABLE stock_info_new RENAME TO stock_info")
            logger.info("stock_info 表迁移完成")

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
        try:
            import shutil
            import tarfile
            import json
            import socket
            import hashlib
            
            url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
            api_url = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
            target_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data")
            backup_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data_backup")
            temp_file = os.path.join(os.path.expanduser("~/.qlib"), "qlib_bin.tar.gz")
            
            http_proxy = os.environ.get('HTTP_PROXY', '')
            https_proxy = os.environ.get('HTTPS_PROXY', '')
            
            if http_proxy or https_proxy:
                proxy_handler = urllib.request.ProxyHandler({
                    'http': http_proxy or https_proxy,
                    'https': https_proxy or http_proxy
                })
                opener = urllib.request.build_opener(proxy_handler)
                print(f"使用代理: HTTP={http_proxy}, HTTPS={https_proxy}")
            else:
                opener = urllib.request.build_opener()
                print("不使用代理")
            urllib.request.install_opener(opener)
            
            def report_progress(progress, message):
                print(f"[下载进度] {progress}%: {message}")
                if progress_callback:
                    progress_callback(progress, message)
            
            report_progress(0, "检查数据发布日期...")
            
            release_date = None
            try:
                req = urllib.request.Request(api_url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                with opener.open(req, timeout=30) as response:
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
            
            socket.setdefaulttimeout(600)
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if retry > 0:
                        report_progress(10, f"第 {retry + 1} 次重试下载...")
                    
                    req = urllib.request.Request(url)
                    req.add_header('User-Agent', 'Mozilla/5.0')
                    req.add_header('Accept', '*/*')
                    
                    with opener.open(req) as response:
                        file_size = int(response.headers.get('Content-Length', 0))
                        if file_size > 0:
                            report_progress(15, f"文件大小: {file_size / 1024 / 1024:.2f} MB")
                        else:
                            report_progress(15, "开始下载（未知文件大小）...")
                        
                        report_progress(20, "开始下载...")
                        
                        downloaded = 0
                        chunk_size = 1024 * 1024
                        
                        with open(temp_file, 'wb') as f:
                            while True:
                                try:
                                    data = response.read(chunk_size)
                                    if not data:
                                        break
                                    f.write(data)
                                    downloaded += len(data)
                                    if file_size > 0:
                                        progress = int(20 + downloaded / file_size * 60)
                                        report_progress(progress, f"下载中 {downloaded/1024/1024:.1f}MB / {file_size/1024/1024:.1f}MB")
                                    else:
                                        progress = int(20 + min(downloaded / (500 * 1024 * 1024) * 60, 60))
                                        report_progress(progress, f"下载中 {downloaded/1024/1024:.1f}MB")
                                except socket.timeout:
                                    print(f"读取超时，已下载 {downloaded/1024/1024:.1f}MB")
                                    if downloaded > 0:
                                        continue
                                    raise
                            f.flush()
                            os.fsync(f.fileno())
                    
                    actual_size = os.path.getsize(temp_file)
                    if file_size > 0 and actual_size != file_size:
                        print(f"文件大小不匹配: 期望 {file_size}, 实际 {actual_size}")
                        if retry < max_retries - 1:
                            continue
                    
                    try:
                        with tarfile.open(temp_file, "r:gz") as test_tar:
                            test_tar.getmembers()
                        print("文件完整性验证通过")
                        break
                    except tarfile.TarError as e:
                        print(f"文件完整性验证失败: {e}")
                        if retry < max_retries - 1:
                            os.remove(temp_file)
                            continue
                        raise Exception(f"下载文件损坏，重试 {max_retries} 次后仍失败")
                    
                except urllib.error.URLError as e:
                    print(f"下载失败: {e}")
                    if retry < max_retries - 1:
                        continue
                    raise
                except Exception as e:
                    print(f"下载异常: {e}")
                    if retry < max_retries - 1:
                        continue
                    raise
            
            report_progress(80, "下载完成")
            
            if os.path.exists(target_dir):
                report_progress(85, "备份旧数据...")
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.move(target_dir, backup_dir)
            
            report_progress(88, "开始解压...")
            
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
    
    MODEL_REGISTRY = {
        "LGBModel": "qlib.contrib.model.gbdt",
        "XGBModel": "qlib.contrib.model.xgboost",
        "CatBoostModel": "qlib.contrib.model.catboost_model",
        "Linear": "qlib.contrib.model.linear",
        "DEnsembleModel": "qlib.contrib.model.double_ensemble",
        "LSTM": "qlib.contrib.model.pytorch_lstm",
        "GRU": "qlib.contrib.model.pytorch_gru",
        "ALSTM": "qlib.contrib.model.pytorch_alstm",
        "DNN": "qlib.contrib.model.pytorch_nn",
        "GATs": "qlib.contrib.model.pytorch_gats",
        "TCN": "qlib.contrib.model.pytorch_tcn",
        "SFM": "qlib.contrib.model.pytorch_sfm",
        "TabnetModel": "qlib.contrib.model.pytorch_tabnet",
    }

    HANDLER_REGISTRY = {
        "Alpha158": "qlib.contrib.data.handler",
        "Alpha360": "qlib.contrib.data.handler",
        "Alpha158DL": "qlib.contrib.data.loader",
        "Alpha360DL": "qlib.contrib.data.loader",
        "Alpha158vwap": "qlib.contrib.data.handler",
        "Alpha360vwap": "qlib.contrib.data.handler",
    }

    def _get_model_config(self, model_type, lr, max_depth, num_leaves, subsample, colsample_bytree, seed, num_threads, handler_type="Alpha158"):
        module_path = self.MODEL_REGISTRY.get(model_type, "qlib.contrib.model.gbdt")
        
        if model_type == "Linear":
            return {
                "class": "LinearModel",
                "module_path": module_path,
                "kwargs": {
                    "estimator": "ridge",
                    "alpha": 0.05,
                },
            }
        
        if model_type == "CatBoostModel":
            return {
                "class": "CatBoostModel",
                "module_path": module_path,
                "kwargs": {
                    "loss": "RMSE",
                    "learning_rate": lr,
                    "max_depth": max_depth,
                    "num_leaves": num_leaves,
                    "subsample": subsample,
                    "thread_count": num_threads,
                    "grow_policy": "Lossguide",
                    "bootstrap_type": "Poisson",
                },
            }
        
        if model_type == "DEnsembleModel":
            return {
                "class": "DEnsembleModel",
                "module_path": module_path,
                "kwargs": {
                    "base_model": "gbm",
                    "loss": "mse",
                    "num_models": 3,
                    "enable_sr": True,
                    "enable_fs": True,
                    "alpha1": 1,
                    "alpha2": 1,
                    "bins_sr": 10,
                    "bins_fs": 5,
                    "decay": 0.5,
                    "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
                    "sub_weights": [1, 1, 1],
                    "epochs": 28,
                    "colsample_bytree": colsample_bytree,
                    "learning_rate": 0.2,
                    "subsample": subsample,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": max_depth,
                    "num_leaves": num_leaves,
                    "num_threads": num_threads,
                    "verbosity": -1,
                },
            }
        
        if model_type == "TabnetModel":
            return {
                "class": "TabnetModel",
                "module_path": module_path,
                "kwargs": {
                    "n_epochs": 200,
                    "lr": lr,
                    "batch_size": 800,
                    "early_stopping_rounds": 50,
                    "seed": seed,
                },
            }
        
        pytorch_models = ["LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM"]
        if model_type in pytorch_models:
            # 根据因子类型动态设置 d_feat
            if "Alpha360" in handler_type:
                d_feat = 360
            elif "Alpha158" in handler_type:
                d_feat = 158
            else:
                d_feat = 158  # 默认值
            
            return {
                "class": model_type,
                "module_path": module_path,
                "kwargs": {
                    "d_feat": d_feat,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 200,
                    "lr": lr,
                    "batch_size": 800,
                    "early_stopping_rounds": 50,
                    "seed": seed,
                },
            }
        
        if model_type == "XGBModel":
            return {
                "class": "XGBModel",
                "module_path": module_path,
                "kwargs": {
                    "eval_metric": "rmse",
                    "colsample_bytree": colsample_bytree,
                    "eta": lr,
                    "max_depth": max_depth,
                    "n_estimators": 647,
                    "subsample": subsample,
                    "nthread": num_threads,
                },
            }
        
        kwargs = {
            "loss": "mse",
            "colsample_bytree": colsample_bytree,
            "learning_rate": lr,
            "subsample": subsample,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "num_threads": num_threads,
        }
        return {
            "class": model_type,
            "module_path": module_path,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_label_config(label_horizon=1):
        """构建标签配置，支持多周期预测

        采用 Qlib benchmarks_dynamic 标准公式:
            Ref($close, -(horizon+1)) / Ref($close, -1) - 1
        即从下一个交易日收盘到 horizon+1 天后收盘的收益率，
        避免使用当天收盘价（可能存在信息泄露）。

        Parameters
        ----------
        label_horizon : int
            预测周期（天），1=次日收益率，5=周收益率，20=月收益率

        Returns
        -------
        list : Qlib handler labels 配置
        """
        if label_horizon <= 1:
            return None  # 使用 handler 默认标签 (Ref($close,-2)/$close(-1)-1)
        # 格式匹配官方 Rolling.basic_task() (qlib/contrib/rolling/base.py L171-173):
        # 单层 list，只包含表达式字符串，名称由 handler 自动推导
        return ["Ref($close, -%d) / Ref($close, -1) - 1" % (label_horizon + 1)]

    def train_model_with_progress(self, progress_callback, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed=42, num_threads=1, handler_type="Alpha158", label_horizon=1):
        """训练模型（带进度回调）"""
        rid = None
        try:
            progress_callback(0, "开始训练...")

            horizon_label = f"（{label_horizon}日收益率）" if label_horizon > 1 else ""
            progress_callback(10, f"配置数据处理参数...{horizon_label}")

            # 官方 LGBModel YAML 不使用自定义 processors，由 Alpha158 handler 使用默认处理
            # 只有 Linear 和 DL 模型需要自定义 processors（RobustZScoreNorm + CSRankNorm）
            is_tree_model = model_type in ("LGBModel", "XGBModel", "CatBoostModel", "DEnsembleModel")

            data_handler_config = {
                "start_time": train_start_date,
                "end_time": test_end_date,
                "fit_start_time": train_start_date,
                "fit_end_time": train_end_date,
                "instruments": market,
            }

            if not is_tree_model:
                # Linear / DL 模型需要自定义 processors（参考官方 linear YAML）
                data_handler_config["infer_processors"] = [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ]
                data_handler_config["learn_processors"] = [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ]

            # 多周期标签：覆盖 handler 默认的次日收益率
            label_config = self._build_label_config(label_horizon)
            if label_config is not None:
                data_handler_config["label"] = label_config

            progress_callback(20, "构建任务配置...")

            model_config = self._get_model_config(model_type, lr, max_depth, num_leaves, subsample, colsample_bytree, seed, num_threads, handler_type)

            handler_module_path = self.HANDLER_REGISTRY.get(handler_type, "qlib.contrib.data.handler")

            task = {
                "model": model_config,
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": handler_type,
                            "module_path": handler_module_path,
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
            
            model = init_instance_by_config(task["model"])
            
            progress_callback(40, "初始化数据集...")
            
            dataset = init_instance_by_config(task["dataset"])

            # 提取特征列名，用于后续因子重要性展示
            feature_names = []
            try:
                train_features = dataset.prepare("train", col_set="feature")
                feature_names = list(train_features.columns)
            except Exception:
                pass
            
            progress_callback(50, "开始模型训练...")
            
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            train_recorder_name = f"{current_time}_{model_type}_{market}_{train_start_date.replace('-', '')}_to_{train_end_date.replace('-', '')}"
            
            with R.start(experiment_name="train_model", recorder_name=train_recorder_name):
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
                    fit_end_time=train_end_date,
                    handler_type=handler_type,
                    model_type=model_type,
                    label_horizon=label_horizon,
                    lr=lr,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    **flatten_dict(task)
                )
                
                recorder = R.get_recorder()
                rid = recorder.id
                
                progress_callback(60, "训练模型中...")
                model.fit(dataset)

                progress_callback(75, "生成预测结果...")
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                sar = SigAnaRecord(recorder)
                sar.generate()

                progress_callback(80, "计算评估指标...")
                pred = recorder.load_object("pred.pkl")
                label = recorder.load_object("label.pkl")

                pred_label = pd.concat([pred, label], axis=1)

                from qlib.workflow.record_temp import calc_ic, calc_long_short_prec, calc_long_short_return
                ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])

                long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], label.iloc[:, 0], is_alpha=True)

                long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])

                metrics = {
                    "IC": float(ic.mean()),
                    "ICIR": float(ic.mean() / ic.std()) if ic.std() != 0 else 0,
                    "Rank_IC": float(ric.mean()),
                    "Rank_ICIR": float(ric.mean() / ric.std()) if ric.std() != 0 else 0,
                    "Long_precision": float(long_pre.mean()),
                    "Short_precision": float(short_pre.mean()),
                    "Long_Short_Avg_Return": float(long_short_r.mean()),
                    "Long_Short_Avg_Sharpe": float(long_short_r.mean() / long_short_r.std()) if long_short_r.std() != 0 else 0,
                }

                R.log_metrics(**metrics)

                pred_label_df = pred_label.copy()
                pred_label_df.columns = ['score', 'label']

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

                long_short = pd.Series(group_returns['Group1']) - pd.Series(group_returns['Group5'])
                long_avg = pd.Series(group_returns['Group1']) - pred_label_df.groupby(level='datetime')['label'].mean()

                group_returns['long_short'] = long_short.dropna().tolist()
                group_returns['long_average'] = long_avg.dropna().tolist()
                group_returns['dates'] = pred_label_df.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()

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

                    test_long_short = pd.Series(test_returns['Group1']) - pd.Series(test_returns['Group5'])
                    test_returns['long_short'] = test_long_short.dropna().tolist()
                    test_returns['dates'] = test_pred_label.index.get_level_values('datetime').unique().strftime('%Y-%m-%d').tolist()
                else:
                    for i in range(N):
                        test_returns[f"Group{i+1}"] = []
                    test_returns['long_short'] = []
                    test_returns['dates'] = []

                ic_data = {
                    'dates': ic.index.strftime('%Y-%m-%d').tolist(),
                    'ic': ic.tolist(),
                    'rank_ic': ric.tolist()
                }

                viz_data = {
                    'metrics': metrics,
                    'group_returns': group_returns,
                    'test_returns': test_returns,
                    'ic_data': ic_data
                }
                R.save_objects(viz_data=viz_data)

                progress_callback(85, "保存模型...")

                recorder = R.get_recorder()
                rid = recorder.id
                print(f"保存模型到 recorder: {rid}")

                recorder.save_objects(trained_model=model)

                # 保存特征列名（供因子重要性展示使用）
                if feature_names:
                    recorder.save_objects(feature_names=feature_names)

                import os
                import pickle

                mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                model_backup_dir = os.path.join(mlruns_dir, "model_backups")
                os.makedirs(model_backup_dir, exist_ok=True)
                backup_path = os.path.join(model_backup_dir, f"{rid}_trained_model.pkl")
                with open(backup_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"模型备份保存到: {backup_path}")

                print(f"模型保存完成")

            progress_callback(90, "训练完成，生成记录ID...")

            return {"success": True, "recorder_id": rid, "message": "模型训练完成！"}
        except Exception as e:
            progress_callback(-1, f"模型训练失败: {str(e)}")
            return {"success": False, "message": f"模型训练失败: {str(e)}"}

    def train_model(self, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed=42, num_threads=1, handler_type="Alpha158", label_horizon=1):
        """训练模型"""
        def noop_callback(progress, message):
            pass
        return self.train_model_with_progress(noop_callback, market, benchmark, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, test_start_date, test_end_date,
                    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree,
                    seed, num_threads, handler_type, label_horizon)

    def train_rolling_with_progress(self, progress_callback, market, benchmark,
                                     train_start_date, train_end_date,
                                     test_start_date, test_end_date,
                                     model_type, lr, max_depth, num_leaves,
                                     subsample, colsample_bytree, seed=42,
                                     num_threads=1, handler_type="Alpha158",
                                     label_horizon=5, rolling_step=20):
        """滚动重训练：按 rolling_step 滑动窗口多次训练并合并预测

        原理参考 Qlib benchmarks_dynamic / Rolling Retrain (RR):
        - 将测试期按 rolling_step（默认20交易日≈1个月）切片
        - 每个切片都用截止到该时间点的最新数据重新训练模型
        - 合并所有切片预测，计算整体 IC/ICIR
        - 这是 IC 从 ~0.04 提升到 ~0.09 的最大贡献因素

        Parameters
        ----------
        rolling_step : int
            滚动步长（交易日数），默认20（约1个月）
        label_horizon : int
            预测周期，滚动训练建议使用 20（月度）以获得最高 IC
        """
        try:
            progress_callback(0, "初始化滚动训练...")

            # 获取测试期间的交易日历
            trading_days = D.calendar(freq="day",
                                       start_time=test_start_date,
                                       end_time=test_end_date)
            trading_days = pd.to_datetime(trading_days)

            if len(trading_days) < rolling_step:
                return {"success": False,
                        "message": f"测试期间交易日不足 {rolling_step} 天，无法滚动训练"}

            # 按 rolling_step 切分测试期
            n_rolls = len(trading_days) // rolling_step
            if n_rolls == 0:
                n_rolls = 1

            progress_callback(5, f"规划滚动窗口：{n_rolls} 轮，每轮 {rolling_step} 交易日")

            all_preds = []
            all_labels = []
            all_ics = []
            all_rics = []
            last_rid = None

            handler_module_path = self.HANDLER_REGISTRY.get(handler_type, "qlib.contrib.data.handler")
            model_config = self._get_model_config(model_type, lr, max_depth, num_leaves,
                                                   subsample, colsample_bytree, seed,
                                                   num_threads, handler_type)
            label_config = self._build_label_config(label_horizon)

            for i in range(n_rolls):
                roll_start_idx = i * rolling_step
                roll_end_idx = min((i + 1) * rolling_step - 1, len(trading_days) - 1)
                roll_test_start = trading_days[roll_start_idx].strftime('%Y-%m-%d')
                roll_test_end = trading_days[roll_end_idx].strftime('%Y-%m-%d')

                # 训练数据结束到该滚动窗口之前
                # 留出 label_horizon + 1 天作为截断，防止未来信息泄露
                trunc_days = label_horizon + 1
                roll_train_end_dt = trading_days[roll_start_idx] - pd.Timedelta(days=trunc_days + 5)
                roll_train_end = roll_train_end_dt.strftime('%Y-%m-%d')

                pct_base = int(5 + (i / n_rolls) * 80)
                progress_callback(pct_base,
                                   f"滚动训练 [{i+1}/{n_rolls}] "
                                   f"训练至 {roll_train_end} → 测试 {roll_test_start}~{roll_test_end}")

                is_tree_model = model_type in ("LGBModel", "XGBModel", "CatBoostModel", "DEnsembleModel")

                data_handler_config = {
                    "start_time": train_start_date,
                    "end_time": roll_test_end,
                    "fit_start_time": train_start_date,
                    "fit_end_time": roll_train_end,
                    "instruments": market,
                }

                if not is_tree_model:
                    data_handler_config["infer_processors"] = [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ]
                    data_handler_config["learn_processors"] = [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ]

                if label_config is not None:
                    data_handler_config["label"] = label_config

                # 计算验证集：取训练期最后 20% 作为验证
                roll_train_start_dt = pd.Timestamp(train_start_date)
                roll_train_end_dt2 = pd.Timestamp(roll_train_end)
                train_duration = (roll_train_end_dt2 - roll_train_start_dt).days
                valid_duration = max(int(train_duration * 0.2), 60)  # 至少60天
                roll_valid_start_dt = roll_train_end_dt2 - pd.Timedelta(days=valid_duration)
                roll_valid_start = roll_valid_start_dt.strftime('%Y-%m-%d')

                task = {
                    "model": model_config,
                    "dataset": {
                        "class": "DatasetH",
                        "module_path": "qlib.data.dataset",
                        "kwargs": {
                            "handler": {
                                "class": handler_type,
                                "module_path": handler_module_path,
                                "kwargs": data_handler_config,
                            },
                            "segments": {
                                "train": (train_start_date, roll_valid_start),
                                "valid": (roll_valid_start, roll_train_end),
                                "test": (roll_test_start, roll_test_end),
                            },
                        },
                    },
                }

                # 确保每轮开始前没有遗留的 MLflow run
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

                try:
                    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    roll_rec_name = (f"{current_time}_{model_type}_rolling_{i+1}of{n_rolls}_"
                                     f"{market}_{roll_test_start.replace('-', '')}")

                    # 与普通训练相同的模式：所有操作都在 R.start() 内部
                    with R.start(experiment_name="train_model", recorder_name=roll_rec_name):
                        R.log_params(
                            market=market, benchmark=benchmark,
                            train_start_date=train_start_date,
                            train_end_date=roll_train_end,
                            test_start_date=roll_test_start,
                            test_end_date=roll_test_end,
                            fit_start_time=train_start_date,
                            fit_end_time=roll_train_end,
                            handler_type=handler_type,
                            model_type=model_type,
                            label_horizon=label_horizon,
                            rolling_step=rolling_step,
                            rolling_index=i+1,
                            rolling_total=n_rolls,
                            rolling_rounds=n_rolls,
                            valid_start_date=roll_valid_start,
                            valid_end_date=roll_train_end,
                            is_rolling="true",
                        )

                        recorder = R.get_recorder()
                        last_rid = recorder.id

                        model = init_instance_by_config(task["model"])
                        dataset = init_instance_by_config(task["dataset"])
                        model.fit(dataset)

                        sr = SignalRecord(model, dataset, recorder)
                        sr.generate()

                        sar = SigAnaRecord(recorder)
                        sar.generate()

                        recorder.save_objects(trained_model=model)

                        # 保存特征列名（供因子重要性展示）
                        try:
                            train_features = dataset.prepare("train", col_set="feature")
                            recorder.save_objects(feature_names=list(train_features.columns))
                        except Exception:
                            pass

                        # 收集该窗口的预测
                        pred = recorder.load_object("pred.pkl")
                        label_obj = recorder.load_object("label.pkl")

                        all_preds.append(pred)
                        all_labels.append(label_obj)

                        from qlib.workflow.record_temp import calc_ic
                        ic, ric = calc_ic(pred.iloc[:, 0], label_obj.iloc[:, 0])
                        all_ics.append(ic)
                        all_rics.append(ric)

                        progress_callback(
                            pct_base + int(80 / n_rolls),
                            f"  轮 {i+1} IC={float(ic.mean()):.4f}"
                        )

                except Exception as e:
                    logger.warning(f"滚动轮 {i+1} 训练失败: {e}")
                    progress_callback(pct_base, f"  轮 {i+1} 失败: {str(e)[:50]}")
                finally:
                    # 确保每轮结束后 MLflow run 被关闭
                    try:
                        import mlflow
                        mlflow.end_run()
                    except Exception:
                        pass

            if not all_preds:
                progress_callback(-1, "所有滚动窗口训练均失败")
                return {"success": False, "message": "所有滚动窗口训练均失败"}

            # 合并所有滚动预测并计算整体指标
            progress_callback(88, "合并滚动预测，计算整体指标...")

            combined_pred = pd.concat(all_preds)
            combined_label = pd.concat(all_labels)
            # 去重（取最后一次的预测）
            combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]
            combined_label = combined_label[~combined_label.index.duplicated(keep='last')]

            aligned = pd.concat([combined_pred, combined_label], axis=1, join='inner')
            aligned.columns = ['score', 'label']

            from qlib.workflow.record_temp import calc_ic
            final_ic, final_ric = calc_ic(aligned['score'], aligned['label'])

            rolling_metrics = {
                "IC": float(final_ic.mean()),
                "ICIR": float(final_ic.mean() / final_ic.std()) if final_ic.std() != 0 else 0,
                "Rank_IC": float(final_ric.mean()),
                "Rank_ICIR": float(final_ric.mean() / final_ric.std()) if final_ric.std() != 0 else 0,
                "rolling_rounds": n_rolls,
                "rolling_step": rolling_step,
            }

            # 将合并后的预测和标签写回最后一轮的 recorder，
            # 这样评估界面 (get_train_result) 能读取到整体滚动指标
            try:
                last_recorder = R.get_recorder(recorder_id=last_rid, experiment_name="train_model")
                # 构造与原格式兼容的 DataFrame
                save_pred = aligned[['score']].copy()
                save_pred.columns = ['score']
                save_label = aligned[['label']].copy()
                save_label.columns = ['label']
                last_recorder.save_objects(**{"pred.pkl": save_pred, "label.pkl": save_label})
                logger.info(f"已将合并滚动预测({len(save_pred)}条)写回 recorder {last_rid}")
            except Exception as e:
                logger.warning(f"保存合并滚动预测失败（不影响训练结果）: {e}")

            progress_callback(95, f"滚动训练完成！整体 IC={rolling_metrics['IC']:.4f}, ICIR={rolling_metrics['ICIR']:.4f}")

            return {
                "success": True,
                "recorder_id": last_rid,
                "message": (f"滚动训练完成！{n_rolls} 轮，"
                            f"IC={rolling_metrics['IC']:.4f}, "
                            f"ICIR={rolling_metrics['ICIR']:.4f}"),
                "metrics": rolling_metrics,
            }

        except Exception as e:
            progress_callback(-1, f"滚动训练失败: {str(e)}")
            return {"success": False, "message": f"滚动训练失败: {str(e)}"}
    
    def backtest_model(self, recorder_id, market, benchmark, start_date, end_date,
                      initial_account, topk, n_drop, hold_days, stop_loss, strategy_type, seed=42,
                      deal_price="close", open_cost=0.0005, close_cost=0.0015,
                      limit_threshold=0.095, only_tradable=True):
        """执行回测
        
        Parameters
        ----------
        seed : int
            随机种子，确保回测结果可复现（默认42）
        deal_price : str
            成交价类型，"close" 或 "vwap"（默认 vwap，更贴近实盘）
        open_cost : float
            买入佣金费率
        close_cost : float
            卖出佣金+印花税费率
        limit_threshold : float
            涨跌停阈值
        only_tradable : bool
            是否只交易可买卖的股票
        """
        try:
            # 获取训练参数，检查日期重叠
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")
            train_params = recorder.list_params()
            train_end = train_params.get('fit_end_time')
            
            if train_end:
                backtest_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                train_end_date = datetime.datetime.strptime(train_end, '%Y-%m-%d')
                
                # 获取 label_horizon，计算安全间隙
                label_horizon = int(train_params.get('label_horizon', 1))
                safe_gap_days = label_horizon + 1  # label_horizon + 1 天的安全间隙
                min_backtest_start = train_end_date + datetime.timedelta(days=safe_gap_days)
                
                if backtest_start <= train_end_date:
                    return {
                        "success": False,
                        "message": f"回测开始日期({start_date})不能早于或等于训练结束日期({train_end})，否则会导致未来函数（数据泄露）。请调整回测开始日期。"
                    }
                
                if backtest_start < min_backtest_start:
                    return {
                        "success": False,
                        "message": f"回测开始日期({start_date})与训练结束日期({train_end})之间需要至少 {safe_gap_days} 天的安全间隙（考虑 label_horizon={label_horizon}），以避免标签计算导致的数据泄露。建议回测开始日期不早于 {min_backtest_start.strftime('%Y-%m-%d')}。"
                    }
            
            # 设置随机种子，确保回测结果可复现
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass
            
            print(f"[回测] 设置随机种子: {seed}")
            
            # 获取指定ID的记录器
            
            # 获取训练参数中的model_type
            train_params = recorder.list_params()
            model_type = train_params.get('model_type', 'Unknown')
            print(f"[回测] 使用模型: {model_type}")

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

            # 读取训练时的参数（handler类型、fit时间等），确保回测使用相同的数据处理
            train_params = recorder.list_params()
            fit_start_time = train_params.get('fit_start_time', start_date)
            fit_end_time = train_params.get('fit_end_time', end_date)
            handler_type = train_params.get('handler_type', 'Alpha158')
            label_horizon = int(train_params.get('label_horizon', 1))
            handler_module_path = self.HANDLER_REGISTRY.get(handler_type, "qlib.contrib.data.handler")

            # 加载数据集（需要重新构建）
            model_type = train_params.get('model_type', 'LGBModel')
            data_handler_config = {
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
                "instruments": market,
            }
            
            # 对于树模型（LGBM/XGB），不使用额外的 processors 避免破坏特征分布
            if model_type not in ["LGBModel", "XGBModel", "CatBoostModel", "DEnsembleModel"]:
                data_handler_config["infer_processors"] = [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ]
                data_handler_config["learn_processors"] = [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ]

            # 回测使用与训练相同的标签周期
            label_config = self._build_label_config(label_horizon)
            if label_config is not None:
                data_handler_config["label"] = label_config
            
            dataset = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": handler_type,
                        "module_path": handler_module_path,
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "test": (start_date, end_date),
                    },
                },
            }
            
            dataset = init_instance_by_config(dataset)
            
            # 回测配置 - 使用自定义策略
            if strategy_type == "TopkDropoutStrategy":
                # 使用增强版自定义策略
                strategy_config = {
                    "class": "EnhancedTopkDropoutStrategy",
                    "module_path": "backend.strategy",
                    "kwargs": {
                        "model": model,
                        "dataset": dataset,
                        "topk": topk,
                        "n_drop": n_drop,
                        "hold_days": hold_days,
                        "stop_loss": stop_loss / 100.0 if stop_loss > 0 else 0.0,
                        "only_tradable": only_tradable,
                        "forbid_all_trade_at_limit": False,
                    },
                }
            else:
                # 使用原始策略
                strategy_config = {
                    "class": strategy_type,
                    "module_path": "qlib.contrib.strategy.signal_strategy",
                    "kwargs": {
                        "model": model,
                        "dataset": dataset,
                        "topk": topk,
                        "n_drop": n_drop,
                        "only_tradable": only_tradable,
                        "forbid_all_trade_at_limit": False,
                    },
                }

            port_analysis_config = {
                "executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "generate_portfolio_metrics": True,
                    },
                },
                "strategy": strategy_config,
                "backtest": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "account": initial_account,
                    "benchmark": benchmark,
                    "exchange_kwargs": {
                        "freq": "day",
                        "limit_threshold": limit_threshold,
                        "deal_price": deal_price,
                        "open_cost": open_cost,
                        "close_cost": close_cost,
                        "min_cost": 5,
                    },
                },
            }
            
            # 生成回测记录名称: 当前日期时间_模型_市场_起止日期
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_recorder_name = f"{current_time}_{model_type}_{market}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}"
            
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

                sar = SigAnaRecord(recorder)
                sar.generate()

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
                params = rec.list_params()
                recorder_list.append({
                    "id": rec_id,
                    "name": rec_name,
                    "start_time": rec_info.get('start_time', '未知'),
                    "train_end": params.get('fit_end_time', ''),
                    "params": params
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

    def _get_trading_days_before(self, target_date, n_days, market="csi300"):
        target_date = pd.Timestamp(target_date)
        start_date = target_date - pd.Timedelta(days=n_days * 3 + 10)
        end_date = target_date + pd.Timedelta(days=5)
        try:
            trading_days = D.calendar(freq="day", start_time=start_date.strftime('%Y-%m-%d'), end_time=end_date.strftime('%Y-%m-%d'))
            trading_days = pd.to_datetime(trading_days)
            trading_days_before = trading_days[trading_days <= target_date]
            if len(trading_days_before) > n_days:
                # 使用 [-n_days-1] 索引，不使用 iloc
                return trading_days_before[-(n_days + 1)]
            elif len(trading_days_before) > 0:
                return trading_days_before[0]
            else:
                return target_date - pd.Timedelta(days=n_days)
        except Exception as e:
            print(f"[DEBUG] 获取交易日历失败: {e}")
            return target_date - pd.Timedelta(days=n_days)

    def _calculate_stock_statistics(self, all_positions_data, positions):
        """统计每只股票的盈亏情况
        
        Returns:
            list: 每只股票的统计信息，包括：
                - stock_code: 股票代码
                - stock_name: 股票名称
                - first_buy_date: 最早买入时间
                - last_sell_date: 最后卖出时间
                - cumulative_profit: 累计盈亏金额（所有交易）
                - hold_profit: 持有盈亏金额（假设从最早买入到最后卖出一直持有）
        """
        from collections import defaultdict
        
        # 收集每只股票的所有持仓记录
        stock_records = defaultdict(list)
        
        for date_str, date_positions in sorted(all_positions_data.items()):
            for pos in date_positions:
                stock_records[pos['stock_code']].append({
                    'date': date_str,
                    'stock_name': pos['stock_name'],
                    'amount': pos['amount'],
                    'cost_price': pos['cost_price'],
                    'current_price': pos['current_price'],
                    'hold_value': pos['hold_value'],
                    'hold_days': pos.get('hold_days', 1)
                })
        
        # 获取所有股票名称
        all_stock_codes = list(stock_records.keys())
        stock_names, _, _ = self.get_stock_names(all_stock_codes)
        
        # 统计每只股票
        statistics = []
        for stock_code, records in stock_records.items():
            if not records:
                continue
            
            # 按日期排序
            records.sort(key=lambda x: x['date'])
            
            first_buy_date = records[0]['date']
            last_sell_date = records[-1]['date']
            stock_name = stock_names.get(stock_code, records[0]['stock_name'])
            
            # 计算持股天数：每次持仓天数的汇总
            total_hold_days = sum(int(r.get('hold_days', 1)) for r in records)
            
            # 计算累计盈亏：每次持仓的盈亏累加
            cumulative_profit = 0.0
            for record in records:
                # 每次持仓的盈亏 = (当前价 - 成本价) * 数量
                profit = (record['current_price'] - record['cost_price']) * record['amount']
                cumulative_profit += profit
            
            # 计算持有盈亏：假设从最早买入到最后卖出一直持有
            # 最早买入时的成本价
            first_cost_price = records[0]['cost_price']
            # 最后卖出时的价格
            last_current_price = records[-1]['current_price']
            # 使用平均持仓数量
            avg_amount = sum(r['amount'] for r in records) / len(records)
            # 持有盈亏 = (最后价格 - 最初成本) * 平均数量
            hold_profit = (last_current_price - first_cost_price) * avg_amount
            
            statistics.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'first_buy_date': first_buy_date,
                'last_sell_date': last_sell_date,
                'hold_days': total_hold_days,
                'trade_count': len(records),
                'cumulative_profit': round(cumulative_profit, 2),
                'hold_profit': round(hold_profit, 2)
            })
        
        # 按累计盈亏排序（从高到低）
        statistics.sort(key=lambda x: x['cumulative_profit'], reverse=True)
        
        return statistics

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
                                open_date = self._get_trading_days_before(date, count_day)
                                start_date = open_date - pd.Timedelta(days=5)
                                end_date = open_date + pd.Timedelta(days=5)

                                price_data = D.features(
                                    instruments=[stock],
                                    fields=['$close'],
                                    start_time=start_date.strftime('%Y-%m-%d'),
                                    end_time=end_date.strftime('%Y-%m-%d')
                                )

                                if not price_data.empty:
                                    date_diffs = abs(price_data.index.get_level_values('datetime') - open_date)
                                    min_diff_idx = date_diffs.argmin()
                                    avg_price = float(price_data.iloc[min_diff_idx]['$close'])
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
                            open_date = self._get_trading_days_before(last_date, count_day)
                            start_date = open_date - pd.Timedelta(days=5)
                            end_date = open_date + pd.Timedelta(days=5)

                            price_data = D.features(
                                instruments=[stock],
                                fields=['$close'],
                                start_time=start_date.strftime('%Y-%m-%d'),
                                end_time=end_date.strftime('%Y-%m-%d')
                            )

                            if not price_data.empty:
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
            
            # 统计每只股票的盈亏情况
            stock_statistics = self._calculate_stock_statistics(all_positions_data, positions)
            
            # 获取回测配置参数
            backtest_config = recorder.list_params()

            return {
                "success": True,
                "key_metrics": key_metrics,
                "cumulative_data": cumulative_data,
                "daily_data": daily_data,
                "positions": positions_data,
                "all_positions": all_positions_data,
                "stock_statistics": stock_statistics,
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

            # 提取因子重要性（仅树模型支持）
            feature_importance = None
            try:
                model_type = params.get('model_type', '')
                if model_type in ('LGBModel', 'XGBModel', 'CatBoostModel', 'DEnsembleModel'):
                    try:
                        model_obj = recorder.load_object("trained_model")
                    except:
                        import pickle
                        mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                        backup_path = os.path.join(mlruns_dir, "model_backups", f"{recorder_id}_trained_model.pkl")
                        if os.path.exists(backup_path):
                            with open(backup_path, 'rb') as f:
                                model_obj = pickle.load(f)
                        else:
                            model_obj = None

                    # 加载保存的特征列名（优先）
                    saved_feature_names = None
                    try:
                        saved_feature_names = recorder.load_object("feature_names")
                    except:
                        pass

                    # 回退：根据 handler 重建特征列名
                    if saved_feature_names is None:
                        try:
                            h_type = params.get('handler_type', 'Alpha158')
                            h_module = self.HANDLER_REGISTRY.get(h_type, "qlib.contrib.data.handler")
                            h_config = {
                                "start_time": params.get('train_start_date', '2020-01-01'),
                                "end_time": params.get('test_end_date', '2024-01-01'),
                                "fit_start_time": params.get('train_start_date', '2020-01-01'),
                                "fit_end_time": params.get('train_end_date', '2022-01-01'),
                                "instruments": params.get('market', 'csi300'),
                            }
                            handler = init_instance_by_config({
                                "class": h_type,
                                "module_path": h_module,
                                "kwargs": h_config,
                            })
                            train_data = handler.fetch(col_set="feature")
                            saved_feature_names = list(train_data.columns)
                        except Exception as e:
                            logger.warning(f"回退获取特征列名失败: {e}")

                    if model_obj is not None:
                        # 获取内部模型对象
                        inner_model = getattr(model_obj, 'model', None)
                        if inner_model is not None:
                            fi_values = None
                            fi_names = None

                            if hasattr(inner_model, 'feature_importance'):
                                # LightGBM
                                fi_values = inner_model.feature_importance(importance_type='gain')
                                fi_names = inner_model.feature_name()
                            elif hasattr(inner_model, 'get_score'):
                                # XGBoost
                                score = inner_model.get_score(importance_type='gain')
                                fi_names = list(score.keys())
                                fi_values = list(score.values())
                            elif hasattr(inner_model, 'get_feature_importance'):
                                # CatBoost
                                fi_values = inner_model.get_feature_importance()
                                fi_names = [f"f{i}" for i in range(len(fi_values))]

                            # 将 Column_X 映射回真实特征名
                            if fi_names is not None and saved_feature_names:
                                mapped_names = []
                                for name in fi_names:
                                    if name.startswith("Column_"):
                                        try:
                                            idx = int(name.split("_")[1])
                                            if idx < len(saved_feature_names):
                                                mapped_names.append(str(saved_feature_names[idx]))
                                                continue
                                        except (ValueError, IndexError):
                                            pass
                                    mapped_names.append(name)
                                fi_names = mapped_names

                            if fi_values is not None and fi_names is not None:
                                # 排序并返回 top features
                                fi_pairs = sorted(zip(fi_names, fi_values), key=lambda x: x[1], reverse=True)
                                feature_importance = [
                                    {"name": name, "importance": float(val)}
                                    for name, val in fi_pairs if val > 0
                                ]
            except Exception as e:
                logger.warning(f"提取因子重要性失败: {e}")

            return {
                "success": True,
                "recorder_id": recorder_id,
                "params": params,
                "metrics": viz_data.get('metrics', {}) if viz_data else {},
                "group_returns": viz_data.get('group_returns', {}) if viz_data else {},
                "test_returns": viz_data.get('test_returns', {}) if viz_data else {},
                "ic_data": viz_data.get('ic_data', {}) if viz_data else {},
                "feature_importance": feature_importance,
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

    def delete_all_backtest_recorders(self):
        """删除所有回测记录"""
        try:
            exp = R.get_exp(experiment_name="backtest_analysis")
            recorders = exp.list_recorders(rtype="list")
            deleted_count = 0
            for recorder in recorders:
                try:
                    exp.delete_recorder(recorder.id)
                    self._clean_mlruns_files(recorder.id)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除记录 {recorder.id} 失败: {e}")
            return {"success": True, "message": f"已删除 {deleted_count} 条回测记录"}
        except Exception as e:
            return {"success": False, "message": f"删除所有回测记录失败: {str(e)}"}

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

    def get_stock_quote(self, stock_code, start_date, end_date):
        """获取股票行情数据（K线）"""
        try:
            from qlib.data import D
            import pandas as pd

            # 确保Qlib已初始化
            self.init_qlib()

            # 获取股票数据
            df = D.features(
                instruments=[stock_code],
                fields=['$open', '$close', '$high', '$low', '$volume', '$amount'],
                start_time=start_date,
                end_time=end_date
            )

            if df.empty:
                return {"success": False, "message": f"未找到股票 {stock_code} 的数据"}

            # 转换为前端需要的格式
            result = []
            
            # Qlib 返回的 DataFrame 索引是 MultiIndex: (datetime, instrument)
            # 需要重置索引以便正确获取日期
            df_reset = df.reset_index()
            
            # 获取列名
            date_col = 'datetime' if 'datetime' in df_reset.columns else 'date'
            
            for _, row in df_reset.iterrows():
                # 获取日期
                date_val = row.get(date_col)
                
                # 确保 date 是 datetime.date 或 datetime.datetime 对象
                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime('%Y-%m-%d')
                elif isinstance(date_val, str):
                    date_str = date_val
                else:
                    date_str = str(date_val)
                    
                result.append({
                    "date": date_str,
                    "open": float(row['$open']),
                    "close": float(row['$close']),
                    "high": float(row['$high']),
                    "low": float(row['$low']),
                    "volume": int(row['$volume']),
                    "amount": float(row['$amount']),
                    "pre_close": float(row['$close'])  # 前一个收盘价，用当前代替
                })

            # 计算涨跌幅
            for i in range(1, len(result)):
                result[i]['pre_close'] = result[i-1]['close']

            return {
                "success": True,
                "data": result,
                "code": stock_code
            }
        except Exception as e:
            return {"success": False, "message": f"获取股票行情失败: {str(e)}"}

    def predict_today(self, recorder_id, market, topk=10, n_drop=1):
        """
        生成每日交易信号：加载模型，用最新数据生成预测，输出买卖建议
        
        Parameters
        ----------
        recorder_id : str
            训练记录ID
        market : str
            市场（csi300, csi500 等）
        topk : int
            目标持仓数量
        n_drop : int
            每日最多换仓数量
            
        Returns
        -------
        dict
            包含 buy_list, sell_list, hold_list, full_scores 等
        """
        try:
            # 1. 加载模型和训练参数
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")
            train_params = recorder.list_params()
            handler_type = train_params.get('handler_type', 'Alpha158')
            label_horizon = int(train_params.get('label_horizon', 1))
            handler_module_path = self.HANDLER_REGISTRY.get(handler_type, "qlib.contrib.data.handler")
            fit_start_time = train_params.get('fit_start_time', train_params.get('train_start_date', '2020-01-01'))
            fit_end_time = train_params.get('fit_end_time', train_params.get('train_end_date', '2024-01-01'))

            # 加载模型
            try:
                model = recorder.load_object("trained_model")
            except:
                import pickle
                mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                backup_path = os.path.join(mlruns_dir, "model_backups", f"{recorder_id}_trained_model.pkl")
                with open(backup_path, 'rb') as f:
                    model = pickle.load(f)

            # 2. 构建今日数据集（用最近 2 个交易日的数据以确保有当天特征）
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            # 往前推 5 天确保覆盖到最近交易日
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

            # 与 train / backtest 保持一致：树模型不使用额外 processors
            model_type = train_params.get('model_type', 'LGBModel')
            is_tree_model = model_type in ("LGBModel", "XGBModel", "CatBoostModel", "DEnsembleModel")

            data_handler_config = {
                "start_time": start_date,
                "end_time": today,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
                "instruments": market,
            }

            if not is_tree_model:
                data_handler_config["infer_processors"] = [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ]
                data_handler_config["learn_processors"] = [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ]

            # 预测使用与训练相同的标签周期
            label_config = self._build_label_config(label_horizon)
            if label_config is not None:
                data_handler_config["label"] = label_config

            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": handler_type,
                        "module_path": handler_module_path,
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "test": (start_date, today),
                    },
                },
            }

            dataset = init_instance_by_config(dataset_config)

            # 3. 模型预测
            pred = model.predict(dataset)
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")

            # 取最新交易日的预测
            latest_date = pred.index.get_level_values('datetime').max()
            today_pred = pred.xs(latest_date, level='datetime').sort_values('score', ascending=False)

            # 4. 生成信号
            # 获取股票名称
            all_stocks = today_pred.index.tolist()
            stock_names, _, _ = self.get_stock_names(all_stocks)

            # Top-K 买入列表
            buy_candidates = today_pred.head(topk)
            full_scores = []
            for stock, row in today_pred.iterrows():
                full_scores.append({
                    "stock_code": stock,
                    "stock_name": stock_names.get(stock, stock),
                    "score": float(row['score']),
                    "rank": len(full_scores) + 1,
                })

            buy_list = []
            for stock, row in buy_candidates.iterrows():
                buy_list.append({
                    "stock_code": stock,
                    "stock_name": stock_names.get(stock, stock),
                    "score": float(row['score']),
                })

            return {
                "success": True,
                "prediction_date": latest_date.strftime('%Y-%m-%d'),
                "market": market,
                "model_type": train_params.get('model_type', 'Unknown'),
                "handler_type": handler_type,
                "topk": topk,
                "buy_list": buy_list,
                "full_scores": full_scores[:100],  # 返回前100只的评分
                "total_stocks": len(today_pred),
                "message": f"生成 {latest_date.strftime('%Y-%m-%d')} 交易信号成功，共评估 {len(today_pred)} 只股票"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"生成预测信号失败: {str(e)}"}
