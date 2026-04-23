import os
import sys
import datetime
import pickle
import numpy as np
import pandas as pd
from typing import Callable, Optional

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.data.dataset import TSDatasetH
from qlib.contrib.data.handler import Alpha158, Alpha360


class DLModelService:
    """
    深度学习模型服务
    
    使用 TSDatasetH + pytorch_lstm_ts 架构
    支持时间序列深度学习模型（LSTM, GRU, ALSTM）
    """
    
    _qlib_initialized = False
    
    def __init__(self):
        self.provider_uri = "~/.qlib/qlib_data/cn_data"
        if not DLModelService._qlib_initialized:
            self.init_qlib()
    
    def init_qlib(self, force=False):
        """初始化 Qlib"""
        try:
            if force or not DLModelService._qlib_initialized:
                qlib.init(provider_uri=self.provider_uri, region=REG_CN)
                DLModelService._qlib_initialized = True
            
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mlruns_dir = os.path.join(backend_dir, "mlruns")
            os.makedirs(mlruns_dir, exist_ok=True)
            sqlite_uri = f"sqlite:///{os.path.join(mlruns_dir, 'mlflow.db')}"
            R.set_uri(sqlite_uri)
        except Exception as e:
            print(f"Qlib初始化失败: {e}")
            try:
                provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
                qlib.init(provider_uri=provider_uri, region=REG_CN)
                DLModelService._qlib_initialized = True
                
                backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mlruns_dir = os.path.join(backend_dir, "mlruns")
                os.makedirs(mlruns_dir, exist_ok=True)
                sqlite_uri = f"sqlite:///{os.path.join(mlruns_dir, 'mlflow.db')}"
                R.set_uri(sqlite_uri)
            except Exception as e2:
                print(f"使用绝对路径初始化Qlib也失败: {e2}")
    
    MODEL_REGISTRY = {
        "LSTM": "qlib.contrib.model.pytorch_lstm_ts",
        "GRU": "qlib.contrib.model.pytorch_gru_ts",
        "ALSTM": "qlib.contrib.model.pytorch_alstm_ts",
    }
    
    HANDLER_REGISTRY = {
        "Alpha158": "qlib.contrib.data.handler",
        "Alpha360": "qlib.contrib.data.handler",
    }
    
    def _get_model_config(self, model_type, d_feat, hidden_size, num_layers, 
                         dropout, lr, n_epochs, batch_size, early_stop, seed, GPU, n_jobs=0):
        """获取深度学习模型配置
        
        GPU利用率优化参数:
        - n_jobs=0: DataLoader单进程，避免CPU瓶颈导致GPU等待
          （官方配置n_jobs=20适合多核服务器，单机建议降低）
        - GPU=0: 使用第一个GPU
        
        如果GPU利用率仍然低，可以尝试：
        - 增大batch_size（如2000）
        - 设置n_jobs=2-4（如果CPU核心足够）
        """
        module_path = self.MODEL_REGISTRY.get(model_type, "qlib.contrib.model.pytorch_lstm_ts")
        
        print(f"[GPU配置] GPU={GPU}, n_jobs={n_jobs}, batch_size={batch_size}")
        print(f"[GPU提示] 如果GPU利用率低，尝试增大batch_size或调整n_jobs")
        
        return {
            "class": model_type,
            "module_path": module_path,
            "kwargs": {
                "d_feat": d_feat,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "early_stop": early_stop,
                "metric": "loss",
                "loss": "mse",
                "seed": seed,
                "GPU": GPU,
                "n_jobs": n_jobs,
            },
        }
    
    def _get_handler_config(self, handler_type, market, train_start_date, train_end_date,
                           test_end_date, step_len, use_all_features=False):
        """获取 TSDatasetH 的 handler 配置
        
        Args:
            use_all_features: 是否使用全部特征（158/360），默认False使用20个特征
        """
        
        # 选择特征列表
        if use_all_features:
            # 使用全部特征，不进行 FilterCol
            col_list = None
        elif handler_type == "Alpha158":
            # Alpha158 的20个特征
            col_list = [
                "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
                "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
                "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
            ]
        elif handler_type == "Alpha360":
            # Alpha360 的前20个特征
            col_list = [
                "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
                "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
                "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
            ]
        else:
            col_list = None
        
        data_handler_config = {
            "start_time": train_start_date,
            "end_time": test_end_date,
            "fit_start_time": train_start_date,
            "fit_end_time": train_end_date,
            "instruments": market,
            "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
        }
        
        # 构建处理器列表
        infer_processors = []
        
        # 添加特征选择（如果指定了col_list）
        if col_list:
            infer_processors.append({
                "class": "FilterCol",
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "feature",
                    "col_list": col_list,
                }
            })
        
        # 添加数据清理处理器（无论是否使用全部特征都需要）
        infer_processors.extend([
            {
                "class": "DropnaProcessor",  # 删除特征中包含NaN的行
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "feature",
                }
            },
            {
                "class": "Fillna",  # 填充剩余的NaN
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "feature",
                }
            },
            {
                "class": "RobustZScoreNorm",
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                }
            },
            {
                "class": "Fillna",  # 再次填充（标准化后可能产生NaN）
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "feature",
                }
            }
        ])
        
        data_handler_config["infer_processors"] = infer_processors
        data_handler_config["learn_processors"] = [
            {
                "class": "DropnaLabel",
            },
            {
                "class": "CSRankNorm",
                "module_path": "qlib.data.dataset.processor",
                "kwargs": {
                    "fields_group": "label",
                }
            }
        ]
        
        return data_handler_config
    
    def train_dl_model_with_progress(self, progress_callback: Optional[Callable], 
                                     market, benchmark, train_start_date, train_end_date,
                                     valid_start_date, valid_end_date, test_start_date, test_end_date,
                                     model_type, handler_type, d_feat, hidden_size, num_layers,
                                     dropout, lr, n_epochs, batch_size, early_stop, step_len,
                                     seed, GPU, use_all_features=False):
        """训练深度学习模型（带进度回调）
        
        Args:
            use_all_features: 是否使用全部特征，默认False使用20个特征
        """
        rid = None
        try:
            progress_callback(0, "开始训练深度学习模型...")
            
            progress_callback(10, "配置数据处理参数...")
            
            # 根据是否使用全部特征设置 d_feat
            if use_all_features:
                if "Alpha158" in handler_type:
                    actual_d_feat = 158
                elif "Alpha360" in handler_type:
                    actual_d_feat = 360
                else:
                    actual_d_feat = d_feat
                print(f"使用全部特征: {handler_type}, d_feat={actual_d_feat}")
            else:
                actual_d_feat = d_feat
                print(f"使用选定特征: d_feat={actual_d_feat}")
            
            handler_config = self._get_handler_config(
                handler_type, market, train_start_date, train_end_date,
                test_end_date, step_len, use_all_features
            )
            
            progress_callback(20, "构建任务配置...")
            
            model_config = self._get_model_config(
                model_type, actual_d_feat, hidden_size, num_layers,
                dropout, lr, n_epochs, batch_size, early_stop, seed, GPU
            )
            
            handler_module_path = self.HANDLER_REGISTRY.get(handler_type, "qlib.contrib.data.handler")
            
            task = {
                "model": model_config,
                "dataset": {
                    "class": "TSDatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": handler_type,
                            "module_path": handler_module_path,
                            "kwargs": handler_config,
                        },
                        "segments": {
                            "train": (train_start_date, train_end_date),
                            "valid": (valid_start_date, valid_end_date),
                            "test": (test_start_date, test_end_date),
                        },
                        "step_len": step_len,
                    },
                },
            }
            
            progress_callback(30, "初始化模型...")
            
            model = init_instance_by_config(task["model"])
            
            # 检查模型是否在GPU上
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"[GPU检查] CUDA可用: {torch.cuda.get_device_name(0)}")
                    if hasattr(model, '_model') and hasattr(model._model, 'parameters'):
                        device = next(model._model.parameters()).device
                        print(f"[GPU检查] 模型设备: {device}")
                    elif hasattr(model, 'parameters'):
                        device = next(model.parameters()).device
                        print(f"[GPU检查] 模型设备: {device}")
                    else:
                        print(f"[GPU检查] 无法直接检查模型设备，GPU参数={GPU}")
                else:
                    print("[GPU检查] CUDA不可用，将使用CPU训练")
            except Exception as e:
                print(f"[GPU检查] 检查失败: {e}")
            
            progress_callback(40, "初始化数据集...")
            
            dataset = init_instance_by_config(task["dataset"])
            
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
                    model_type=model_type,
                    handler_type=handler_type,
                    d_feat=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    lr=lr,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    early_stop=early_stop,
                    step_len=step_len,
                    seed=seed,
                    GPU=GPU,
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
                    "ICIR": float(ic.mean() / ic.std()),
                    "Rank_IC": float(ric.mean()),
                    "Rank_ICIR": float(ric.mean() / ric.std()),
                    "Long_precision": float(long_pre.mean()),
                    "Short_precision": float(short_pre.mean()),
                    "Long_Short_Avg_Return": float(long_short_r.mean()),
                    "Long_Short_Avg_Sharpe": float(long_short_r.mean() / long_short_r.std()) if long_short_r.std() != 0 else 0,
                }
                
                R.log_metrics(**metrics)
                
                # 生成分组收益数据
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
                
                # 测试集收益
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
                
                # 备份模型
                mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")
                model_backup_dir = os.path.join(mlruns_dir, "model_backups")
                os.makedirs(model_backup_dir, exist_ok=True)
                backup_path = os.path.join(model_backup_dir, f"{rid}_trained_model.pkl")
                with open(backup_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"模型备份保存到: {backup_path}")
                
                print(f"模型保存完成")
            
            progress_callback(90, "训练完成，生成记录ID...")
            
            return {"success": True, "recorder_id": rid, "message": "深度学习模型训练完成！"}
            
        except Exception as e:
            progress_callback(-1, f"模型训练失败: {str(e)}")
            return {"success": False, "message": f"模型训练失败: {str(e)}"}