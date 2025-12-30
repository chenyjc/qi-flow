# A Streamlit App for Qlib Trainnng & Backtest
# Run with command below(activate venv first):
# streamlit run streamlit/app.py

"""
## Data Preparation
❗ Due to more restrict data security policy. The official dataset is disabled temporarily. 
You can try [this data source](https://github.com/chenditc/investment_data/releases) contributed by the community.
Here is an example to download the latest data.
```bash
curl -O https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

The official dataset below will resume in short future.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 设置中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 导入Qlib相关模块
import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

# 设置页面为宽屏模式
st.set_page_config(layout="wide")

# 初始化Qlib
provider_uri = "~/.qlib/qlib_data/cn_data"

# 应用标题
st.title("Qlib 量化交易策略回测系统")

# 数据下载函数
def download_data():
    with st.spinner("正在下载和初始化Qlib数据..."):
        try:
            import os
            import tarfile
            import urllib.request
            from tqdm import tqdm
            
            # 数据URL
            url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
            
            # 临时文件路径
            temp_file = "qlib_bin.tar.gz"
            
            # 目标目录
            target_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data")
            
            st.info(f"正在下载数据: {url}")
            
            # 使用tqdm显示下载进度条
            def download_progress(count, block_size, total_size):
                if not hasattr(download_progress, "pbar"):
                    download_progress.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中")
                downloaded = count * block_size
                if downloaded < total_size:
                    download_progress.pbar.update(block_size)
                else:
                    download_progress.pbar.close()
            
            # 下载文件
            urllib.request.urlretrieve(url, temp_file, reporthook=download_progress)
            
            st.info("数据下载完成，正在创建目录...")
            
            # 创建目标目录
            os.makedirs(target_dir, exist_ok=True)
            
            st.info(f"正在解压到: {target_dir}")
            
            # 解压文件，实现--strip-components=1效果
            with tarfile.open(temp_file, "r:gz") as tar:
                # 跳过顶层目录
                for member in tar.getmembers():
                    if '/' in member.name:
                        member.name = member.name.split('/', 1)[1]
                        tar.extract(member, target_dir)
            
            st.info("正在清理临时文件...")
            
            # 删除临时文件
            os.remove(temp_file)
            
            st.success("Qlib数据加载完成！")
            return True
            
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

# 侧边栏 - 可折叠的配置和功能区域
with st.sidebar:
    # 配置选项作为大标题
    st.header("配置选项")
    
    # 数据加载分组 - 放在最前面
    with st.expander("数据加载", expanded=True):
        st.subheader("数据操作")
        # 添加市场选择
        preview_market = st.selectbox("预览市场", ["all", "csi300", "csi500", "csi800", "csi1000", "csiall"], index=1)
        
        # 数据操作按钮行
        col1, col2 = st.columns(2)
        with col1:
            download_btn = st.button("下载数据")
        with col2:
            preview_data = st.button("预览数据")
        
        # 如果点击了下载数据按钮
        if download_btn:
            download_data()

# 初始化Qlib
try:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    st.success(f"Qlib初始化成功，数据路径: {provider_uri}")
except Exception as e:
    st.error(f"Qlib初始化失败: {e}")
    st.stop()

# 侧边栏继续
with st.sidebar:
    # 训练相关分组 - 可折叠（包含所有训练需要的参数，不与回测共享）
    with st.expander("训练相关", expanded=True):
        st.subheader("训练基础配置")
        # 训练专用的市场和基准
        train_market = st.selectbox("训练市场", ["all", "csi300", "csi500", "csi800", "csi1000", "csiall"])
        train_benchmark = st.selectbox("训练基准指数", ["SH000300", "SH000016", "SH000852", "SH000905"])
        
        st.subheader("训练时间范围")
        # 训练专用的时间范围
        train_start_date = st.date_input("训练开始日期", value=pd.to_datetime("2008-01-01"))
        train_end_date = st.date_input("训练结束日期", value=pd.to_datetime("2023-12-31"))
        valid_start_date = st.date_input("验证开始日期", value=pd.to_datetime("2024-01-01"))
        valid_end_date = st.date_input("验证结束日期", value=pd.to_datetime("2024-12-31"))
        
        # 训练专用的测试集配置
        test_start_date = st.date_input("测试集开始日期", value=pd.to_datetime("2025-01-01"))
        test_end_date = st.date_input("测试集结束日期", value=pd.to_datetime("2025-12-15"))
        
        st.subheader("模型参数")
        # 训练专用的模型参数
        model_type = st.selectbox("模型类型", ["LGBModel", "XGBModel", "Linear"])
        lr = st.slider("学习率", 0.001, 0.1, 0.0421)
        max_depth = st.slider("最大深度", 3, 15, 8)
        num_leaves = st.slider("叶子数量", 30, 300, 210)
        subsample = st.slider("采样比例", 0.5, 1.0, 0.8789)
        colsample_bytree = st.slider("列采样比例", 0.5, 1.0, 0.8879)
        
        # 训练模型按钮
        if st.button("训练模型"):
            # 记录训练开始时间
            train_start_time = pd.Timestamp.now()
            
            with st.spinner("正在训练模型..."):
                try:
                    # 数据处理配置
                    data_handler_config = {
                        "start_time": train_start_date.strftime("%Y-%m-%d"),
                        "end_time": test_end_date.strftime("%Y-%m-%d"),  # 使用测试集结束日期
                        "fit_start_time": train_start_date.strftime("%Y-%m-%d"),
                        "fit_end_time": valid_end_date.strftime("%Y-%m-%d"),
                        "instruments": train_market,  # 使用训练专用市场
                    }
                    
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
                                    "train": (train_start_date.strftime("%Y-%m-%d"), train_end_date.strftime("%Y-%m-%d")),
                                    "valid": (valid_start_date.strftime("%Y-%m-%d"), valid_end_date.strftime("%Y-%m-%d")),
                                    "test": (test_start_date.strftime("%Y-%m-%d"), test_end_date.strftime("%Y-%m-%d")),  # 使用测试集日期
                                },
                            },
                        },
                    }
                    
                    # 初始化模型和数据集
                    model = init_instance_by_config(task["model"])
                    dataset = init_instance_by_config(task["dataset"])
                    
                    # 训练模型
                    with R.start(experiment_name="train_model"):
                        # 保存训练参数
                        R.log_params(
                            # 训练基础配置
                            market=train_market,
                            benchmark=train_benchmark,
                            
                            # 时间范围
                            train_start_date=train_start_date.strftime("%Y-%m-%d"),
                            train_end_date=train_end_date.strftime("%Y-%m-%d"),
                            valid_start_date=valid_start_date.strftime("%Y-%m-%d"),
                            valid_end_date=valid_end_date.strftime("%Y-%m-%d"),
                            test_start_date=test_start_date.strftime("%Y-%m-%d"),
                            test_end_date=test_end_date.strftime("%Y-%m-%d"),
                            
                            # 模型参数
                            model_type=model_type,
                            lr=lr,
                            max_depth=max_depth,
                            num_leaves=num_leaves,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            
                            # 任务配置
                            **flatten_dict(task)
                        )
                        model.fit(dataset)
                        R.save_objects(trained_model=model)
                        rid = R.get_recorder().id
                    
                    # 记录训练结束时间
                    train_end_time = pd.Timestamp.now()
                    train_duration = train_end_time - train_start_time
                    
                    st.success(f"模型训练完成！Recorder ID: {rid}")
                    st.success(f"训练耗时: {train_duration.total_seconds():.2f} 秒")
                    
                except Exception as e:
                    # 记录训练结束时间
                    train_end_time = pd.Timestamp.now()
                    train_duration = train_end_time - train_start_time
                    
                    st.error(f"模型训练失败: {e}")
                    st.error(f"训练耗时: {train_duration.total_seconds():.2f} 秒")
                    import traceback
                    st.error(traceback.format_exc())
        

    
    # 回测相关分组 - 可折叠（包含所有回测需要的参数，不与训练共享）
    with st.expander("回测相关", expanded=True):
        st.subheader("回测基础配置")
        # 回测专用的市场和基准
        backtest_market = st.selectbox("回测市场", ["all", "csi300", "csi500", "csi800", "csi1000", "csiall"], index=1)  # 默认改为csi300
        backtest_benchmark = st.selectbox("回测基准指数", ["SH000300", "SH000016", "SH000852", "SH000905"])
        
        st.subheader("回测日期配置")
        backtest_start_date = st.date_input("回测开始日期", value=pd.to_datetime("2025-01-01"))
        backtest_end_date = st.date_input("回测结束日期", value=pd.to_datetime("2025-12-15"))
        
        st.subheader("回测参数配置")
        initial_account = st.number_input("初始资金", min_value=100000, max_value=1000000000, value=500000, step=1000000)  # 最小值改为10万，默认值改为50万
        topk = st.slider("Topk", 1, 100, 5)  # 最小值改为1，默认值改为5
        n_drop = st.slider("每次调仓卖出数量", 1, 20, 2)  # 调整为2
        # 策略类型（移到回测相关组）
        strategy_type = st.selectbox("策略类型", ["TopkDropoutStrategy", "WeightStrategyBase"])
        
        # 训练记录选择（移到回测相关组）
        st.subheader("训练记录")
        
        # 定义实验名称（全局可用）
        experiment_name = "train_model"
        selected_rid = None
        
        # 尝试获取所有训练记录
        recorders = []
        try:
            exp = R.get_exp(experiment_name=experiment_name)
            recorders = exp.list_recorders(rtype="list")
            
        except Exception as e:
            st.error(f"获取实验记录失败: {e}")
        
        # 显示训练记录选择
        if isinstance(recorders, list) and len(recorders) > 0:
            # 为每条记录创建可读的选项
            recorder_options = {}
            for rec in recorders:
                # 获取记录的创建时间和ID
                rec_info = rec.info
                rec_id = rec.id
                # 创建可读的显示名称，只显示时间
                option_name = f"{rec_info.get('start_time', '未知')}"
                recorder_options[option_name] = rec_id
            
            # 让用户选择要使用的记录
            selected_option = st.selectbox(
                "选择要使用的训练记录",
                options=list(recorder_options.keys()),
                index=len(recorder_options)-1  # 默认选择最新的记录
            )
            
            # 获取选择的记录ID
            selected_rid = recorder_options[selected_option]
            st.write(f"已选择记录: {selected_rid}")
        else:
            st.warning("没有找到训练记录，请先训练模型！")
            selected_rid = None
        
        # 回测和分析按钮
        if selected_rid and st.button("执行回测和分析"):
            # 记录回测开始时间
            backtest_start_time = pd.Timestamp.now()
            
            with st.spinner("正在执行回测和分析..."):
                try:
                    # 获取指定ID的记录器，需要同时提供实验名称
                    recorder = R.get_recorder(recorder_id=selected_rid, experiment_name=experiment_name)
                    
                    # 加载模型
                    model = recorder.load_object("trained_model")
                    
                    # 加载数据集（需要重新构建）
                    data_handler_config = {
                        "start_time": train_start_date.strftime("%Y-%m-%d"),
                        "end_time": backtest_end_date.strftime("%Y-%m-%d"),
                        "fit_start_time": train_start_date.strftime("%Y-%m-%d"),
                        "fit_end_time": valid_end_date.strftime("%Y-%m-%d"),
                        "instruments": backtest_market,  # 使用回测专用市场
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
                                "train": (train_start_date.strftime("%Y-%m-%d"), train_end_date.strftime("%Y-%m-%d")),
                                "valid": (valid_start_date.strftime("%Y-%m-%d"), valid_end_date.strftime("%Y-%m-%d")),
                                "test": (backtest_start_date.strftime("%Y-%m-%d"), backtest_end_date.strftime("%Y-%m-%d")),
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
                                "time_per_step": "day",  # 固定为day周期，数据不支持其他周期
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
                            "start_time": backtest_start_date.strftime("%Y-%m-%d"),
                            "end_time": backtest_end_date.strftime("%Y-%m-%d"),
                            "account": initial_account,
                            "benchmark": backtest_benchmark,  # 使用回测专用基准
                            "exchange_kwargs": {
                                "freq": "day",  # 固定为day周期，数据不支持其他周期
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
                        
                        # 确保selected_rid被正确获取
                        if selected_rid:
                            st.write(f"使用训练记录ID: {selected_rid} 执行回测")
                        else:
                            st.warning("未找到训练记录ID，回测可能会失败")
                        
                        # 保存训练记录ID和参数到回测记录中
                        # 使用当前recorder对象的log_params方法，确保参数被正确保存
                        recorder.log_params(
                            # 训练记录ID
                            train_recorder_id=str(selected_rid) if selected_rid else "",
                            
                            # 回测参数
                            backtest_market=backtest_market,
                            backtest_benchmark=backtest_benchmark,
                            backtest_start_date=backtest_start_date.strftime("%Y-%m-%d"),
                            backtest_end_date=backtest_end_date.strftime("%Y-%m-%d"),
                            initial_account=initial_account,
                            topk=topk,
                            n_drop=n_drop,
                            strategy_type=strategy_type
                        )
                        
                        sr = SignalRecord(model, dataset, recorder)
                        sr.generate()
                        
                        par = PortAnaRecord(recorder, port_analysis_config, "day")
                        par.generate()
                    
                    # 记录回测结束时间
                    backtest_end_time = pd.Timestamp.now()
                    backtest_duration = backtest_end_time - backtest_start_time
                    
                    st.success(f"回测和分析完成！Recorder ID: {ba_rid}")
                    st.success(f"回测耗时: {backtest_duration.total_seconds():.2f} 秒")
                    
                except Exception as e:
                    # 记录回测结束时间
                    backtest_end_time = pd.Timestamp.now()
                    backtest_duration = backtest_end_time - backtest_start_time
                    
                    st.error(f"回测和分析失败: {e}")
                    st.error(f"回测耗时: {backtest_duration.total_seconds():.2f} 秒")
                    import traceback
                    st.error(traceback.format_exc())
        
        # 回测记录选择
        st.subheader("回测记录")
        
        # 尝试获取所有回测记录
        backtest_recorders = []
        try:
            backtest_exp = R.get_exp(experiment_name="backtest_analysis")
            backtest_recorders = backtest_exp.list_recorders(rtype="list")
            
        except Exception as e:
            st.error(f"获取回测记录失败: {e}")
        
        # 显示回测记录选择
        if isinstance(backtest_recorders, list) and len(backtest_recorders) > 0:
            # 为每条记录创建可读的选项，只显示时间
            backtest_recorder_options = {}
            for rec in backtest_recorders:
                rec_info = rec.info
                rec_id = rec.id
                option_name = f"{rec_info.get('start_time', '未知')}"
                backtest_recorder_options[option_name] = rec_id
            
            # 让用户选择要查看的回测记录
            selected_backtest_option = st.selectbox(
                "选择回测记录",
                options=list(backtest_recorder_options.keys())
            )
            
            # 获取选择的回测记录ID，并保存到session_state中
            selected_ba_rid = backtest_recorder_options[selected_backtest_option]
            st.session_state.selected_ba_rid = selected_ba_rid
            st.write(f"已选择回测记录: {selected_ba_rid}")
        else:
            st.write("暂无回测记录，请先执行回测")
    
    # 运行说明
    st.markdown("""
    ## 运行说明
    
    1. **数据准备**: 确保已下载Qlib数据
    2. **配置参数**: 在左侧选择合适的参数
    3. **训练模型**: 点击"训练模型"按钮
    4. **执行回测**: 选择训练记录后点击"执行回测和分析"按钮
    5. **查看结果**: 在右侧查看回测结果和图表
    
    ## 注意事项
    
    - 首次运行需要下载数据
    - 训练和回测可能需要较长时间
    - 建议先使用小数据量进行测试
    """)

# 主内容区
# 数据预览功能
st.header("数据预览与回测结果")

# 实现数据预览功能
if 'preview_data' in locals() and preview_data:
    with st.spinner("正在获取数据..."):
        try:
            # 获取选择市场的标的列表
            market_instruments = []
            
            # 简化实现：直接获取所有标的，避免复杂的市场过滤
            # 修复D.list_instruments()函数调用错误
            try:
                # 尝试获取所有标的
                # 修复D.list_instruments()函数调用错误, instruments : dict
                instruments = D.instruments(market=preview_market)
                market_instruments_dict = D.list_instruments(instruments)
                market_instruments = list(market_instruments_dict.keys())
                print("market_instruments:", market_instruments)
            except Exception as e:
                # 如果获取失败，使用固定的标的列表
                st.warning(f"无法获取标的列表，使用默认标的: {e}")
                # 使用固定的5个标的作为示例
                market_instruments = ['SH600000', 'SH600004', 'SH600006', 'SH600007', 'SH600008']
            
            if market_instruments:
                # 取前5个股票作为示例
                sample_instruments = market_instruments[:5]
                
                # 获取日历数据
                calendar = D.calendar()
                if len(calendar) == 0:
                    st.warning("无法获取日历数据")
                else:
                    # 获取最新的日期和20天前的日期作为时间范围
                    latest_date = calendar[-1]
                    start_date = calendar[max(0, len(calendar) - 20)]
                    
                    # 定义要获取的特征
                    fields = ['$open', '$close', '$high', '$low', '$volume', '$factor', '$amount']
                    
                    # 获取数据
                    data = D.features(
                        instruments=sample_instruments,
                        fields=fields,
                        start_time=start_date,
                        end_time=latest_date
                    )
                    
                    # 显示数据预览表格
                    st.subheader("最新数据预览")
                    st.write(f"数据起始时间: {data.index.get_level_values('datetime').min()}")
                    st.write(f"数据结束时间: {data.index.get_level_values('datetime').max()}")
                    st.write(f"股票数量: {len(data.index.get_level_values('instrument').unique())}")
                    st.write(f"市场: {preview_market}")
                    st.dataframe(data.tail(10), use_container_width=True)
            else:
                st.warning("未找到股票数据")
                
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            import traceback
            st.error(traceback.format_exc())

# 回测结果部分
st.header("回测结果")

# 定义回测结果展示函数
def show_backtest_results(selected_ba_rid=None):
    # 尝试获取回测分析记录
    try:
        # 获取所有回测分析记录
        backtest_exp = R.get_exp(experiment_name="backtest_analysis")
        backtest_recorders = backtest_exp.list_recorders(rtype="list")
        
        if backtest_recorders:
            # 加载最新的回测记录或用户选择的记录
            if selected_ba_rid:
                # 使用用户在侧边栏选择的回测记录
                recorder = R.get_recorder(recorder_id=selected_ba_rid, experiment_name="backtest_analysis")
            else:
                # 使用最新的回测记录
                latest_recorder = backtest_recorders[-1]
                recorder = latest_recorder
            
            # 加载结果数据，添加异常处理
            report_normal_df = None
            positions = None
            analysis_df = None
            
            try:
                report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
                positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
                analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
            except Exception as e:
                st.error(f"加载回测结果文件失败: {e}")
                st.warning("某些回测结果文件可能不存在或路径不正确，尝试加载其他可用数据...")
                
            # 从回测记录中获取保存的参数
            try:
                # 获取回测记录中保存的参数，使用list_params()方法
                params = recorder.list_params()
            except Exception as e:
                st.error(f"获取回测参数失败: {e}")
                params = {}
            
            # 创建主内容区的左右两栏，左侧显示关键指标和图表，右侧显示用户要求的信息
            main_col, info_col = st.columns([2, 1])
            
            # 左侧主内容区
            with main_col:
                # 检查是否成功加载回测结果数据
                if report_normal_df is not None and analysis_df is not None:
                    # 显示关键指标
                    st.subheader("关键指标")
                    
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
                    
                    # 创建关键指标表格数据
                    key_metrics_data = [
                        {"指标名称": "策略总收益", "数值": f"{total_return:.2f}%"},
                        {"指标名称": "基准总收益", "数值": f"{bench_return:.2f}%"},
                        {"指标名称": "超额收益", "数值": f"{excess_return_total:.2f}%"},
                        {"指标名称": "年化收益率", "数值": f"{annualized_return:.2%}"},
                        {"指标名称": "信息比率", "数值": f"{information_ratio:.3f}"},
                        {"指标名称": "最大回撤", "数值": f"{max_drawdown:.2%}"}
                    ]
                    
                    # 显示关键指标表格
                    key_metrics_df = pd.DataFrame(key_metrics_data)
                    st.dataframe(key_metrics_df, hide_index=True)
                    
                    # 绘制累计收益曲线
                    st.subheader("累计收益曲线")
                    
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # 1. 累计收益曲线对比
                    axes[0, 0].plot(report_normal_df.index, report_normal_df['cumulative_return'], 
                                   label='策略累计收益', color='red', linewidth=2)
                    axes[0, 0].plot(report_normal_df.index, report_normal_df['cumulative_bench'], 
                                   label='基准累计收益', color='blue', linewidth=2)
                    axes[0, 0].set_title('累计收益曲线对比', fontsize=14, fontweight='bold')
                    axes[0, 0].set_ylabel('累计收益率')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 2. 日收益率对比
                    axes[0, 1].plot(report_normal_df.index, report_normal_df['return'], 
                                   label='策略日收益', color='red', alpha=0.7)
                    axes[0, 1].plot(report_normal_df.index, report_normal_df['bench'], 
                                   label='基准日收益', color='blue', alpha=0.7)
                    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[0, 1].set_title('日收益率对比', fontsize=14, fontweight='bold')
                    axes[0, 1].set_ylabel('日收益率')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # 3. 超额收益
                    excess_return = report_normal_df['return'] - report_normal_df['bench']
                    axes[1, 0].plot(report_normal_df.index, excess_return, 
                                   label='超额收益', color='green', linewidth=1.5)
                    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[1, 0].set_title('超额收益（策略-基准）', fontsize=14, fontweight='bold')
                    axes[1, 0].set_ylabel('超额收益率')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 4. 累计超额收益
                    cumulative_excess = report_normal_df['cumulative_return'] - report_normal_df['cumulative_bench']
                    axes[1, 1].plot(report_normal_df.index, cumulative_excess, 
                                   label='累计超额收益', color='green', linewidth=2)
                    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[1, 1].set_title('累计超额收益', fontsize=14, fontweight='bold')
                    axes[1, 1].set_ylabel('累计超额收益率')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("无法加载回测结果数据，可能是回测记录不完整或路径不正确")
                
                # 显示最后一次持仓（放到最后）
                st.subheader("最后一次持仓")
                if positions:
                    # 获取最后一个交易日
                    last_date = list(positions.keys())[-1]
                    last_position = positions[last_date]
                    
                    # 提取股票持仓数据
                    stocks_data = []
                    
                    try:
                        # 正确获取持仓字典：Position对象的position属性
                        if hasattr(last_position, 'position'):
                            pos_dict = last_position.position
                        elif isinstance(last_position, dict):
                            # 兼容旧版本数据格式
                            pos_dict = last_position
                        else:
                            st.error(f"无法解析持仓数据，类型: {type(last_position)}")
                            pos_dict = {}
                        
                        # 提取股票数据
                        for stock, info in pos_dict.items():
                            if stock not in ['cash', 'now_account_value'] and isinstance(info, dict):
                                stocks_data.append({
                                    '股票代码': stock,
                                    '权重': float(info.get('weight', 0)),
                                    '持仓天数': int(info.get('count_day', 0)),
                                    '持仓数量': float(info.get('amount', 0)),
                                    '价格': float(info.get('price', 0)),
                                    '持仓金额': float(info.get('amount', 0)) * float(info.get('price', 0))
                                })
                        
                    except Exception as e:
                        st.error(f"处理持仓数据时出错: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                    
                    if stocks_data:
                        df_positions = pd.DataFrame(stocks_data)
                        st.dataframe(df_positions)
                    else:
                        st.write("没有找到股票持仓数据")
                else:
                    st.write("没有找到持仓数据")
            
            # 右侧信息区（模拟右侧边栏）
            with info_col:
                # 显示当前使用的训练记录（从回测记录中获取）
                st.subheader("当前使用的训练记录")
                try:
                    # 从回测记录中获取训练记录ID
                    # 兼容两种获取方式
                    train_recorder_id = None
                    
                    # 方式1：直接使用recorder对象的list_params方法
                    params = recorder.list_params()
                    if 'train_recorder_id' in params and params['train_recorder_id']:
                        train_recorder_id = params['train_recorder_id']
                    
                    # 方式2：直接从MLflow客户端获取（备用方案）
                    if not train_recorder_id:
                        import mlflow
                        mlflow_client = mlflow.tracking.MlflowClient()
                        run = mlflow_client.get_run(recorder.id)
                        if 'train_recorder_id' in run.data.params and run.data.params['train_recorder_id']:
                            train_recorder_id = run.data.params['train_recorder_id']
                    
                    # 确保train_recorder_id不是空字符串
                    if train_recorder_id and train_recorder_id != "":
                        # 获取训练记录信息
                        train_recorder = R.get_recorder(recorder_id=train_recorder_id, experiment_name="train_model")
                        train_info = train_recorder.info
                        train_time = train_info.get('start_time', '未知')
                        
                        # 获取训练记录的参数，包括市场信息
                        train_params = train_recorder.list_params()
                        train_market = train_params.get('market', '未知')
                        
                        # 显示训练ID、时间和市场
                        st.markdown(f"**训练ID**: {train_recorder_id}")
                        st.markdown(f"**训练时间**: {train_time}")
                        st.markdown(f"**训练市场**: {train_market}")
                    else:
                        st.write("未找到关联的训练记录")
                        st.write("提示：对于旧的回测记录，可能需要重新执行回测才能看到关联的训练记录信息")
                except Exception as e:
                    st.error(f"获取训练记录信息失败: {e}")
                
                # 显示当前回测ID和时间
                st.subheader("当前回测记录")
                try:
                    # 显示当前回测ID、时间和市场
                    backtest_info = recorder.info
                    backtest_time = backtest_info.get('start_time', '未知')
                    
                    # 获取回测市场信息
                    backtest_params = recorder.list_params()
                    backtest_market = backtest_params.get('backtest_market', backtest_params.get('market', '未知'))
                    
                    st.markdown(f"**回测ID**: {recorder.id}")
                    st.markdown(f"**回测时间**: {backtest_time}")
                    st.markdown(f"**回测市场**: {backtest_market}")
                except Exception as e:
                    st.error(f"获取回测记录信息失败: {e}")
                
                # 显示使用的参数（从回测记录中获取）
                st.subheader("使用的参数")
                
                # 辅助函数：将字符串转换为数值类型
                def to_numeric(value, default=0):
                    if isinstance(value, (int, float)):
                        return value
                    if isinstance(value, str):
                        try:
                            return float(value)
                        except ValueError:
                            return default
                    return default
                
                # 创建参数表格数据，从回测记录的params中获取
                params_data = [
                    {"参数类型": "回测参数", "参数名称": "回测市场", "数值": str(params.get('backtest_market', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "回测基准指数", "数值": str(params.get('backtest_benchmark', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "回测开始日期", "数值": str(params.get('backtest_start_date', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "回测结束日期", "数值": str(params.get('backtest_end_date', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "初始资金", "数值": f"{to_numeric(params.get('initial_account', 0)):,.0f}"},
                    {"参数类型": "回测参数", "参数名称": "Topk", "数值": str(params.get('topk', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "每次调仓卖出数量", "数值": str(params.get('n_drop', '未知'))},
                    {"参数类型": "回测参数", "参数名称": "策略类型", "数值": str(params.get('strategy_type', '未知'))}
                ]
                
                # 显示参数表格
                params_df = pd.DataFrame(params_data)
                st.dataframe(params_df, hide_index=True)
        else:
            st.write("暂无回测结果，请先执行回测。")
    
    except Exception as e:
        st.error(f"加载回测结果失败: {e}")
        import traceback
        st.error(traceback.format_exc())

# 显示回测结果
# 从session_state获取用户选择的回测记录ID，如果没有则显示最新的回测结果
show_backtest_results(st.session_state.get('selected_ba_rid'))