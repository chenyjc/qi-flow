"""
QiFlow CLI Core — 对 QlibService 的精简封装，供 CLI/TUI 调用。

所有函数均为同步阻塞调用，内部通过 progress_callback 报告进度。
"""

import sys
import os
import datetime

# 确保可以 import backend 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.qlib_service import QlibService


def _get_service():
    """延迟初始化 QlibService 单例"""
    if not hasattr(_get_service, "_instance"):
        _get_service._instance = QlibService()
    return _get_service._instance


# ---------- 默认参数 ----------

MARKET_BENCHMARK = {
    "csi300": "SH000300",
    "csi500": "SH000905",
    "csi800": "SH000906",
    "csi1000": "SH000852",
}

DEFAULT_MARKET = "csi300"
DEFAULT_MODEL = "LGBModel"
DEFAULT_HANDLER = "Alpha158"
DEFAULT_LABEL_HORIZON = 5  # 5=周收益率预测（调优结论：IC比次日预测高2-3倍，超额收益显著）

LABEL_HORIZON_OPTIONS = {1: "次日", 5: "周(5日)", 10: "双周(10日)", 20: "月(20日)"}

# 参考 Qlib benchmarks_dynamic：训练窗口 7 年，提供充足历史数据
def _default_dates():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    seven_years_ago = yesterday.replace(year=yesterday.year - 7)
    five_months_ago = _months_ago(yesterday, 5)
    three_months_ago = _months_ago(yesterday, 3)
    return {
        "train_start": seven_years_ago.strftime("%Y-%m-%d"),
        "train_end": five_months_ago.strftime("%Y-%m-%d"),
        "valid_start": five_months_ago.strftime("%Y-%m-%d"),
        "valid_end": three_months_ago.strftime("%Y-%m-%d"),
        "test_start": three_months_ago.strftime("%Y-%m-%d"),
        "test_end": yesterday.strftime("%Y-%m-%d"),
    }


def _months_ago(d, months):
    """日期往前推 N 个月"""
    m = d.month - months
    y = d.year
    while m <= 0:
        m += 12
        y -= 1
    import calendar
    max_day = calendar.monthrange(y, m)[1]
    return d.replace(year=y, month=m, day=min(d.day, max_day))


# ---------- 公开 API ----------

def update_data(progress_callback=None):
    """下载/更新 Qlib 数据
    
    Returns: dict with success, message
    """
    svc = _get_service()
    return svc.download_data_with_progress(progress_callback)


def train(
    market=DEFAULT_MARKET,
    model_type=DEFAULT_MODEL,
    handler_type=DEFAULT_HANDLER,
    label_horizon=DEFAULT_LABEL_HORIZON,
    dates=None,
    progress_callback=None,
):
    """训练模型（使用大量默认参数）

    Parameters
    ----------
    market : str
    model_type : str
    handler_type : str
    label_horizon : int
        预测周期（天），1=次日, 5=周, 10=双周, 20=月
    dates : dict, optional
        包含 train_start, train_end, valid_start, valid_end, test_start, test_end
    progress_callback : callable(progress: int, message: str)

    Returns: dict with success, recorder_id, message
    """
    svc = _get_service()
    if dates is None:
        dates = _default_dates()

    benchmark = MARKET_BENCHMARK.get(market, "SH000300")
    cb = progress_callback or (lambda p, m: None)

    return svc.train_model_with_progress(
        progress_callback=cb,
        market=market,
        benchmark=benchmark,
        train_start_date=dates["train_start"],
        train_end_date=dates["train_end"],
        valid_start_date=dates["valid_start"],
        valid_end_date=dates["valid_end"],
        test_start_date=dates["test_start"],
        test_end_date=dates["test_end"],
        model_type=model_type,
        lr=0.2,
        max_depth=8,
        num_leaves=210,
        subsample=0.8789,
        colsample_bytree=0.8879,
        seed=42,
        num_threads=-1,
        handler_type=handler_type,
        label_horizon=label_horizon,
    )


def backtest(
    recorder_id,
    market=DEFAULT_MARKET,
    start_date=None,
    end_date=None,
    topk=10,
    n_drop=1,
    hold_days=None,
    initial_account=1000000,
):
    """执行回测

    Parameters
    ----------
    recorder_id : str  — 训练记录 ID
    start_date / end_date : 默认取最近 3 个月
    hold_days : int, optional
        持仓天数，默认根据训练时的 label_horizon 自动推断

    Returns: dict with success, recorder_id, message
    """
    svc = _get_service()
    if start_date is None:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        start_date = _months_ago(yesterday, 3).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # 自动从训练参数推断 hold_days（与 label_horizon 对齐）
    if hold_days is None:
        try:
            from qlib.workflow import R
            rec = R.get_recorder(recorder_id=recorder_id, experiment_name="train_model")
            params = rec.list_params()
            label_horizon = int(params.get('label_horizon', 1))
            hold_days = max(label_horizon, 1)
        except Exception:
            hold_days = 1

    benchmark = MARKET_BENCHMARK.get(market, "SH000300")

    return svc.backtest_model(
        recorder_id=recorder_id,
        market=market,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        initial_account=initial_account,
        topk=topk,
        n_drop=n_drop,
        hold_days=hold_days,
        stop_loss=0.0,
        strategy_type="TopkDropoutStrategy",
        seed=42,
        deal_price="close",
        open_cost=0.0005,
        close_cost=0.0015,
        limit_threshold=0.095,
        only_tradable=True,
    )


def train_rolling(
    market=DEFAULT_MARKET,
    model_type=DEFAULT_MODEL,
    handler_type=DEFAULT_HANDLER,
    label_horizon=DEFAULT_LABEL_HORIZON,
    rolling_step=20,
    dates=None,
    progress_callback=None,
):
    """滚动重训练（Qlib benchmarks_dynamic 核心方法）

    将测试期按 rolling_step 切片，每个切片重新训练模型，合并预测。
    这是 IC 从 ~0.04 提升到 ~0.09 的最大贡献方法。

    Parameters
    ----------
    rolling_step : int
        滚动步长（交易日数），默认20（约1个月）
    """
    svc = _get_service()
    if dates is None:
        dates = _default_dates()

    benchmark = MARKET_BENCHMARK.get(market, "SH000300")
    cb = progress_callback or (lambda p, m: None)

    return svc.train_rolling_with_progress(
        progress_callback=cb,
        market=market,
        benchmark=benchmark,
        train_start_date=dates["train_start"],
        train_end_date=dates["train_end"],
        test_start_date=dates["test_start"],
        test_end_date=dates["test_end"],
        model_type=model_type,
        lr=0.2,
        max_depth=8,
        num_leaves=210,
        subsample=0.8789,
        colsample_bytree=0.8879,
        seed=42,
        num_threads=-1,
        handler_type=handler_type,
        label_horizon=label_horizon,
        rolling_step=rolling_step,
    )


def predict(recorder_id, market=DEFAULT_MARKET, topk=10, n_drop=1):
    """生成今日交易信号

    Returns: dict with success, buy_list, full_scores, prediction_date, ...
    """
    svc = _get_service()
    return svc.predict_today(
        recorder_id=recorder_id,
        market=market,
        topk=topk,
        n_drop=n_drop,
    )


def list_recorders():
    """列出所有训练记录"""
    svc = _get_service()
    return svc.get_recorders("train_model")


def list_backtest_recorders():
    """列出所有回测记录"""
    svc = _get_service()
    return svc.get_backtest_recorders()


def get_backtest_result(recorder_id):
    """获取回测结果摘要"""
    svc = _get_service()
    return svc.get_backtest_result(recorder_id)


def get_train_result(recorder_id):
    """获取训练结果（指标 + 因子重要性）"""
    svc = _get_service()
    return svc.get_train_result(recorder_id)


def preview_data(market=DEFAULT_MARKET):
    """预览市场数据（最近30天行情）"""
    svc = _get_service()
    return svc.preview_data(market)


def delete_train_recorder(recorder_id):
    """删除单条训练记录"""
    svc = _get_service()
    return svc.delete_train_recorder(recorder_id)


def delete_all_train_recorders():
    """删除所有训练记录"""
    svc = _get_service()
    return svc.delete_all_train_recorders()


def delete_backtest_recorder(recorder_id):
    """删除单条回测记录"""
    svc = _get_service()
    return svc.delete_backtest_recorder(recorder_id)


def delete_all_backtest_recorders():
    """删除所有回测记录"""
    svc = _get_service()
    return svc.delete_all_backtest_recorders()
