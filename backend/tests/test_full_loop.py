import sys, os, time, logging
logging.disable(logging.WARNING)
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
from backend.services.qlib_service import QlibService

svc = QlibService()

if __name__ == '__main__':
    # === 迭代5: h=5 周收益率，更强信号 ===
    # 周收益率比日收益率信噪比更高 (学术文献共识)
    # rolling_step=20 → ~4轮滚动
    print("=== Iter5: h=5 (weekly return), official params ===")
    res = svc.train_rolling_with_progress(
        progress_callback=lambda p,m: print(f'[{p}%] {m}'),
        market='csi300', benchmark='SH000300',
        train_start_date='2015-01-01', train_end_date='2025-12-31',
        test_start_date='2026-01-01', test_end_date='2026-05-15',
        model_type='LGBModel', lr=0.2, max_depth=8, num_leaves=210,
        subsample=0.8789, colsample_bytree=0.8879, seed=42, num_threads=4,
        handler_type='Alpha158', label_horizon=5, rolling_step=20
    )

    ic = res.get('metrics', {}).get('IC', 0)
    icir = res.get('metrics', {}).get('ICIR', 0)
    rank_ic = res.get('metrics', {}).get('Rank_IC', 0)
    print(f'\n=== TRAIN RESULT ===')
    print(f'IC={ic:.4f}  ICIR={icir:.4f}  Rank_IC={rank_ic:.4f}')

    if not res.get('success'):
        print("Train FAILED:", res.get('message'))
        sys.exit(1)

    print("\n--- Running Backtest ---")
    bt = svc.backtest_model(
        recorder_id=res['recorder_id'],
        market='csi300', benchmark='SH000300',
        start_date='2026-01-01', end_date='2026-05-15',
        initial_account=100000000, topk=10, n_drop=1, hold_days=5,
        stop_loss=0.0, strategy_type='TopkDropoutStrategy',
        deal_price='close', open_cost=0.0005, close_cost=0.0015,
        limit_threshold=0.095, only_tradable=True
    )

    if not bt.get('success'):
        print("Backtest FAILED:", bt.get('message'))
        sys.exit(1)

    bt_res = svc.get_backtest_result(bt['recorder_id'])
    km = bt_res.get('key_metrics', {})
    print(f'\n=== BACKTEST RESULT ===')
    print(f"Total Return: {km.get('total_return', 'N/A')}%")
    print(f"Bench Return: {km.get('bench_return', 'N/A')}%")
    print(f"Excess Return: {km.get('excess_return', 'N/A')}%")
    print(f"Annualized Return: {km.get('annualized_return', 'N/A')}")
    print(f"Information Ratio: {km.get('information_ratio', 'N/A')}")
    print(f"Max Drawdown: {km.get('max_drawdown', 'N/A')}")
    print(f"\n=== SUMMARY: IC={ic:.4f} | Excess={km.get('excess_return', 'N/A')}% | IR={km.get('information_ratio', 'N/A')} ===")
