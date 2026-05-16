"""QiFlow CLI 入口 — 基于 click 的命令行工具"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
from rich.console import Console
from rich.table import Table

console = Console()

MARKETS = ["csi300", "csi500", "csi800", "csi1000"]
MODELS = [
    "LGBModel", "DEnsembleModel", "XGBModel", "CatBoostModel", "Linear",
    "LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM", "TabnetModel",
]
HANDLERS = ["Alpha158", "Alpha360", "Alpha158DL", "Alpha360DL", "Alpha158vwap", "Alpha360vwap"]


def _progress_cb(pct, msg):
    console.print(f"  [{pct:3d}%] {msg}")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """QiFlow — 量化策略回测平台 CLI

    \b
    快速开始:
      qiflow-cli tui                        # 交互式 TUI 界面
      qiflow-cli update                     # 更新行情数据
      qiflow-cli train                      # 训练模型（默认参数）
      qiflow-cli list                       # 查看训练记录
      qiflow-cli eval     <recorder_id>     # 查看模型指标+因子重要性
      qiflow-cli backtest <recorder_id>     # 回测
      qiflow-cli predict  <recorder_id>     # 今日交易信号
      qiflow-cli delete -t train --all      # 删除全部训练记录
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def tui():
    """启动交互式 TUI 界面"""
    from cli.app import main
    main()


@cli.command()
def update():
    """下载/更新 Qlib 行情数据（~500MB）"""
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import update_data
    console.print("[bold]📥 开始更新数据...[/]")
    result = update_data(progress_callback=_progress_cb)
    if result and result.get("success"):
        console.print("[bold green]✓ 完成[/]")
    else:
        console.print(f"[bold red]✗ {result.get('message', '?') if result else '失败'}[/]")
        sys.exit(1)


@cli.command()
@click.option("-m", "--market", type=click.Choice(MARKETS), default="csi300", show_default=True, help="市场")
@click.option("-M", "--model", "model_type", type=click.Choice(MODELS), default="LGBModel", show_default=True, help="模型")
@click.option("-H", "--handler", "handler_type", type=click.Choice(HANDLERS), default="Alpha158", show_default=True, help="因子集")
@click.option("--horizon", "label_horizon", type=click.Choice(["1", "5", "10", "20"]), default="1", show_default=True, help="预测周期(天): 1=次日 5=周 10=双周 20=月")
@click.option("--train-start", default=None, help="训练开始日期 (默认: 2年前)")
@click.option("--train-end", default=None, help="训练结束日期 (默认: 5个月前)")
@click.option("--valid-start", default=None, help="验证开始日期 (默认: 5个月前)")
@click.option("--valid-end", default=None, help="验证结束日期 (默认: 3个月前)")
@click.option("--test-start", default=None, help="测试开始日期 (默认: 3个月前)")
@click.option("--test-end", default=None, help="测试结束日期 (默认: 昨天)")
def train(market, model_type, handler_type, label_horizon, train_start, train_end, valid_start, valid_end, test_start, test_end):
    """训练预测模型"""
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import train as do_train, _default_dates, LABEL_HORIZON_OPTIONS

    horizon = int(label_horizon)
    dates = _default_dates()
    if train_start: dates["train_start"] = train_start
    if train_end:   dates["train_end"] = train_end
    if valid_start: dates["valid_start"] = valid_start
    if valid_end:   dates["valid_end"] = valid_end
    if test_start:  dates["test_start"] = test_start
    if test_end:    dates["test_end"] = test_end

    horizon_desc = LABEL_HORIZON_OPTIONS.get(horizon, f"{horizon}日")
    console.print(f"[bold]🧠 训练: 市场={market} 模型={model_type} 因子={handler_type} 预测周期={horizon_desc}[/]")
    console.print(f"[dim]  训练 {dates['train_start']}~{dates['train_end']}  "
                  f"验证 {dates['valid_start']}~{dates['valid_end']}  "
                  f"测试 {dates['test_start']}~{dates['test_end']}[/]")

    result = do_train(market=market, model_type=model_type, handler_type=handler_type,
                      label_horizon=horizon, dates=dates, progress_callback=_progress_cb)
    if result and result.get("success"):
        console.print(f"[bold green]✓ 训练完成！记录ID: {result['recorder_id']}[/]")
    else:
        console.print(f"[bold red]✗ {result.get('message', '?') if result else '失败'}[/]")
        sys.exit(1)


@cli.command()
@click.argument("recorder_id")
@click.option("-m", "--market", type=click.Choice(MARKETS), default="csi300", show_default=True, help="市场")
@click.option("-k", "--topk", default=10, show_default=True, help="持仓股票数")
@click.option("-d", "--n-drop", default=1, show_default=True, help="每日换仓数")
@click.option("--start", "start_date", default=None, help="回测起始日期 (默认: 3个月前)")
@click.option("--end", "end_date", default=None, help="回测结束日期 (默认: 昨天)")
@click.option("--account", "initial_account", default=1000000, show_default=True, help="初始资金（元）")
def backtest(recorder_id, market, topk, n_drop, start_date, end_date, initial_account):
    """执行策略回测

    RECORDER_ID: 训练记录ID（通过 list 命令查看）
    """
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import backtest as do_backtest

    console.print(f"[bold]📊 回测: 记录={recorder_id[:12]}... 市场={market} TopK={topk}[/]")
    result = do_backtest(recorder_id=recorder_id, market=market, topk=topk,
                         n_drop=n_drop, start_date=start_date, end_date=end_date,
                         initial_account=initial_account)
    if result and result.get("success"):
        console.print(f"[bold green]✓ 回测完成！回测记录ID: {result['recorder_id']}[/]")
    else:
        console.print(f"[bold red]✗ {result.get('message', '?') if result else '失败'}[/]")
        sys.exit(1)


@cli.command()
@click.argument("recorder_id")
@click.option("-m", "--market", type=click.Choice(MARKETS), default="csi300", show_default=True, help="市场")
@click.option("-k", "--topk", default=10, show_default=True, help="推荐股票数")
@click.option("-d", "--n-drop", default=1, show_default=True, help="换仓数")
def predict(recorder_id, market, topk, n_drop):
    """生成今日交易信号（买入推荐）

    RECORDER_ID: 训练记录ID（通过 list 命令查看）
    """
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import predict as do_predict

    console.print(f"[bold]🔮 预测: 记录={recorder_id[:12]}... 市场={market} TopK={topk}[/]")
    result = do_predict(recorder_id=recorder_id, market=market, topk=topk, n_drop=n_drop)
    if result and result.get("success"):
        console.print(f"[bold green]✓ 预测日期: {result['prediction_date']}[/]")
        table = Table(title="买入推荐")
        table.add_column("#", width=4)
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("评分", justify="right", style="green")
        for i, item in enumerate(result.get("buy_list", []), 1):
            table.add_row(str(i), item["stock_code"], item["stock_name"], f"{item['score']:.4f}")
        console.print(table)
    else:
        console.print(f"[bold red]✗ {result.get('message', '?') if result else '失败'}[/]")
        sys.exit(1)


@cli.command("list")
@click.option("-t", "--type", "rec_type", type=click.Choice(["train", "backtest"]),
              default="train", show_default=True, help="记录类型")
def list_cmd(rec_type):
    """查看训练/回测记录"""
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import list_recorders, list_backtest_recorders

    if rec_type == "train":
        recs = list_recorders()
        title = "训练记录"
    else:
        recs = list_backtest_recorders()
        title = "回测记录"

    if recs.get("success") and recs.get("recorders"):
        table = Table(title=title)
        table.add_column("ID", style="cyan")
        table.add_column("名称", max_width=45)
        table.add_column("模型", style="yellow")
        table.add_column("市场")
        table.add_column("时间", style="dim")
        for rec in recs["recorders"][-15:]:
            params = rec.get("params", {})
            table.add_row(
                rec["id"],
                rec.get("name", "")[:45],
                params.get("model_type", "?"),
                params.get("market", params.get("backtest_market", "?")),
                str(rec.get("start_time", ""))[:19],
            )
        console.print(table)
    else:
        console.print(f"[yellow]暂无{title}[/]")


@cli.command()
@click.argument("recorder_id")
def result(recorder_id):
    """查看回测结果详情

    RECORDER_ID: 回测记录ID（通过 list -t backtest 查看）
    """
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import get_backtest_result

    res = get_backtest_result(recorder_id)
    if not res or not res.get("success"):
        console.print(f"[red]获取失败: {res.get('message', '?') if res else '?'}[/]")
        sys.exit(1)

    metrics = res.get("key_metrics", {})
    table = Table(title="回测结果")
    table.add_column("指标", style="bold")
    table.add_column("值", justify="right")
    table.add_row("总收益率", f"[{'green' if metrics.get('total_return', 0) > 0 else 'red'}]{metrics.get('total_return', 0):.2f}%[/]")
    table.add_row("基准收益", f"{metrics.get('bench_return', 0):.2f}%")
    table.add_row("超额收益", f"[{'green' if metrics.get('excess_return', 0) > 0 else 'red'}]{metrics.get('excess_return', 0):.2f}%[/]")
    table.add_row("年化收益", f"{metrics.get('annualized_return', 0):.4f}")
    table.add_row("信息比率", f"{metrics.get('information_ratio', 0):.3f}")
    table.add_row("最大回撤", f"[red]{metrics.get('max_drawdown', 0):.4f}[/]")
    console.print(table)


@cli.command("eval")
@click.argument("recorder_id")
@click.option("--top", "top_n", default=20, show_default=True, help="显示前N个重要因子")
def eval_cmd(recorder_id, top_n):
    """查看训练结果（IC指标 + 因子重要性）

    RECORDER_ID: 训练记录ID（通过 list 命令查看）
    """
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import get_train_result, LABEL_HORIZON_OPTIONS

    res = get_train_result(recorder_id)
    if not res or not res.get("success"):
        console.print(f"[red]获取失败: {res.get('message', '?') if res else '?'}[/]")
        sys.exit(1)

    # 指标表
    metrics = res.get("metrics", {})
    params = res.get("params", {})

    label_horizon = int(params.get('label_horizon', 1))
    horizon_desc = LABEL_HORIZON_OPTIONS.get(label_horizon, f"{label_horizon}日")
    console.print(f"\n[bold]模型: {params.get('model_type', '?')}  市场: {params.get('market', '?')}  "
                  f"因子: {params.get('handler_type', '?')}  预测周期: {horizon_desc}[/]")
    console.print(f"[dim]训练 {params.get('train_start_date', '?')}~{params.get('train_end_date', '?')}  "
                  f"测试 {params.get('test_start_date', '?')}~{params.get('test_end_date', '?')}[/]\n")

    table = Table(title="模型评估指标")
    table.add_column("指标", style="bold")
    table.add_column("值", justify="right")
    table.add_column("评价")

    def _grade(val, thresholds):
        """根据阈值给出评价"""
        if val is None: return "[dim]N/A[/]"
        if val > thresholds[0]: return "[bold green]优秀[/]"
        if val > thresholds[1]: return "[green]良好[/]"
        if val > 0: return "[yellow]一般[/]"
        return "[red]差[/]"

    ic = metrics.get("IC")
    icir = metrics.get("ICIR")
    ric = metrics.get("Rank_IC")
    ricir = metrics.get("Rank_ICIR")
    lp = metrics.get("Long_precision")
    sp = metrics.get("Short_precision")

    table.add_row("IC", f"{ic:.4f}" if ic else "N/A", _grade(ic, [0.05, 0.03]))
    table.add_row("ICIR", f"{icir:.4f}" if icir else "N/A", _grade(icir, [0.5, 0.3]))
    table.add_row("Rank IC", f"{ric:.4f}" if ric else "N/A", _grade(ric, [0.05, 0.03]))
    table.add_row("Rank ICIR", f"{ricir:.4f}" if ricir else "N/A", _grade(ricir, [0.5, 0.3]))
    table.add_row("多头精度", f"{lp*100:.1f}%" if lp else "N/A", _grade(lp, [0.55, 0.52]) if lp else "[dim]N/A[/]")
    table.add_row("空头精度", f"{sp*100:.1f}%" if sp else "N/A", _grade(sp, [0.55, 0.52]) if sp else "[dim]N/A[/]")
    console.print(table)

    # 因子重要性
    fi = res.get("feature_importance")
    if fi:
        console.print()
        fi_table = Table(title=f"因子重要性 (Top {top_n})")
        fi_table.add_column("#", style="dim", width=4)
        fi_table.add_column("因子名称", style="cyan")
        fi_table.add_column("重要性", justify="right", style="green")
        fi_table.add_column("占比", justify="right")
        total = sum(item["importance"] for item in fi)
        for i, item in enumerate(fi[:top_n], 1):
            pct = item["importance"] / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (20 - int(pct / 2))
            fi_table.add_row(str(i), item["name"], f"{item['importance']:.0f}", f"{pct:.1f}% {bar}")
        console.print(fi_table)
    else:
        console.print("\n[dim]该模型不支持因子重要性分析（仅 LGBModel/XGBModel/CatBoostModel 支持）[/]")


@cli.command("delete")
@click.option("-t", "--type", "rec_type", type=click.Choice(["train", "backtest"]),
              required=True, help="记录类型")
@click.option("--id", "recorder_id", default=None, help="要删除的记录ID（不指定则删除全部）")
@click.option("--all", "delete_all", is_flag=True, help="删除该类型全部记录")
@click.option("-y", "--yes", "skip_confirm", is_flag=True, help="跳过确认")
def delete_cmd(rec_type, recorder_id, delete_all, skip_confirm):
    """删除训练或回测记录

    \b
    示例:
      qiflow-cli delete -t train --id <recorder_id>   # 删除单条训练记录
      qiflow-cli delete -t train --all                 # 删除全部训练记录
      qiflow-cli delete -t backtest --id <recorder_id> # 删除单条回测记录
      qiflow-cli delete -t backtest --all -y           # 删除全部回测（跳过确认）
    """
    os.environ["TQDM_DISABLE"] = "1"
    from cli.core import (
        delete_train_recorder, delete_all_train_recorders,
        delete_backtest_recorder, delete_all_backtest_recorders,
        list_recorders, list_backtest_recorders,
    )

    if not recorder_id and not delete_all:
        console.print("[red]请指定 --id <recorder_id> 或 --all[/]")
        sys.exit(1)

    type_label = "训练" if rec_type == "train" else "回测"

    if recorder_id:
        if not skip_confirm:
            click.confirm(f"确认删除{type_label}记录 {recorder_id[:12]}...？", abort=True)
        if rec_type == "train":
            result = delete_train_recorder(recorder_id)
        else:
            result = delete_backtest_recorder(recorder_id)
    else:
        # 删除全部
        if rec_type == "train":
            recs = list_recorders()
        else:
            recs = list_backtest_recorders()
        count = len(recs.get("recorders", [])) if recs.get("success") else 0
        if count == 0:
            console.print(f"[yellow]暂无{type_label}记录[/]")
            return
        if not skip_confirm:
            click.confirm(f"确认删除全部 {count} 条{type_label}记录？此操作不可恢复！", abort=True)
        if rec_type == "train":
            result = delete_all_train_recorders()
        else:
            result = delete_all_backtest_recorders()

    if result.get("success"):
        console.print(f"[bold green]✓ {result['message']}[/]")
    else:
        console.print(f"[bold red]✗ {result['message']}[/]")
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
