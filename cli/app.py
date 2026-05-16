"""
QiFlow TUI — 基于 rich + prompt_toolkit 的交互式终端界面
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore", message=".*cursor position.*")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

console = Console()

MARKETS = ["csi300", "csi500", "csi800", "csi1000"]
MODELS = [
    "DEnsembleModel", "LGBModel", "XGBModel", "CatBoostModel", "Linear",
    "LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM", "TabnetModel",
]
HANDLERS = ["Alpha158", "Alpha360", "Alpha158DL", "Alpha360DL", "Alpha158vwap", "Alpha360vwap"]


def _lazy_import():
    """延迟导入重量级模块，避免启动时卡住"""
    from cli.core import (
        update_data, train, backtest, predict,
        list_recorders, list_backtest_recorders, get_backtest_result,
        get_train_result, preview_data,
        delete_train_recorder, delete_all_train_recorders,
        delete_backtest_recorder, delete_all_backtest_recorders,
        train_rolling,
        DEFAULT_MARKET, DEFAULT_MODEL, DEFAULT_HANDLER,
        DEFAULT_LABEL_HORIZON, LABEL_HORIZON_OPTIONS,
        _default_dates, _months_ago,
    )
    return {
        "update_data": update_data, "train": train, "backtest": backtest,
        "predict": predict, "list_recorders": list_recorders,
        "list_backtest_recorders": list_backtest_recorders,
        "get_backtest_result": get_backtest_result,
        "get_train_result": get_train_result,
        "preview_data": preview_data,
        "delete_train_recorder": delete_train_recorder,
        "delete_all_train_recorders": delete_all_train_recorders,
        "delete_backtest_recorder": delete_backtest_recorder,
        "delete_all_backtest_recorders": delete_all_backtest_recorders,
        "train_rolling": train_rolling,
        "DEFAULT_MARKET": DEFAULT_MARKET, "DEFAULT_MODEL": DEFAULT_MODEL,
        "DEFAULT_HANDLER": DEFAULT_HANDLER,
        "DEFAULT_LABEL_HORIZON": DEFAULT_LABEL_HORIZON,
        "LABEL_HORIZON_OPTIONS": LABEL_HORIZON_OPTIONS,
        "_default_dates": _default_dates,
        "_months_ago": _months_ago,
    }


_core = None

def _get_core():
    global _core
    if _core is None:
        with console.status("[bold cyan]正在初始化 Qlib 引擎...[/]", spinner="dots"):
            _core = _lazy_import()
    return _core


def _header():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]QiFlow[/] — 量化策略回测平台\n"
        "[dim]交互式终端界面  |  Ctrl+C 退出[/]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    console.print()
    console.print("[dim]典型工作流: 更新数据 → 训练模型 → 回测 → 预测[/]")
    console.print("[dim]所有参数均有默认值，直接回车即可快速体验[/]")
    console.print()


def _menu():
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style="bold yellow", width=6)
    table.add_column()
    table.add_row("[1]", "📥 更新数据     下载最新 Qlib 日线行情")
    table.add_row("[2]", "🔮 今日预测     生成买入推荐信号")
    table.add_row("[3]", "📊 预览数据     查看市场行情数据样本")
    table.add_row("[4]", "🧠 训练模型     默认: LGBModel / csi300 / Alpha158")
    table.add_row("[5]", "🔄 滚动训练    滚动重训练（高IC核心方法）")
    table.add_row("[6]", "📈 训练评估     查看模型 IC 指标与因子重要性")
    table.add_row("[7]", "📊 策略回测     对训练好的模型执行回测")
    table.add_row("[8]", "📋 训练记录     查看已有训练记录")
    table.add_row("[9]", "📋 回测记录     查看回测记录及结果")
    table.add_row("[10]", "🗑  删除记录     删除训练/回测记录")
    console.print(table)
    console.print("[dim]输入编号选择操作，Ctrl+C 或 q 退出[/]")


def _prompt(msg, default="", completer=None):
    """带默认值的输入提示，Ctrl+C 抛出 KeyboardInterrupt"""
    suffix = f" [{default}]" if default else ""
    try:
        val = prompt(f"{msg}{suffix}: ", completer=completer).strip()
    except EOFError:
        return default
    return val if val else default


# ---------- 1. 更新数据 ----------

def do_update_data():
    console.print("\n[bold]📥 更新 Qlib 数据[/]")
    console.print("[dim]从 GitHub 下载最新日线数据（~500MB），需要科学上网[/]\n")

    confirm = _prompt("确认开始下载？(y/n)", "y")
    if confirm.lower() != "y":
        console.print("[yellow]已取消[/]")
        return

    core = _get_core()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("下载中...", total=100)

        def cb(pct, msg):
            progress.update(task, completed=pct, description=msg)

        result = core["update_data"](progress_callback=cb)

    if result and result.get("success"):
        console.print("[bold green]✓ 数据更新完成！[/]")
    else:
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 失败: {msg}[/]")


# ---------- 2. 预览数据 ----------

def do_preview_data():
    console.print("\n[bold]📊 预览市场数据[/]")
    console.print("[dim]从 Qlib 加载最近行情数据样本[/]\n")

    core = _get_core()
    market = _prompt("市场", core["DEFAULT_MARKET"], WordCompleter(MARKETS))

    with console.status("[bold cyan]加载数据中...[/]", spinner="dots"):
        result = core["preview_data"](market)

    if not result or not result.get("success"):
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 失败: {msg}[/]")
        return

    data = result.get("data", {})
    stock_data_list = data.get("data", [])
    dates = data.get("dates", [])

    if not stock_data_list:
        console.print("[yellow]无数据[/]")
        return

    stock_info = stock_data_list[0]
    stock_code = stock_info.get("stock", "?")

    console.print(f"\n[bold]市场: [cyan]{market}[/]  样本股票: [cyan]{stock_code}[/]  "
                  f"日期范围: {result.get('start_date', '?')} ~ {result.get('end_date', '?')}[/]\n")

    table = Table(title=f"{stock_code} 最近行情", box=box.ROUNDED, show_lines=True)
    table.add_column("日期", style="dim")
    table.add_column("开盘", justify="right")
    table.add_column("收盘", justify="right", style="bold")
    table.add_column("最高", justify="right", style="green")
    table.add_column("最低", justify="right", style="red")
    table.add_column("成交量", justify="right")
    table.add_column("涨跌", justify="right")

    opens = stock_info.get("open", [])
    closes = stock_info.get("close", [])
    highs = stock_info.get("high", [])
    lows = stock_info.get("low", [])
    volumes = stock_info.get("volume", [])

    for i in range(min(len(dates), len(closes))):
        o = opens[i] if i < len(opens) else 0
        c = closes[i] if i < len(closes) else 0
        h = highs[i] if i < len(highs) else 0
        l = lows[i] if i < len(lows) else 0
        v = volumes[i] if i < len(volumes) else 0

        # 涨跌幅
        change = ""
        if i + 1 < len(closes) and closes[i + 1] and closes[i + 1] != 0:
            pct = (c - closes[i + 1]) / closes[i + 1] * 100
            color = "green" if pct >= 0 else "red"
            change = f"[{color}]{pct:+.2f}%[/]"
        elif o and o != 0:
            pct = (c - o) / o * 100
            color = "green" if pct >= 0 else "red"
            change = f"[{color}]{pct:+.2f}%[/]"

        vol_str = f"{v/10000:.0f}万" if v and v > 10000 else f"{v:.0f}" if v else "-"

        table.add_row(
            dates[i] if i < len(dates) else "?",
            f"{o:.2f}" if o else "-",
            f"{c:.2f}" if c else "-",
            f"{h:.2f}" if h else "-",
            f"{l:.2f}" if l else "-",
            vol_str,
            change,
        )

    console.print(table)


# ---------- 3. 训练 ----------

def do_train():
    console.print("\n[bold]🧠 训练模型[/]")
    core = _get_core()
    dates = core["_default_dates"]()
    console.print(f"[dim]默认日期: 训练 {dates['train_start']}~{dates['train_end']}, "
                  f"验证 {dates['valid_start']}~{dates['valid_end']}, "
                  f"测试 {dates['test_start']}~{dates['test_end']}[/]\n")

    market = _prompt("市场", core["DEFAULT_MARKET"], WordCompleter(MARKETS))
    model_type = _prompt("模型", core["DEFAULT_MODEL"], WordCompleter(MODELS))
    handler_type = _prompt("因子", core["DEFAULT_HANDLER"], WordCompleter(HANDLERS))

    # 预测周期选择
    horizon_options = core["LABEL_HORIZON_OPTIONS"]
    horizon_desc = "  ".join(f"{k}={v}" for k, v in horizon_options.items())
    console.print(f"[dim]预测周期选项: {horizon_desc}[/]")
    label_horizon = int(_prompt("预测周期(天)", str(core["DEFAULT_LABEL_HORIZON"]),
                                WordCompleter(["1", "5", "10", "20"])))

    # 允许自定义日期
    custom = _prompt("自定义日期？(y/n)", "n")
    if custom.lower() == "y":
        dates["train_start"] = _prompt("训练开始", dates["train_start"])
        dates["train_end"] = _prompt("训练结束", dates["train_end"])
        dates["valid_start"] = _prompt("验证开始", dates["valid_start"])
        dates["valid_end"] = _prompt("验证结束", dates["valid_end"])
        dates["test_start"] = _prompt("测试开始", dates["test_start"])
        dates["test_end"] = _prompt("测试结束", dates["test_end"])

    horizon_label = horizon_options.get(label_horizon, f"{label_horizon}日")
    console.print(f"\n[cyan]市场={market}  模型={model_type}  因子={handler_type}  预测周期={horizon_label}[/]")
    confirm = _prompt("开始训练？(y/n)", "y")
    if confirm.lower() != "y":
        console.print("[yellow]已取消[/]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("初始化...", total=100)

        def cb(pct, msg):
            progress.update(task, completed=max(pct, 0), description=msg)

        result = core["train"](
            market=market,
            model_type=model_type,
            handler_type=handler_type,
            label_horizon=label_horizon,
            dates=dates,
            progress_callback=cb,
        )

    if result and result.get("success"):
        rid = result.get("recorder_id", "?")
        console.print(f"[bold green]✓ 训练完成！记录ID: {rid}[/]")
        # 自动进入评估
        _show_eval_result(rid)
    else:
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 训练失败: {msg}[/]")


# ---------- 4. 训练评估 ----------

def _show_eval_result(recorder_id):
    """显示指定 recorder 的评估结果（IC指标 + 因子重要性）"""
    core = _get_core()

    with console.status("[bold cyan]加载训练结果...[/]", spinner="dots"):
        result = core["get_train_result"](recorder_id)

    if not result or not result.get("success"):
        msg = result.get("message", "?") if result else "?"
        console.print(f"[bold red]✗ 获取评估数据失败: {msg}[/]")
        return

    # 显示模型参数
    params = result.get("params", {})
    label_horizon = int(params.get('label_horizon', 1))
    horizon_options = core["LABEL_HORIZON_OPTIONS"]
    horizon_desc = horizon_options.get(label_horizon, f"{label_horizon}日")
    console.print(f"\n[bold]模型: [cyan]{params.get('model_type', '?')}[/]  "
                  f"市场: [cyan]{params.get('market', '?')}[/]  "
                  f"因子: [cyan]{params.get('handler_type', '?')}[/]  "
                  f"预测周期: [cyan]{horizon_desc}[/][/]")
    console.print(f"[dim]训练: {params.get('train_start_date', '?')} ~ {params.get('train_end_date', '?')}  "
                  f"测试: {params.get('test_start_date', '?')} ~ {params.get('test_end_date', '?')}[/]\n")

    # 显示 IC 指标
    metrics = result.get("metrics", {})
    if metrics:
        mt = Table(title="模型评估指标", box=box.ROUNDED)
        mt.add_column("指标", style="bold")
        mt.add_column("值", justify="right")
        mt.add_column("评级", justify="center")

        def _grade(name, val):
            if val is None:
                return "[dim]-[/]"
            if "IC" == name or "Rank_IC" == name:
                if val < 0:
                    return "[bold red]⚠反向[/]"
                elif val > 0.08:
                    return "[bold green]优秀[/]"
                elif val > 0.05:
                    return "[green]良好[/]"
                elif val > 0.03:
                    return "[yellow]一般[/]"
                else:
                    return "[red]较弱[/]"
            elif "ICIR" in name:
                if val < 0:
                    return "[bold red]⚠反向[/]"
                elif val > 0.5:
                    return "[bold green]优秀[/]"
                elif val > 0.3:
                    return "[green]良好[/]"
                else:
                    return "[yellow]一般[/]"
            elif "precision" in name.lower():
                if val > 0.55:
                    return "[bold green]优秀[/]"
                elif val > 0.52:
                    return "[green]良好[/]"
                elif val > 0.5:
                    return "[yellow]一般[/]"
                else:
                    return "[red]不足[/]"
            return ""

        for key in ["IC", "ICIR", "Rank_IC", "Rank_ICIR", "Long_precision", "Short_precision"]:
            val = metrics.get(key)
            val_str = f"{val:.4f}" if val is not None else "-"
            mt.add_row(key, val_str, _grade(key, val))
        console.print(mt)
    else:
        console.print("[yellow]无 IC 指标数据[/]")

    # 显示因子重要性 Top 20
    fi = result.get("feature_importance")
    if fi:
        console.print()
        fi_table = Table(title="因子重要性 Top 20", box=box.ROUNDED)
        fi_table.add_column("#", style="dim", width=4)
        fi_table.add_column("因子名称", style="cyan")
        fi_table.add_column("重要性", justify="right")
        fi_table.add_column("占比", justify="right")
        fi_table.add_column("", width=20)

        total_imp = sum(item["importance"] for item in fi[:20])
        for i, item in enumerate(fi[:20], 1):
            pct = item["importance"] / total_imp * 100 if total_imp > 0 else 0
            bar_len = int(pct / 5)  # max ~20 chars
            bar = "█" * bar_len
            fi_table.add_row(
                str(i),
                item["name"],
                f"{item['importance']:.1f}",
                f"{pct:.1f}%",
                f"[green]{bar}[/]",
            )
        console.print(fi_table)
    else:
        console.print("\n[dim]无因子重要性数据（仅 LGB/XGB/CatBoost 支持）[/]")


def do_eval():
    """菜单入口：手动选择记录查看评估"""
    console.print("\n[bold]📈 训练评估[/]")
    console.print("[dim]查看模型 IC/ICIR 指标与因子重要性[/]\n")
    core = _get_core()

    recs = core["list_recorders"]()
    if not recs.get("success") or not recs.get("recorders"):
        console.print("[red]没有可用的训练记录，请先训练模型[/]")
        return

    _show_recorders_table(recs["recorders"], title="可用训练记录")

    recorder_id = _prompt("输入训练记录ID (可粘贴)")
    if not recorder_id:
        console.print("[yellow]已取消[/]")
        return

    _show_eval_result(recorder_id)


# ---------- 5. 回测 ----------

def do_backtest():
    console.print("\n[bold]📊 回测[/]")
    core = _get_core()

    recs = core["list_recorders"]()
    if not recs.get("success") or not recs.get("recorders"):
        console.print("[red]没有可用的训练记录，请先训练模型[/]")
        return

    _show_recorders_table(recs["recorders"], title="可用训练记录")

    recorder_id = _prompt("输入训练记录ID (可粘贴)")
    if not recorder_id:
        console.print("[yellow]已取消[/]")
        return

    market = _prompt("市场", core["DEFAULT_MARKET"], WordCompleter(MARKETS))
    topk = int(_prompt("TopK (持仓数)", "10"))
    n_drop = int(_prompt("N_drop (每日换仓数)", "1"))

    import datetime
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    default_start = core["_months_ago"](yesterday, 3).strftime("%Y-%m-%d")
    default_end = yesterday.strftime("%Y-%m-%d")
    start_date = _prompt("回测起始", default_start)
    end_date = _prompt("回测结束", default_end)

    console.print(f"\n[cyan]记录={recorder_id[:8]}...  市场={market}  TopK={topk}  "
                  f"日期={start_date}~{end_date}[/]")
    console.print("[dim]持仓天数将根据训练时的预测周期自动设置[/]")
    confirm = _prompt("开始回测？(y/n)", "y")
    if confirm.lower() != "y":
        console.print("[yellow]已取消[/]")
        return

    with console.status("[bold cyan]回测执行中...[/]", spinner="dots"):
        result = core["backtest"](
            recorder_id=recorder_id,
            market=market,
            start_date=start_date,
            end_date=end_date,
            topk=topk,
            n_drop=n_drop,
        )

    if result and result.get("success"):
        rid = result.get("recorder_id", "?")
        console.print(f"[bold green]✓ 回测完成！回测记录ID: {rid}[/]")

        # 显示回测结果摘要
        show = _prompt("查看回测结果？(y/n)", "y")
        if show.lower() == "y":
            _show_backtest_result(rid)
    else:
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 回测失败: {msg}[/]")


# ---------- 6. 预测 ----------

def do_predict():
    console.print("\n[bold]🔮 生成今日交易信号[/]")
    core = _get_core()

    recs = core["list_recorders"]()
    if not recs.get("success") or not recs.get("recorders"):
        console.print("[red]没有可用的训练记录，请先训练模型[/]")
        return

    _show_recorders_table(recs["recorders"], title="可用训练记录")

    recorder_id = _prompt("输入训练记录ID")
    if not recorder_id:
        console.print("[yellow]已取消[/]")
        return

    market = _prompt("市场", core["DEFAULT_MARKET"], WordCompleter(MARKETS))
    topk = int(_prompt("TopK", "10"))

    with console.status("[bold cyan]生成预测信号中...[/]", spinner="dots"):
        result = core["predict"](recorder_id=recorder_id, market=market, topk=topk)

    if result and result.get("success"):
        console.print(f"\n[bold green]✓ 预测完成 — {result['prediction_date']}[/]")
        console.print(f"[dim]市场: {result['market']}  模型: {result.get('model_type', '?')}  "
                      f"评估股票数: {result.get('total_stocks', '?')}[/]\n")

        # 买入推荐表
        table = Table(title="📈 买入推荐", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("评分", justify="right", style="green")

        for i, item in enumerate(result.get("buy_list", []), 1):
            table.add_row(
                str(i),
                item["stock_code"],
                item["stock_name"],
                f"{item['score']:.4f}",
            )
        console.print(table)
    else:
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 预测失败: {msg}[/]")


# ---------- 7/8. 查看记录 ----------

def _show_recorders_table(recorders, title="记录"):
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="cyan", max_width=36)
    table.add_column("名称", max_width=50)
    table.add_column("模型", style="yellow")
    table.add_column("市场")
    table.add_column("时间")

    for rec in recorders[:10]:  # 显示最新的 10 条
        params = rec.get("params", {})
        table.add_row(
            rec["id"],
            rec.get("name", "")[:50],
            params.get("model_type", "?"),
            params.get("market", params.get("backtest_market", "?")),
            str(rec.get("start_time", ""))[:19],
        )
    console.print(table)


def do_list_train():
    console.print("\n[bold]📋 训练记录[/]")
    core = _get_core()
    recs = core["list_recorders"]()
    if recs.get("success") and recs.get("recorders"):
        _show_recorders_table(recs["recorders"], title="训练记录")
    else:
        console.print("[yellow]暂无训练记录[/]")


def do_list_backtest():
    console.print("\n[bold]📋 回测记录[/]")
    core = _get_core()
    recs = core["list_backtest_recorders"]()
    if recs.get("success") and recs.get("recorders"):
        _show_recorders_table(recs["recorders"], title="回测记录")
        rid = _prompt("输入记录ID查看详情 (留空跳过)", "")
        if rid:
            _show_backtest_result(rid)
    else:
        console.print("[yellow]暂无回测记录[/]")


def _show_backtest_result(recorder_id):
    """显示回测结果摘要"""
    core = _get_core()
    with console.status("[bold cyan]加载回测结果...[/]"):
        result = core["get_backtest_result"](recorder_id)

    if not result or not result.get("success"):
        console.print(f"[red]获取失败: {result.get('message', '?') if result else '?'}[/]")
        return

    metrics = result.get("key_metrics", {})
    table = Table(title="回测结果", box=box.ROUNDED)
    table.add_column("指标", style="bold")
    table.add_column("值", justify="right")

    table.add_row("总收益率", f"[{'green' if metrics.get('total_return', 0) > 0 else 'red'}]{metrics.get('total_return', 0):.2f}%[/]")
    table.add_row("基准收益", f"{metrics.get('bench_return', 0):.2f}%")
    table.add_row("超额收益", f"[{'green' if metrics.get('excess_return', 0) > 0 else 'red'}]{metrics.get('excess_return', 0):.2f}%[/]")
    table.add_row("年化收益", f"{metrics.get('annualized_return', 0):.4f}")
    table.add_row("信息比率", f"{metrics.get('information_ratio', 0):.3f}")
    table.add_row("最大回撤", f"[red]{metrics.get('max_drawdown', 0):.4f}[/]")

    console.print(table)


# ---------- 9. 删除记录 ----------

def do_delete():
    console.print("\n[bold]🗑  删除记录[/]")
    core = _get_core()

    sub_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    sub_table.add_column(style="bold yellow", width=6)
    sub_table.add_column()
    sub_table.add_row("[1]", "删除单条训练记录")
    sub_table.add_row("[2]", "删除全部训练记录")
    sub_table.add_row("[3]", "删除单条回测记录")
    sub_table.add_row("[4]", "删除全部回测记录")
    console.print(sub_table)

    choice = _prompt("选择操作 (留空返回)", "")
    if not choice:
        return

    if choice == "1":
        recs = core["list_recorders"]()
        if not recs.get("success") or not recs.get("recorders"):
            console.print("[yellow]暂无训练记录[/]")
            return
        _show_recorders_table(recs["recorders"], title="训练记录")
        rid = _prompt("输入要删除的训练记录ID")
        if not rid:
            console.print("[yellow]已取消[/]")
            return
        confirm = _prompt(f"确认删除 {rid[:12]}...？(y/n)", "n")
        if confirm.lower() != "y":
            console.print("[yellow]已取消[/]")
            return
        result = core["delete_train_recorder"](rid)
        if result.get("success"):
            console.print(f"[bold green]✓ {result['message']}[/]")
        else:
            console.print(f"[bold red]✗ {result['message']}[/]")

    elif choice == "2":
        recs = core["list_recorders"]()
        count = len(recs.get("recorders", [])) if recs.get("success") else 0
        if count == 0:
            console.print("[yellow]暂无训练记录[/]")
            return
        confirm = _prompt(f"确认删除全部 {count} 条训练记录？此操作不可恢复！(yes/n)", "n")
        if confirm != "yes":
            console.print("[yellow]已取消（需输入 yes 确认）[/]")
            return
        with console.status("[bold cyan]删除中...[/]"):
            result = core["delete_all_train_recorders"]()
        if result.get("success"):
            console.print(f"[bold green]✓ {result['message']}[/]")
        else:
            console.print(f"[bold red]✗ {result['message']}[/]")

    elif choice == "3":
        recs = core["list_backtest_recorders"]()
        if not recs.get("success") or not recs.get("recorders"):
            console.print("[yellow]暂无回测记录[/]")
            return
        _show_recorders_table(recs["recorders"], title="回测记录")
        rid = _prompt("输入要删除的回测记录ID")
        if not rid:
            console.print("[yellow]已取消[/]")
            return
        confirm = _prompt(f"确认删除 {rid[:12]}...？(y/n)", "n")
        if confirm.lower() != "y":
            console.print("[yellow]已取消[/]")
            return
        result = core["delete_backtest_recorder"](rid)
        if result.get("success"):
            console.print(f"[bold green]✓ {result['message']}[/]")
        else:
            console.print(f"[bold red]✗ {result['message']}[/]")

    elif choice == "4":
        recs = core["list_backtest_recorders"]()
        count = len(recs.get("recorders", [])) if recs.get("success") else 0
        if count == 0:
            console.print("[yellow]暂无回测记录[/]")
            return
        confirm = _prompt(f"确认删除全部 {count} 条回测记录？此操作不可恢复！(yes/n)", "n")
        if confirm != "yes":
            console.print("[yellow]已取消（需输入 yes 确认）[/]")
            return
        with console.status("[bold cyan]删除中...[/]"):
            result = core["delete_all_backtest_recorders"]()
        if result.get("success"):
            console.print(f"[bold green]✓ {result['message']}[/]")
        else:
            console.print(f"[bold red]✗ {result['message']}[/]")
    else:
        console.print("[yellow]无效选项[/]")


# ---------- 10. 滚动训练 ----------

def do_train_rolling():
    console.print("\n[bold]🔄 滚动重训练（Rolling Retrain）[/]")
    console.print("[dim]参考 Qlib benchmarks_dynamic：每月重新训练模型以对抗概念漂移[/]")
    console.print("[dim]预期 IC 从 ~0.04 提升到 ~0.09+，训练时间较长，请耐心等待[/]\n")
    core = _get_core()
    dates = core["_default_dates"]()
    console.print(f"[dim]默认日期: 训练 {dates['train_start']}~{dates['train_end']}, "
                  f"测试 {dates['test_start']}~{dates['test_end']}[/]\n")

    market = _prompt("市场", core["DEFAULT_MARKET"], WordCompleter(MARKETS))
    model_type = _prompt("模型", core["DEFAULT_MODEL"], WordCompleter(MODELS))
    handler_type = _prompt("因子", core["DEFAULT_HANDLER"], WordCompleter(HANDLERS))
    label_horizon = int(_prompt("预测周期(天)", str(core["DEFAULT_LABEL_HORIZON"]),
                                WordCompleter(["1", "5", "10", "20"])))
    rolling_step = int(_prompt("滚动步长(交易日)", "20",
                               WordCompleter(["10", "20", "40", "60"])))

    custom = _prompt("自定义日期？(y/n)", "n")
    if custom.lower() == "y":
        dates["train_start"] = _prompt("训练开始", dates["train_start"])
        dates["train_end"] = _prompt("训练结束", dates["train_end"])
        dates["test_start"] = _prompt("测试开始", dates["test_start"])
        dates["test_end"] = _prompt("测试结束", dates["test_end"])

    horizon_options = core["LABEL_HORIZON_OPTIONS"]
    horizon_label = horizon_options.get(label_horizon, f"{label_horizon}日")
    console.print(f"\n[cyan]市场={market}  模型={model_type}  因子={handler_type}  "
                  f"预测周期={horizon_label}  滚动步长={rolling_step}日[/]")
    confirm = _prompt("开始滚动训练？(y/n)", "y")
    if confirm.lower() != "y":
        console.print("[yellow]已取消[/]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("初始化...", total=100)

        def cb(pct, msg):
            progress.update(task, completed=max(pct, 0), description=msg)

        result = core["train_rolling"](
            market=market,
            model_type=model_type,
            handler_type=handler_type,
            label_horizon=label_horizon,
            rolling_step=rolling_step,
            dates=dates,
            progress_callback=cb,
        )

    if result and result.get("success"):
        rid = result.get("recorder_id", "?")
        metrics = result.get("metrics", {})
        console.print(f"[bold green]✓ 滚动训练完成！最后一轮记录ID: {rid}[/]")
        if metrics:
            console.print(f"  [bold]整体合并指标:[/]  IC={metrics.get('IC', 0):.4f}  "
                          f"ICIR={metrics.get('ICIR', 0):.4f}  "
                          f"Rank_IC={metrics.get('Rank_IC', 0):.4f}  "
                          f"Rank_ICIR={metrics.get('Rank_ICIR', 0):.4f}")
        # 自动显示最后一轮的详细评估
        _show_eval_result(rid)
    else:
        msg = result.get("message", "未知错误") if result else "未知错误"
        console.print(f"[bold red]✗ 滚动训练失败: {msg}[/]")


# ---------- Main Loop ----------

def main():
    _header()

    while True:
        console.print()
        _menu()
        try:
            choice = _prompt("\n选择操作", "")
        except KeyboardInterrupt:
            console.print()
            break

        if choice == "1":
            try:
                do_update_data()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "2":
            try:
                do_predict()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "3":
            try:
                do_preview_data()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "4":
            try:
                do_train()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "5":
            try:
                do_train_rolling()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "6":
            try:
                do_eval()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "7":
            try:
                do_backtest()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "8":
            try:
                do_list_train()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "9":
            try:
                do_list_backtest()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice == "10":
            try:
                do_delete()
            except KeyboardInterrupt:
                console.print("\n[yellow]已取消[/]")
        elif choice.lower() == "q":
            break
        elif choice == "":
            continue
        else:
            console.print("[yellow]无效选项，请输入 1-10 或 q[/]")

    console.print("\n[dim]再见 👋[/]")


if __name__ == "__main__":
    main()
