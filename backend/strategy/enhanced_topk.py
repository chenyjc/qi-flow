"""
Enhanced Topk Dropout Strategy with Hold Days and Stop Loss
带有持仓周期和止损功能的增强版 Topk 策略
"""

import pandas as pd
import numpy as np
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class EnhancedTopkDropoutStrategy(TopkDropoutStrategy):
    """
    增强版 Topk Dropout 策略

    新增功能：
    1. hold_days: 持仓周期，每隔 N 天检查调仓（默认1=每天调仓）
    2. stop_loss: 止损比例，亏损超过设定值强制平仓（默认0=不止损）

    Parameters
    ----------
    hold_days : int
        持仓周期天数，每隔 N 天才考虑调仓
    stop_loss : float
        止损比例，如 0.05 表示亏损 5% 止损
    """

    def __init__(self, hold_days=1, stop_loss=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hold_days = hold_days
        self.stop_loss = stop_loss

        # 追踪持仓信息
        self.position_info = {}  # {stock_code: {'buy_date': date, 'buy_price': price, 'hold_days': days}}
        self.last_trade_date = None
        self.trade_counter = 0  # 交易日计数器

    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策
        """
        # 获取当前交易日
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_date = trade_start_time
        if current_date is None:
            return TradeDecisionWO([], self)

        # 初始化或更新交易日计数
        if self.last_trade_date != current_date:
            self.trade_counter += 1
            self.last_trade_date = current_date

            # 更新持仓天数
            for stock_code in self.position_info:
                self.position_info[stock_code]['hold_days'] += 1

        # 获取当前持仓
        current_position = self.trade_position

        # 检查止损（优先处理）
        stop_loss_orders = self._check_stop_loss(current_position, trade_start_time, trade_end_time)
        if stop_loss_orders:
            # 如果有止损订单，执行止损并返回
            for order in stop_loss_orders:
                if order.stock_id in self.position_info:
                    del self.position_info[order.stock_id]
            return TradeDecisionWO(stop_loss_orders, self)

        # 检查是否到达调仓日
        if self.trade_counter % self.hold_days != 0:
            # 非调仓日，不调仓
            return TradeDecisionWO([], self)

        # 调用父类方法生成正常调仓订单
        decision = super().generate_trade_decision(execute_result)

        # 父类返回的是 TradeDecisionWO 对象，获取其中的订单
        if isinstance(decision, TradeDecisionWO):
            orders = decision.get_decision()
        else:
            orders = decision

        # 更新持仓信息（记录新买入的股票）
        for order in orders:
            if order.direction == OrderDir.BUY:
                # 买入订单
                if order.stock_id not in self.position_info:
                    # 获取买入价格（使用当前收盘价）
                    try:
                        price = self._get_stock_price(order.stock_id, trade_start_time)
                        self.position_info[order.stock_id] = {
                            'buy_date': current_date,
                            'buy_price': price,
                            'hold_days': 0
                        }
                    except:
                        pass
            elif order.direction == OrderDir.SELL:
                # 卖出订单，从持仓信息中移除
                if order.stock_id in self.position_info:
                    del self.position_info[order.stock_id]

        return TradeDecisionWO(orders, self)

    def _check_stop_loss(self, position, trade_start_time, trade_end_time):
        """
        检查止损条件

        Parameters
        ----------
        position : Position
            当前持仓
        trade_start_time : pd.Timestamp
            交易开始时间
        trade_end_time : pd.Timestamp
            交易结束时间

        Returns
        -------
        list[Order]
            需要止损的卖出订单列表
        """
        if self.stop_loss <= 0:
            return []

        stop_loss_orders = []

        for stock_code, info in list(self.position_info.items()):
            try:
                # 获取当前价格
                current_price = self._get_stock_price(stock_code, trade_start_time)
                buy_price = info['buy_price']

                if buy_price > 0:
                    # 计算亏损比例
                    loss_ratio = (current_price - buy_price) / buy_price

                    # 检查是否触发止损
                    if loss_ratio <= -self.stop_loss:
                        # 触发止损，创建卖出订单
                        sell_amount = position.get_stock_amount(code=stock_code)
                        order = Order(
                            stock_id=stock_code,
                            amount=sell_amount,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=OrderDir.SELL
                        )
                        stop_loss_orders.append(order)
                        print(f"[止损] {stock_code} 买入价: {buy_price:.2f}, 当前价: {current_price:.2f}, 亏损: {loss_ratio*100:.2f}%")
            except Exception as e:
                print(f"止损检查失败 {stock_code}: {e}")
                continue

        return stop_loss_orders

    def _get_stock_price(self, stock_code, date):
        """
        获取指定股票在指定日期的收盘价
        """
        from qlib.data import D

        try:
            # 获取当天数据
            df = D.features(
                instruments=[stock_code],
                fields=['$close'],
                start_time=date,
                end_time=date
            )

            if not df.empty:
                return float(df.iloc[0]['$close'])
            else:
                raise ValueError(f"No price data for {stock_code} on {date}")
        except Exception as e:
            raise ValueError(f"Failed to get price for {stock_code}: {e}")
