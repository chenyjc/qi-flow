"""
EnhancedTopkDropoutStrategy 单元测试
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockAccount:
    """模拟交易账户"""
    def __init__(self, position):
        self.current_position = position


class MockCommonInfra:
    """模拟 CommonInfrastructure"""
    def __init__(self, exchange, position):
        self.trade_exchange = exchange
        self.trade_account = MockAccount(position)

    def get(self, name):
        return getattr(self, name, None)


class MockLevelInfra:
    """模拟 LevelInfrastructure"""
    def __init__(self, calendar, exchange, position):
        self.trade_calendar = calendar
        self.common_infra = MockCommonInfra(exchange, position)

    def get(self, name):
        return getattr(self, name, None)


class MockTradeCalendar:
    """模拟交易日历"""
    def __init__(self):
        self.current_step = 0
        self.trade_dates = pd.date_range(start='2024-01-01', periods=20, freq='B')
        self.freq_str = "day"

    def get_trade_step(self):
        return self.current_step

    def get_step_time(self, step=None, shift=0):
        if step is None:
            step = self.current_step
        idx = min(step + shift, len(self.trade_dates) - 1)
        return self.trade_dates[idx], self.trade_dates[idx]

    def get_freq(self):
        return pd.Timedelta(days=1)

    def advance(self):
        self.current_step += 1


class MockPosition:
    """模拟持仓"""
    def __init__(self):
        self.stocks = {}
        self.cash = 1000000

    def get_stock_list(self):
        return list(self.stocks.keys())

    def get_stock_amount(self, code):
        return self.stocks.get(code, 0)

    def get_cash(self):
        return self.cash

    def get_stock_count(self, code, bar=None):
        return 10  # 模拟持仓10天


class MockTradeExchange:
    """模拟交易所"""
    def __init__(self):
        self.prices = {}

    def set_price(self, stock_id, price):
        self.prices[stock_id] = price

    def get_deal_price(self, stock_id, **kwargs):
        return self.prices.get(stock_id, 10.0)

    def is_stock_tradable(self, **kwargs):
        return True

    def check_order(self, order):
        return True

    def deal_order(self, order, position):
        price = self.get_deal_price(order.stock_id)
        return price * order.amount, 0, price

    def get_factor(self, **kwargs):
        return 1.0

    def round_amount_by_trade_unit(self, amount, factor):
        return int(amount / 100) * 100


class TestEnhancedTopkDropoutStrategy(unittest.TestCase):
    """测试 EnhancedTopkDropoutStrategy"""

    def setUp(self):
        """设置测试环境"""
        self.calendar = MockTradeCalendar()
        self.position = MockPosition()
        self.exchange = MockTradeExchange()

        # 创建信号数据 (MultiIndex: date, instrument)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
        instruments = [f"SH{i:06d}" for i in range(1, 11)]

        # 创建笛卡尔积（所有日期和股票的组合）
        index = pd.MultiIndex.from_product([dates, instruments], names=['date', 'instrument'])
        values = np.random.rand(len(index))

        self.signal = pd.Series(values, index=index)

        # 设置一些测试股票价格
        for i in range(1, 11):
            self.exchange.set_price(f"SH{i:06d}", 10.0 + i * 0.5)

    def test_init(self):
        """测试初始化"""
        from backend.strategy.enhanced_topk import EnhancedTopkDropoutStrategy

        strategy = EnhancedTopkDropoutStrategy(
            hold_days=3,
            stop_loss=0.05,
            signal=self.signal,
            topk=5,
            n_drop=1
        )
        self.assertEqual(strategy.hold_days, 3)
        self.assertEqual(strategy.stop_loss, 0.05)
        self.assertEqual(strategy.position_info, {})
        self.assertEqual(strategy.trade_counter, 0)

    def test_hold_days_logic(self):
        """测试持仓周期逻辑"""
        from backend.strategy.enhanced_topk import EnhancedTopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO

        strategy = EnhancedTopkDropoutStrategy(
            hold_days=3,
            stop_loss=0.0,
            signal=self.signal,
            topk=5,
            n_drop=1,
            only_tradable=False
        )
        # 创建 mock level infrastructure
        level_infra = MockLevelInfra(self.calendar, self.exchange, self.position)

        # 设置 mock 依赖
        strategy.reset_level_infra(level_infra)
        strategy.common_infra = level_infra.common_infra

        rebalanced_days = []

        # 模拟5个交易日
        for day in range(5):
            print(f"\n=== Day {day + 1} (step={self.calendar.current_step}) ===")

            # 生成交易决策
            decision = strategy.generate_trade_decision()

            # 检查是否返回 TradeDecisionWO
            self.assertIsInstance(decision, TradeDecisionWO)

            # 检查是否有 get_range_limit 方法
            self.assertTrue(hasattr(decision, 'get_range_limit'))

            # 获取订单
            orders = decision.get_decision()
            print(f"Orders count: {len(orders)}")

            # 判断是否应该调仓
            if strategy.trade_counter % strategy.hold_days == 0 and strategy.trade_counter > 0:
                print(f"Rebalance day! (counter={strategy.trade_counter})")
                rebalanced_days.append(day + 1)
            else:
                print(f"Skip rebalance (counter={strategy.trade_counter})")

            self.calendar.advance()

        print(f"\nRebalanced on days: {rebalanced_days}")
        print(f"Final trade_counter: {strategy.trade_counter}")

        # 验证调仓逻辑：counter = 1,2,3,4,5；hold_days=3
        # counter 3 应该调仓（3 % 3 == 0）
        self.assertIn(3, rebalanced_days, "第3天应该调仓")

    def test_stop_loss_logic(self):
        """测试止损逻辑"""
        from backend.strategy.enhanced_topk import EnhancedTopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO, OrderDir

        strategy = EnhancedTopkDropoutStrategy(
            hold_days=1,
            stop_loss=0.05,  # 5%止损
            signal=self.signal,
            topk=5,
            n_drop=1,
            only_tradable=False
        )
        # 创建 mock level infrastructure
        level_infra = MockLevelInfra(self.calendar, self.exchange, self.position)

        # 设置 mock 依赖
        strategy.reset_level_infra(level_infra)
        strategy.common_infra = level_infra.common_infra

        # 手动添加持仓信息
        strategy.position_info = {
            "SH000001": {
                'buy_date': pd.Timestamp('2024-01-01'),
                'buy_price': 10.0,
                'hold_days': 0
            }
        }

        # 模拟持仓
        self.position.stocks["SH000001"] = 1000

        # Mock _get_stock_price 方法
        def mock_get_price(stock_code, date):
            return self.exchange.get_deal_price(stock_code)

        strategy._get_stock_price = mock_get_price

        print("\n=== Testing Stop Loss ===")
        print(f"Buy price: 10.0")
        print(f"Current price: 9.4 (loss = 6%)")
        print(f"Stop loss threshold: 5%")

        # 生成交易决策
        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()

        # 检查是否触发止损
        stop_loss_triggered = False
        for order in orders:
            print(f"Order: stock_id={order.stock_id}, direction={order.direction}")
            if order.stock_id == "SH000001" and order.direction == OrderDir.SELL:
                stop_loss_triggered = True
                print(f"Stop loss triggered!")

        self.assertTrue(stop_loss_triggered, "止损应该被触发")

    def test_stop_loss_not_triggered(self):
        """测试未触发止损的情况"""
        from backend.strategy.enhanced_topk import EnhancedTopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO

        strategy = EnhancedTopkDropoutStrategy(
            hold_days=1,
            stop_loss=0.05,  # 5%止损
            signal=self.signal,
            topk=5,
            n_drop=1,
            only_tradable=False
        )
        # 创建 mock level infrastructure
        level_infra = MockLevelInfra(self.calendar, self.exchange, self.position)

        # 设置 mock 依赖
        strategy.reset_level_infra(level_infra)
        strategy.common_infra = level_infra.common_infra

        # 手动添加持仓信息
        strategy.position_info = {
            "SH000001": {
                'buy_date': pd.Timestamp('2024-01-01'),
                'buy_price': 10.0,
                'hold_days': 0
            }
        }

        self.position.stocks["SH000001"] = 1000

        # 设置当前价格，不触发止损（亏损3%）
        self.exchange.set_price("SH000001", 9.7)  # 亏损 3%

        print("\n=== Testing No Stop Loss ===")
        print(f"Buy price: 10.0")
        print(f"Current price: 9.7 (loss = 3%)")
        print(f"Stop loss threshold: 5%")

        # 生成交易决策
        decision = strategy.generate_trade_decision()
        orders = decision.get_decision()

        # 检查是否没有止损订单（针对 SH000001）
        for order in orders:
            print(f"Order: stock_id={order.stock_id}, direction={order.direction}")
            self.assertNotEqual(order.stock_id, "SH000001",
                              "不应该有 SH000001 的止损订单")

    def test_return_type(self):
        """测试返回值类型"""
        from backend.strategy.enhanced_topk import EnhancedTopkDropoutStrategy
        from qlib.backtest.decision import TradeDecisionWO

        strategy = EnhancedTopkDropoutStrategy(
            hold_days=1,
            stop_loss=0.0,
            signal=self.signal,
            topk=5,
            n_drop=1,
            only_tradable=False
        )
        # 创建 mock level infrastructure
        level_infra = MockLevelInfra(self.calendar, self.exchange, self.position)

        # 设置 mock 依赖
        strategy.reset_level_infra(level_infra)
        strategy.common_infra = level_infra.common_infra

        # 测试返回类型
        decision = strategy.generate_trade_decision()
        self.assertIsInstance(decision, TradeDecisionWO)

        # 检查是否有 get_range_limit 方法
        self.assertTrue(hasattr(decision, 'get_range_limit'))


if __name__ == '__main__':
    # 设置详细输出
    unittest.main(verbosity=2)

