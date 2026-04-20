"""
[Deprecated] 该服务已不再维护
请使用 QlibService 中的相关功能

股票数据服务模块
提供股票数据获取、因子计算和回测功能
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockService:
    def __init__(self):
        pass
    
    def get_csi300_stocks(self):
        """获取沪深300成分股"""
        result = {
            "methods": [],
            "stocks": []
        }
        
        # 方法1: 尝试使用index_stock_info
        try:
            csi300_stocks = ak.index_stock_info(symbol="000300.SH")
            stocks = []
            for _, row in csi300_stocks.iterrows():
                stocks.append({
                    "code": row.get('成分券代码', row.get('code', '')),
                    "name": row.get('成分券名称', row.get('name', ''))
                })
            result["methods"].append({"name": "index_stock_info", "success": True})
            result["stocks"] = stocks[:10]  # 只返回前10只
        except Exception as e:
            result["methods"].append({"name": "index_stock_info", "success": False, "error": str(e)})
        
        # 方法2: 尝试使用stock_zh_a_spot获取所有A股
        try:
            all_stocks = ak.stock_zh_a_spot()
            result["methods"].append({"name": "stock_zh_a_spot", "success": True, "count": len(all_stocks)})
        except Exception as e:
            result["methods"].append({"name": "stock_zh_a_spot", "success": False, "error": str(e)})
        
        # 方法3: 手动定义一些沪深300成分股作为示例
        if not result["stocks"]:
            csi300_sample = [
                {"code": "600519", "name": "贵州茅台"},
                {"code": "601318", "name": "中国平安"},
                {"code": "600036", "name": "招商银行"},
                {"code": "601888", "name": "中国中免"},
                {"code": "600276", "name": "恒瑞医药"},
                {"code": "601899", "name": "紫金矿业"},
                {"code": "601628", "name": "中国人寿"},
                {"code": "600887", "name": "伊利股份"},
                {"code": "601398", "name": "工商银行"},
                {"code": "600031", "name": "三一重工"}
            ]
            result["stocks"] = csi300_sample
            result["methods"].append({"name": "manual_sample", "success": True})
        
        return result
    
    def get_stock_history(self, symbol, start_date, end_date):
        """获取股票历史行情数据"""
        try:
            # 获取股票历史行情数据
            stock_data = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust='qfq')
            
            # 数据预处理
            stock_data['date'] = pd.to_datetime(stock_data['日期'])
            stock_data.set_index('date', inplace=True)
            stock_data = stock_data.sort_index()
            
            # 转换为前端可用格式
            data = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "data": {
                    "dates": stock_data.index.strftime('%Y-%m-%d').tolist(),
                    "open": stock_data['开盘'].tolist(),
                    "high": stock_data['最高'].tolist(),
                    "low": stock_data['最低'].tolist(),
                    "close": stock_data['收盘'].tolist(),
                    "volume": stock_data['成交量'].tolist()
                }
            }
            
            return data
        except Exception as e:
            raise Exception(f"获取股票历史数据失败: {str(e)}")
    
    def get_stock_factors(self, symbol, start_date, end_date):
        """获取股票因子数据"""
        try:
            # 获取股票历史数据
            stock_data = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust='qfq')
            stock_data['date'] = pd.to_datetime(stock_data['日期'])
            stock_data.set_index('date', inplace=True)
            stock_data = stock_data.sort_index()
            
            df = stock_data.copy()
            
            # 1. 收益率因子
            df['return'] = df['收盘'].pct_change()
            
            # 2. 波动率因子（20日）
            df['volatility'] = df['return'].rolling(20).std() * np.sqrt(252)
            
            # 3. MACD因子
            exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
            exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 4. RSI因子（14日）
            delta = df['收盘'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 5. 移动平均线因子
            df['ma5'] = df['收盘'].rolling(5).mean()
            df['ma10'] = df['收盘'].rolling(10).mean()
            df['ma20'] = df['收盘'].rolling(20).mean()
            df['ma60'] = df['收盘'].rolling(60).mean()
            
            # 6. 成交量因子
            df['volume_ma5'] = df['成交量'].rolling(5).mean()
            df['volume_ma20'] = df['成交量'].rolling(20).mean()
            df['volume_ratio'] = df['成交量'] / df['volume_ma20']
            
            # 转换为前端可用格式
            factors = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "data": {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    "return": df['return'].fillna(0).tolist(),
                    "volatility": df['volatility'].fillna(0).tolist(),
                    "macd": df['macd'].fillna(0).tolist(),
                    "macd_signal": df['macd_signal'].fillna(0).tolist(),
                    "macd_hist": df['macd_hist'].fillna(0).tolist(),
                    "rsi": df['rsi'].fillna(0).tolist(),
                    "ma5": df['ma5'].fillna(0).tolist(),
                    "ma10": df['ma10'].fillna(0).tolist(),
                    "ma20": df['ma20'].fillna(0).tolist(),
                    "ma60": df['ma60'].fillna(0).tolist(),
                    "volume_ratio": df['volume_ratio'].fillna(0).tolist()
                }
            }
            
            return factors
        except Exception as e:
            raise Exception(f"计算因子数据失败: {str(e)}")
    
    def backtest_strategy(self, symbol, start_date, end_date):
        """策略回测"""
        try:
            # 获取股票历史数据
            stock_data = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust='qfq')
            stock_data['date'] = pd.to_datetime(stock_data['日期'])
            stock_data.set_index('date', inplace=True)
            stock_data = stock_data.sort_index()
            
            df = stock_data.copy()
            
            # 计算MACD
            exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
            exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # 生成信号
            df['signal'] = 0
            df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
            df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
            
            # 计算策略收益率
            df['return'] = df['收盘'].pct_change()
            df['strategy_return'] = df['signal'].shift(1) * df['return']
            
            # 计算累计收益率
            df['cumulative_return'] = (1 + df['return']).cumprod()
            df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
            
            # 计算性能指标
            total_days = len(df)
            benchmark_return = (df['cumulative_return'].iloc[-1] - 1) * 100
            strategy_return = (df['cumulative_strategy_return'].iloc[-1] - 1) * 100
            
            # 转换为前端可用格式
            backtest_result = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "performance": {
                    "benchmark_return": round(benchmark_return, 2),
                    "strategy_return": round(strategy_return, 2),
                    "outperformance": round(strategy_return - benchmark_return, 2),
                    "total_days": total_days
                },
                "data": {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    "benchmark": df['cumulative_return'].tolist(),
                    "strategy": df['cumulative_strategy_return'].tolist(),
                    "signal": df['signal'].tolist()
                }
            }
            
            return backtest_result
        except Exception as e:
            raise Exception(f"策略回测失败: {str(e)}")
    
    def batch_analysis(self, symbols, start_date, end_date):
        """批量分析股票"""
        try:
            results = []
            
            for symbol in symbols:
                try:
                    # 获取股票数据
                    stock_data = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust='qfq')
                    stock_data['date'] = pd.to_datetime(stock_data['日期'])
                    stock_data.set_index('date', inplace=True)
                    stock_data = stock_data.sort_index()
                    
                    # 计算收益率
                    stock_data['return'] = stock_data['收盘'].pct_change()
                    total_return = (stock_data['收盘'].iloc[-1] / stock_data['收盘'].iloc[0] - 1) * 100
                    
                    # 计算波动率
                    volatility = stock_data['return'].std() * np.sqrt(252) * 100
                    
                    results.append({
                        "symbol": symbol,
                        "total_return": round(total_return, 2),
                        "volatility": round(volatility, 2),
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "symbol": symbol,
                        "total_return": 0,
                        "volatility": 0,
                        "status": "error",
                        "error": str(e)
                    })
            
            batch_result = {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "results": results
            }
            
            return batch_result
        except Exception as e:
            raise Exception(f"批量分析失败: {str(e)}")
