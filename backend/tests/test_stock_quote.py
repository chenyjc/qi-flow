"""
股票行情 API 单元测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qlib_service import QlibService

def test_get_stock_quote():
    """测试获取股票行情数据"""
    qlib_service = QlibService()
    
    # 测试不同的股票代码格式
    test_cases = [
        {
            "code": "SH600519",
            "name": "贵州茅台"
        },
        {
            "code": "SZ002594",
            "name": "比亚迪"
        },
        {
            "code": "SH600000",
            "name": "浦发银行"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试股票：{test_case['name']} ({test_case['code']})")
        print("-" * 50)
        
        result = qlib_service.get_stock_quote(
            stock_code=test_case['code'],
            start_date="2026-03-01",
            end_date="2026-04-21"
        )
        
        if result['success']:
            print(f"✓ 获取数据成功")
            print(f"  数据条数：{len(result['data'])}")
            
            if len(result['data']) > 0:
                # 打印第一条和最后一条数据
                first = result['data'][0]
                last = result['data'][-1]
                print(f"\n  第一条数据:")
                print(f"    日期：{first['date']}")
                print(f"    开盘：{first['open']:.2f}")
                print(f"    收盘：{first['close']:.2f}")
                print(f"    最高：{first['high']:.2f}")
                print(f"    最低：{first['low']:.2f}")
                print(f"    成交量：{first['volume']:,}")
                
                if len(result['data']) > 1:
                    print(f"\n  最后一条数据:")
                    print(f"    日期：{last['date']}")
                    print(f"    开盘：{last['open']:.2f}")
                    print(f"    收盘：{last['close']:.2f}")
                    print(f"    最高：{last['high']:.2f}")
                    print(f"    最低：{last['low']:.2f}")
                    print(f"    成交量：{last['volume']:,}")
        else:
            print(f"✗ 获取数据失败")
            print(f"  错误信息：{result['message']}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("股票行情 API 单元测试")
    print("=" * 50)
    test_get_stock_quote()
    print("\n测试完成！")
