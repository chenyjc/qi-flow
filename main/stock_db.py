#!/usr/bin/env python3
"""
股票信息数据库管理脚本
用于初始化、更新和查询A股股票信息数据库
"""

import sqlite3
import os
import datetime

# 定义数据库文件路径
DB_FILE = os.path.join(os.path.dirname(__file__), 'stock_info.db')

# 初始化股票信息数据库
def init_stock_db():
    """初始化股票信息数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 创建股票信息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_info (
        code TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# 更新股票信息数据库
def update_stock_db():
    """更新股票信息数据库，从akshare获取最新数据"""
    try:
        import akshare as ak
        # 获取A股所有股票信息
        stock_info_df = ak.stock_info_a_code_name()
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 获取当前时间
        updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 清空旧数据并插入新数据
        cursor.execute('DELETE FROM stock_info')
        
        # 插入新数据
        for _, row in stock_info_df.iterrows():
            code = row['code']
            name = row['name']
            cursor.execute('INSERT INTO stock_info (code, name, updated_at) VALUES (?, ?, ?)', 
                          (code, name, updated_at))
        
        conn.commit()
        conn.close()
        print("股票信息数据库更新成功！")
        return True
    except Exception as e:
        print(f"更新股票数据库失败: {e}")
        return False

# 从数据库获取股票名称
def get_stock_names_from_db(stock_keys):
    """从数据库获取股票名称映射"""
    stock_names = {}
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 获取所有股票信息
        cursor.execute('SELECT code, name FROM stock_info')
        stock_data = cursor.fetchall()
        
        conn.close()
        
        # 创建股票代码到名称的映射，支持带前缀和不带前缀的代码
        for code, name in stock_data:
            # 保存不带前缀的代码映射
            stock_names[code] = name
            # 保存带SH前缀的代码映射
            stock_names[f"SH{code}"] = name
            # 保存带SZ前缀的代码映射
            stock_names[f"SZ{code}"] = name
        
        return stock_names, True, None
    except Exception as e:
        print(f"从数据库获取股票名称失败: {e}")
        return stock_names, False, e

# 主函数，用于独立运行脚本
if __name__ == "__main__":
    print("股票信息数据库管理脚本")
    print("1. 初始化数据库")
    print("2. 更新数据库")
    print("3. 退出")
    
    choice = input("请选择操作 (1-3): ")
    
    if choice == "1":
        init_stock_db()
        print("数据库初始化成功！")
    elif choice == "2":
        if update_stock_db():
            print("数据库更新成功！")
        else:
            print("数据库更新失败！")
    elif choice == "3":
        print("退出脚本")
    else:
        print("无效选择")
