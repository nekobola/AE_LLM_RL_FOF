"""调查 ClickHouse 中的 ETF 数据和宏观数据"""
from quantchdb import ClickHouseDatabase

db = ClickHouseDatabase(config={
    'host': '10.13.66.5',
    'port': 20108,
    'user': 'cufel_arena_etf_reader',
    'password': 'cufel_arena_etf_404',
    'database': 'etf',
}, terminal_log=False, file_log=False)
print('[OK] 成功连接 ClickHouse')

# ========== 问题1: ETF 数据 - 160217 替代代码 ==========
print('\n' + '='*60)
print('问题1: 160217 替代代码查询')
print('='*60)

# 查找以 16 开头的代码
sql = """
    SELECT code, count() as cnt, min(date) as min_date, max(date) as max_date
    FROM etf.etf_day 
    WHERE code LIKE '16%'
    GROUP BY code
    ORDER BY cnt DESC
    LIMIT 20
"""
result = db.fetch(sql)
print('\n以 16 开头的 ETF 代码:')
if result is not None and len(result) > 0:
    for _, row in result.iterrows():
        print(f"  {row['code']}: {row['cnt']}条记录, {row['min_date']} ~ {row['max_date']}")

# 也检查 fetcher.py 中提到的所有代码
print('\n--- 检查 track_b fetcher 中提到的所有代码 ---')
known_codes = ['510300', '512850', '511010', '518880', '160217']
for code in known_codes:
    sql = f"""
        SELECT count() as cnt, min(date) as min_date, max(date) as max_date
        FROM etf.etf_day 
        WHERE code = '{code}'
    """
    result = db.fetch(sql)
    if result is not None and len(result) > 0:
        row = result.iloc[0]
        status = '存在' if row['cnt'] > 0 else '不存在'
        print(f"  {code}: {status} ({row['cnt']}条, {row['min_date']} ~ {row['max_date']})")

# ========== 问题2: 宏观数据 ==========
print('\n' + '='*60)
print('问题2: 宏观特征数据查询')
print('='*60)

# 检查所有数据库
dbs = db.fetch('SHOW DATABASES')
print('\n所有数据库:')
if dbs is not None:
    for _, row in dbs.iterrows():
        print(f"  {row['name']}")

# 对每个数据库检查表
print('\n--- 检查各数据库的表 ---')
if dbs is not None:
    for _, row in dbs.iterrows():
        db_name = row['name']
        try:
            tables_in_db = db.fetch(f'SHOW TABLES FROM `{db_name}`')
            if tables_in_db is not None and len(tables_in_db) > 0:
                print(f'\n数据库: {db_name}')
                for _, t in tables_in_db.iterrows():
                    table_name = t['name']
                    print(f'  - {table_name}')
                    # 检查表结构
                    try:
                        cols = db.fetch(f'DESCRIBE TABLE `{db_name}`.`{table_name}`')
                        if cols is not None:
                            col_names = cols['name'].tolist()
                            print(f'    列: {col_names}')
                    except:
                        pass
        except Exception as e:
            print(f'数据库 {db_name} 无法访问: {e}')
