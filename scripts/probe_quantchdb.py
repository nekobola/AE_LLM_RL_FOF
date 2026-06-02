"""
数据探测脚本 - 探测 ClickHouse (quantchdb) 自有数据库

功能：
1. 连接 ClickHouse 数据库
2. 探测所有数据库和表
3. 获取每个表的时间范围
4. 获取每个表的所有字段信息
5. 输出完整的数据字典

用法：
    python scripts/probe_quantchdb.py [--host HOST] [--port PORT] [--user USER] [--password PASSWORD] [--database DATABASE]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 环境变量
try:
    from dotenv import load_dotenv
    _env_path = PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv 未安装时静默跳过

# ── CLI ─────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="探测 ClickHouse (quantchdb) 自有数据库结构和时间范围"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("CHDB_HOST", "localhost"),
        help="ClickHouse 主机 (CHDB_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CHDB_PORT", "9000")),
        help="ClickHouse 端口 (CHDB_PORT)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("CHDB_USER", "default"),
        help="ClickHouse 用户 (CHDB_USER)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("CHDB_PASSWORD", ""),
        help="ClickHouse 密码 (CHDB_PASSWORD)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("CHDB_DATABASE", "etf"),
        help="默认数据库 (CHDB_DATABASE)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认打印到 stdout)",
    )
    parser.add_argument(
        "--skip-tables",
        type=str,
        nargs="+",
        default=[],
        help="跳过的表名 (部分匹配)",
    )
    return parser.parse_args()


# ── 连接 ───────────────────────────────────────────────────────────────────────


def connect(db_config: Dict[str, Any]):
    """连接 ClickHouse"""
    try:
        from quantchdb import ClickHouseDatabase
    except ImportError:
        raise ImportError(
            "quantchdb 未安装: pip install quantchdb==0.1.11\n"
            "或联系数据库管理员获取连接信息"
        )

    try:
        db = ClickHouseDatabase(
            config={
                "host": db_config["host"],
                "port": db_config["port"],
                "user": db_config["user"],
                "password": db_config["password"],
                "database": db_config["database"],
            },
            terminal_log=False,
            file_log=False,
        )
        print(f"[OK] 成功连接 ClickHouse: {db_config['host']}:{db_config['port']}")
        return db
    except Exception as e:
        raise ConnectionError(f"连接 ClickHouse 失败: {e}")


# ── 探测数据库和表 ─────────────────────────────────────────────────────────────


def list_databases(db) -> List[str]:
    """列出所有数据库"""
    try:
        result = db.fetch("SHOW DATABASES")
        if result is None or result.empty:
            return []
        return result["name"].tolist()
    except Exception as e:
        print(f"[WARN]  获取数据库列表失败: {e}")
        return []


def list_tables(db, database: str) -> List[str]:
    """列出指定数据库的所有表"""
    try:
        result = db.fetch(f"SHOW TABLES FROM `{database}`")
        if result is None or result.empty:
            return []
        return result["name"].tolist()
    except Exception as e:
        print(f"[WARN]  获取表列表失败: {e}")
        return []


def get_table_columns(db, database: str, table: str) -> pd.DataFrame:
    """获取表的字段信息 (列名, 类型, 默认值, Nullable)"""
    try:
        query = f"""
            SELECT 
                name         AS column_name,
                type         AS data_type,
                default_kind AS default_kind,
                default_expression AS default_value,
                is_in_primary_key  AS is_primary_key,
                comment      AS comment
            FROM system.columns
            WHERE database = '{database}'
              AND table = '{table}'
            ORDER BY position
        """
        result = db.fetch(query)
        return result if result is not None else pd.DataFrame()
    except Exception as e:
        print(f"[WARN]  获取字段信息失败 ({database}.{table}): {e}")
        return pd.DataFrame()


def get_table_row_count(db, database: str, table: str) -> int:
    """获取表的行数"""
    try:
        result = db.fetch(f"SELECT count() AS cnt FROM `{database}`.`{table}`")
        if result is not None and not result.empty:
            return int(result["cnt"].iloc[0])
        return 0
    except Exception:
        return -1  # 无法获取


def get_table_time_range(
    db, database: str, table: str, date_col: str = "date"
) -> Tuple[Optional[str], Optional[str], str]:
    """
    获取表的时间范围 (最小日期, 最大日期)
    
    Returns
    -------
    Tuple[min_date, max_date, status]
        status: 'ok' | 'no_date_col' | 'error'
    """
    try:
        query = f"""
            SELECT 
                MIN({date_col}) AS min_date,
                MAX({date_col}) AS max_date
            FROM `{database}`.`{table}`
        """
        result = db.fetch(query)
        if result is None or result.empty:
            return None, None, "empty"

        min_date = result["min_date"].iloc[0]
        max_date = result["max_date"].iloc[0]

        # 转换日期格式
        if pd.notna(min_date):
            min_date = pd.to_datetime(min_date).strftime("%Y-%m-%d")
        else:
            min_date = None

        if pd.notna(max_date):
            max_date = pd.to_datetime(max_date).strftime("%Y-%m-%d")
        else:
            max_date = None

        return min_date, max_date, "ok"
    except Exception as e:
        return None, None, f"error: {str(e)[:50]}"


def guess_date_columns(db, database: str, table: str) -> List[str]:
    """猜测表中的日期类型列"""
    columns = get_table_columns(db, database, table)
    if columns.empty:
        return []

    date_types = [
        "Date", "Date32", "DateTime", "DateTime64",
        "DateTime('Asia/Shanghai')", "DateTime64('Asia/Shanghai')",
    ]

    date_cols = []
    for _, row in columns.iterrows():
        col_type = str(row["data_type"])
        # 移除参数部分，如 DateTime('Asia/Shanghai') -> DateTime
        base_type = col_type.split("(")[0].strip()
        if base_type in date_types or "Date" in col_type:
            date_cols.append(row["column_name"])

    return date_cols


def probe_database(
    db,
    database: str,
    skip_tables: List[str] = None,
) -> Dict[str, Any]:
    """
    探测指定数据库的所有表
    """
    if skip_tables is None:
        skip_tables = []

    tables = list_tables(db, database)
    report = {
        "database": database,
        "total_tables": len(tables),
        "tables": {},
    }

    for table in tables:
        # 跳过包含指定关键词的表
        if any(skip in table.lower() for skip in skip_tables):
            continue

        # 基础信息
        row_count = get_table_row_count(db, database, table)
        columns_df = get_table_columns(db, database, table)
        date_cols = guess_date_columns(db, database, table)

        # 优先用 date 列探测时间范围
        time_range = {"min": None, "max": None, "status": "unknown"}
        if "date" in date_cols:
            min_d, max_d, status = get_table_time_range(db, database, table, "date")
            time_range = {"min": min_d, "max": max_d, "status": status}
        elif date_cols:
            # 尝试第一个日期列
            min_d, max_d, status = get_table_time_range(db, database, table, date_cols[0])
            time_range = {"min": min_d, "max": max_d, "status": status}

        report["tables"][table] = {
            "row_count": row_count,
            "columns": columns_df.to_dict("records") if not columns_df.empty else [],
            "column_count": len(columns_df),
            "date_columns": date_cols,
            "time_range": time_range,
        }

    return report


# ── 格式化输出 ────────────────────────────────────────────────────────────────


def format_report(report: Dict) -> str:
    """将探测报告格式化为可读文本"""
    lines = []
    db = report["database"]

    lines.append("=" * 80)
    lines.append(f"  ClickHouse 数据探测报告")
    lines.append(f"  数据库: {db}")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # 汇总
    total_tables = report["total_tables"]
    probed = len(report["tables"])
    lines.append(f"[STAT] 汇总")
    lines.append(f"   总表数: {total_tables}")
    lines.append(f"   已探测: {probed}")
    lines.append("")

    # 时间范围统计
    tables_with_time = {
        t: info["time_range"]
        for t, info in report["tables"].items()
        if info["time_range"]["status"] == "ok" and info["time_range"]["min"]
    }

    if tables_with_time:
        all_mins = [v["min"] for v in tables_with_time.values() if v["min"]]
        all_maxs = [v["max"] for v in tables_with_time.values() if v["max"]]
        if all_mins:
            earliest = min(all_mins)
            latest = max(all_maxs) if all_maxs else "unknown"
            lines.append(f"[DATE] 数据时间范围")
            lines.append(f"   最早记录: {earliest}")
            lines.append(f"   最新记录: {latest}")
            lines.append("")

    # 逐表详情
    lines.append("-" * 80)
    lines.append("📋 表详情")
    lines.append("-" * 80)

    for table, info in sorted(report["tables"].items()):
        lines.append("")
        lines.append(f"  ┌{'─' * 76}┐")
        lines.append(f"  │ TABLE: {table:<66} │")
        lines.append(f"  ├{'─' * 76}┤")

        # 行数
        rc = info["row_count"]
        rc_str = f"{rc:,}" if rc >= 0 else "未知"
        lines.append(f"  │ 行数: {rc_str:<68} │")

        # 时间范围
        tr = info["time_range"]
        if tr["status"] == "ok" and tr["min"]:
            time_str = f"{tr['min']} ~ {tr['max']}"
        elif tr["status"] == "no_date_col":
            time_str = "(无日期列)"
        elif tr["status"] == "error":
            time_str = f"(探测失败)"
        else:
            time_str = "未知"
        lines.append(f"  │ 时间范围: {time_str:<62} │")

        # 日期列
        dc = info["date_columns"]
        dc_str = ", ".join(dc) if dc else "无"
        lines.append(f"  │ 日期列: {dc_str:<65} │")
        lines.append(f"  │ 字段数: {info['column_count']:<64} │")
        lines.append(f"  └{'─' * 76}┘")

        # 字段详情
        if info["columns"]:
            lines.append("     字段列表:")
            lines.append(
                f"     {'列名':<30} {'数据类型':<25} {' Nullable':<8} {'默认值'}"
            )
            lines.append(f"     {'-' * 30} {'-' * 25} {'-' * 8} {'-' * 20}")
            for col in info["columns"][:50]:  # 最多显示50个字段
                name = str(col.get("column_name", ""))[:28]
                dtype = str(col.get("data_type", ""))[:23]
                nullable = "YES" if str(col.get("default_kind", "")) == "DEFAULT" else ""
                default = str(col.get("default_value", ""))[:18]
                lines.append(
                    f"     {name:<30} {dtype:<25} {nullable:<8} {default}"
                )
            if len(info["columns"]) > 50:
                lines.append(f"     ... (共 {len(info['columns'])} 个字段)")
        lines.append("")

    return "\n".join(lines)


def format_json_report(report: Dict) -> Dict:
    """将探测报告格式化为 JSON 友好的 dict"""
    output = {
        "metadata": {
            "database": report["database"],
            "probed_at": datetime.now().isoformat(),
            "total_tables": report["total_tables"],
            "probed_tables": len(report["tables"]),
        },
        "tables": {},
    }

    # 全局时间范围
    all_times = []
    for t, info in report["tables"].items():
        tr = info["time_range"]
        if tr["status"] == "ok" and tr["min"]:
            all_times.append({"table": t, "min": tr["min"], "max": tr["max"]})

    if all_times:
        output["metadata"]["earliest_record"] = min(t["min"] for t in all_times)
        output["metadata"]["latest_record"] = max(t["max"] for t in all_times)

    for table, info in report["tables"].items():
        output["tables"][table] = {
            "row_count": info["row_count"],
            "column_count": info["column_count"],
            "date_columns": info["date_columns"],
            "time_range": info["time_range"],
            "columns": info["columns"],
        }

    return output


# ── 主流程 ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    db_config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "database": args.database,
    }

    print(f"\n{'=' * 60}")
    print(f"  ClickHouse 数据探测工具")
    print(f"  主机: {args.host}:{args.port}")
    print(f"  用户: {args.user}")
    print(f"  数据库: {args.database}")
    print(f"{'=' * 60}\n")

    # 连接
    try:
        db = connect(db_config)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # 探测数据库
    print(f"\n[PROBE] 探测数据库...")
    all_dbs = list_databases(db)
    print(f"   发现 {len(all_dbs)} 个数据库: {', '.join(all_dbs)}")

    # 探测指定数据库
    if args.database not in all_dbs:
        print(f"[WARN]  数据库 '{args.database}' 不存在，可用: {all_dbs}")
        sys.exit(1)

    print(f"\n[PROBE] 探测表结构 ({args.database})...")
    report = probe_database(db, args.database, skip_tables=args.skip_tables)

    # 输出
    text_report = format_report(report)

    if args.output:
        # 同时输出文本和 JSON
        json_output = format_json_report(report)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 文本报告
        txt_path = out_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        print(f"[OK] 文本报告已保存: {txt_path}")

        # JSON 报告
        import json

        json_path = out_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON 报告已保存: {json_path}")
    else:
        print(text_report)

    print(f"\n[OK] 探测完成!")


if __name__ == "__main__":
    main()
