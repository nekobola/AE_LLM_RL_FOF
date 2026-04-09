#!/usr/bin/env python3
"""
run_llm_batch.py — 异步大模型批量推理（断点续传）

定位：剥离最耗时、最易报错的 LLM API 调用，构建本地宏观语义数据库。

核心约束：
  - 断点续传（Checkpointing）：每个周五打分后立即写入 SQLite
  - 重启时扫描已打分的周，跳过历史部分
  - 网络中断不影响历史数据，优雅恢复

调度链路：
  1. 读取历史时间轴，按周频切片
  2. 调用 llm_engine.text_etl 提取当周文本池
  3. 调用 llm_engine.async_semantic_engine 执行并发打分
  4. 实时写入 SQLite（每完成一周立即落盘）

输出落盘：
  - data/llm_cache/llm_scores.db (SQLite)
  - 列: week_end (周五日期), concept, d1, d2, d3, completed_at

用法：
  python scripts/run_llm_batch.py [--start-week YYYY-MM-DD] [--end-date YYYY-MM-DD] [--concurrency 5]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd

from src.llm_engine.async_semantic_engine import AsyncSemanticEngine, LLMCallError

# ── Logging ───────────────────────────────────────────────────────────────────
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "run_llm_batch.log", mode="a"),
    ],
)
log = logging.getLogger("run_llm_batch")


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw["llm"]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    return raw


# ── SQLite Checkpoint ──────────────────────────────────────────────────────────

def init_db(db_path: Path) -> None:
    """初始化 SQLite 表结构（幂等操作）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_scores (
            week_end     TEXT    PRIMARY KEY,  -- 周五日期 YYYY-MM-DD
            concept      TEXT    NOT NULL,
            d1           REAL,
            d2           REAL,
            d3           REAL,
            completed_at TEXT    NOT NULL,
            error        TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_week ON llm_scores(week_end)")
    conn.commit()
    conn.close()


def get_completed_weeks(db_path: Path) -> set[str]:
    """返回已成功打分的周五集合。"""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT DISTINCT week_end FROM llm_scores WHERE error IS NULL"
    )
    weeks = {row[0] for row in cur.fetchall()}
    conn.close()
    return weeks


def insert_scores(db_path: Path, scores: dict, week_end: str) -> None:
    """将一周的打分结果批量写入 SQLite。"""
    import datetime
    conn = sqlite3.connect(db_path)
    now = datetime.datetime.now().isoformat()
    rows = [
        (week_end, concept, vals["d1"], vals["d2"], vals["d3"], now, None)
        for concept, vals in scores.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO llm_scores (week_end,concept,d1,d2,d3,completed_at,error) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def mark_error(db_path: Path, week_end: str, error_msg: str) -> None:
    """记录打分失败的周（但不阻塞后续）。"""
    import datetime
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO llm_scores (week_end,concept,completed_at,error) "
        "VALUES (?,'',?,?)",
        (week_end, datetime.datetime.now().isoformat(), error_msg),
    )
    conn.commit()
    conn.close()


# ── 核心批处理逻辑 ─────────────────────────────────────────────────────────────

async def process_week(
    engine: AsyncSemanticEngine,
    week_end: str,
    db_path: Path,
    config: dict,
) -> bool:
    """
    对单个周五执行LLM打分，写入checkpoint。
    Returns True if successful.
    """
    try:
        scores = await engine.evaluate(week_end)
        insert_scores(db_path, scores, week_end)
        log.info(f"  ✅ {week_end}: {list(scores.keys())}")
        return True
    except LLMCallError as e:
        mark_error(db_path, week_end, str(e))
        log.warning(f"  ⚠️  {week_end}: LLM失败 → 已checkpoint错误，继续: {e}")
        return False
    except Exception as e:
        mark_error(db_path, week_end, str(e))
        log.error(f"  ❌ {week_end}: 未知错误: {e}")
        return False


async def run_batch(
    start_week: str,
    end_date: str | None,
    concurrency: int,
    db_path: Path,
    config: dict,
) -> pd.DataFrame:
    """
    执行完整批处理。

    1. 构建周频日期列表（每周五）
    2. 过滤掉已完成且无错的周
    3. 用信号量控制并发
    4. 每完成一周立即写checkpoint
    """
    # ── 构建周频日期列表 ────────────────────────────────────────────────────
    if end_date is None:
        end = date.today()
    else:
        end = date.fromisoformat(end_date)

    start = date.fromisoformat(start_week)

    # 找到 start 后的第一个周五
    days_to_friday = (4 - start.weekday()) % 7  # Monday=0
    if days_to_friday == 0 and start.weekday() == 4:
        first_friday = start
    else:
        first_friday = start + timedelta(days=days_to_friday if days_to_friday > 0 else 7)

    fridays = []
    current = first_friday
    while current <= end:
        fridays.append(current.isoformat())
        current += timedelta(weeks=1)

    log.info(f"══ LLM Batch: {len(fridays)} 个周五待处理 ═")
    log.info(f"    日期范围: {fridays[0]} → {fridays[-1]}")
    log.info(f"    并发数: {concurrency}")

    # ── 过滤已完成的周 ────────────────────────────────────────────────────
    completed = get_completed_weeks(db_path)
    pending = [f for f in fridays if f not in completed]
    skipped = len(fridays) - len(pending)
    if skipped > 0:
        log.info(f"    跳过已完成的周: {skipped} 个")
    log.info(f"    待处理: {len(pending)} 个")

    if not pending:
        log.info("无待处理周，加载已有数据返回")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM llm_scores WHERE error IS NULL", conn)
        conn.close()
        return df

    # ── 并发打分 ──────────────────────────────────────────────────────────
    engine = AsyncSemanticEngine(config)
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, bool] = {}

    async def bounded_process(week_end: str) -> tuple[str, bool]:
        async with semaphore:
            return week_end, await process_week(engine, week_end, db_path, config)

    tasks = [bounded_process(f) for f in pending]
    for coro in asyncio.as_completed(tasks):
        week_end, ok = await coro
        results[week_end] = ok

    # ── 汇总 ──────────────────────────────────────────────────────────────
    n_ok = sum(1 for v in results.values() if v)
    n_fail = len(results) - n_ok
    log.info(f"══ Batch 完成: ✅{n_ok}  ❌{n_fail} ═")

    # ── 加载全部数据 ───────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM llm_scores WHERE error IS NULL", conn)
    conn.close()
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="run_llm_batch: LLM批量推理（断点续传）")
    parser.add_argument("--start-week", default="2022-01-07")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--checkpoint-db", default=None)
    args = parser.parse_args()

    config = load_config()

    if args.checkpoint_db:
        db_path = Path(args.checkpoint_db)
    else:
        db_path = PROJECT_ROOT / config.get("paths", {}).get(
            "llm_cache", "data/llm_cache"
        ) / "llm_scores.db"

    init_db(db_path)
    log.info(f"Checkpoint数据库: {db_path}")

    try:
        df = asyncio.run(run_batch(
            start_week=args.start_week,
            end_date=args.end_date,
            concurrency=args.concurrency,
            db_path=db_path,
            config=config,
        ))
        log.info(f"✅ 累计打分: {len(df)} 条，覆盖 {df['week_end'].nunique()} 个周五")
        print(df.groupby("concept")[["d1","d2","d3"]].mean().to_string())
    except Exception as e:
        log.error(f"Batch失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
