# AE_LLM_RL_FOF: dual-track data pipeline and features

from pathlib import Path

# 自动加载项目根目录的 .env 文件
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env, override=False)
    except ImportError:
        pass  # python-dotenv 未安装时静默跳过
