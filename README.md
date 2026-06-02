# AE_LLM_RL_FOF

> 架构解耦、动态路由的 FOF 智能体工作流 —— Autoencoder 感知宏观压迫感，LLM 降维充当特征工程与风险阻断器，PPO 升维为系统超参数调度的元控制器。

设计文档: [飞书 Wiki](https://mcnx64hcm9yb.feishu.cn/wiki/OW5NwsH8rinjrQkN99UcOwH3nHd)

---

## 全局数据流图

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                                   数  据  源  层                                      │
 ├──────────────────────┬─────────────────────────────┬─────────────────────────────────┤
 │  ClickHouse etf_day  │  akshare 宏观 API           │  ClickHouse text_db             │
 │  5只ETF日频OHLCV     │  Shibor/汇率/国债/两融       │  zgrmyh/csrc/govcn/news         │
 │  track_b/fetcher.py  │  macro_features.py           │  text_etl.py                    │
 └────────┬─────────────┴──────────┬──────────────────┴────────────┬────────────────────┘
          │                        │                               │
          ▼                        ▼                               │
 ┌────────────────────────────────────────┐                        │
 │  asset_features.py   macro_features   │                        │
 │  5资产×4特征=20维  +  5维宏观         │                        │
 │  (动量/波动率/相关性/周收益)           │                        │
 └──────────────────┬─────────────────────┘                        │
                    │                                              │
                    ▼                                              │
 ┌──────────────────────────────────────┐                          │
 │  normalizer.py  滚动Z-score标准化     │                          │
 │  窗口 [t-252, t-1]  严格防穿越        │                          │
 └──────────────────┬───────────────────┘                          │
                    │                                              │
                    ▼                                              │
          ┌─────────────────────┐                                  │
          │ features_master     │                                  │
          │ .parquet  (T × 25)  │                                  │
          └────────┬────────────┘                                  │
                   │                                               │
     ┌─────────────┼─────────────┐                                 │
     │                           │                                 │
     ▼                           │                                 ▼
 ┌─────────────────────────┐     │    ┌──────────────────────────────────────────┐
 │   AE 自编码器 (离线)     │     │    │   LLM 语义引擎 (离线批量)                  │
 │   models/                │     │    │   llm_engine/                             │
 │                         │     │    │                                           │
 │  Encoder: 25→16→6→Tanh │     │    │  TextETL.extract_per_concept(t)            │
 │  Decoder: 6→16→25      │     │    │    ├─ MPC会议记录 (最近1条)                 │
 │                         │     │    │    ├─ CSRC动态标题 [t-7, t]                │
 │  X ──► Z(6维) ──► X̂    │     │    │    ├─ govcn政策 [t-30, t] 概念模糊匹配     │
 │                         │     │    │    └─ 财经新闻 [t-7, t] Top20截断           │
 │  Loss = MSE(X, X̂)      │     │    │                                           │
 └────────────┬────────────┘     │    │  PromptBuilder.build()                     │
              │                  │    │    ├─ System: d1/d2/d3 评分指南             │
              ▼                  │    │    └─ User: 共享宏观+各概念政策+上周参考     │
 ┌─────────────────────────┐     │    │                                           │
 │  E_t = ||X - X̂||²       │     │    │  AsyncOpenAI (2路并发)                     │
 │  reconstruction_error   │     │    │    ├─ equity pool (8概念)                   │
 │  宏观压迫感 (1维标量)    │     │    │    └─ fixed_income pool (2概念)             │
 └────────────┬────────────┘     │    │                                           │
              │                  │    │  ResponseParser 校验 → SQLite落盘           │
              │                  │    └──────────────────┬────────────────────────┘
              │                  │                       │
              ▼                  │                       ▼
 ┌─────────────────────────┐     │    ┌──────────────────────────────────────────┐
 │  inference/ 后处理       │     │    │  llm_scores.db                            │
 │  EMA(α=0.05) → Robust   │     │    │  {concept: {d1, d2, d3}}                  │
 │  ZScore → Clip[-5,5]    │     │    │  d1=流动性顺风 d2=资金情绪 d3=风险压力     │
 │  → BurnIn(156周盲区)    │     │    │  (10概念 × 3维 → 聚合3维信号)               │
 └────────────┬────────────┘     │    └──────────────────┬────────────────────────┘
              │                  │                       │
              ▼                  │                       ▼
        E_t (标准化)             │              llm_macro / llm_sentiment / llm_risk
              │                  │                       │
              └──────────────────┼───────────────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
     │  + vol_mkt_20d            │  + port_sharpe / mdd      │
     │    (市场20日波动率)        │    + regret_ema           │
     │                           │    + tau_prev / alpha_prev│
     └───────────────────────────┼───────────────────────────┘
                                 │
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                      StateAssembler.assemble()  →  S_t ∈ R^10                        │
 │  s0:AE误差  s1:波动率  s2:LLM宏观  s3:LLM情绪  s4:LLM风险                            │
 │  s5:Sharpe  s6:回撤    s7:Regret   s8:τ上期   s9:α上期                               │
 └────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
 ┌──────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────┐
 │  NormalTrack (防御轨)         │  │  EventTrack (进攻轨)      │  │  PPO Actor-Critic    │
 │  compute/                    │  │  compute/                │  │  ppo/                │
 │                             │  │                          │  │                      │
 │  LedoitWolf 协方差收缩       │  │  三原型 softmax 混合:     │  │  Actor: 10→64→64→2  │
 │  ERC 等风险贡献优化(SLSQP)   │  │  Crisis / Reflation      │  │  Critic:10→64→64→1  │
 │  bounds: 各资产[5%-60%]     │  │  / Growth                │  │  正交初始化          │
 │                             │  │                          │  │                      │
 │  → W_Normal (5维)           │  │  LLM信号 + 资产波动率     │  │  GAE(γ=0.99,λ=0.95) │
 │    防御权重向量              │  │  → W_Event (5维)         │  │  Clip(ε=0.2) K=4    │
 │                             │  │    进攻权重向量           │  │  PPO Clip Loss      │
 └──────────────┬──────────────┘  └────────────┬─────────────┘  └──────────┬───────────┘
                │                              │                            │
                │                              │               ActionMapper │
                │                              │               a1→Δα[-0.5,+0.1]         │
                │                              │               a2→Δτ[-0.1,+0.1]         │
                │                              │                            │
                │                              │                    α_new, τ_new        │
                │                              │                            │
                └──────────────┬───────────────┘                            │
                               │                                            │
                               ▼                                            │
 ┌──────────────────────────────────────────────────────────────────────────┴──────────┐
 │                                                                                        │
 │              w_final = α · W_Event  +  (1-α) · W_Normal                               │
 │              clip(0,1) → normalize(sum=1)                                              │
 │                                                                                        │
 │              α>0.5 → 进攻偏向 (EventTrack主导)     regime = ae_error < τ ? Bull : Bear │
 │              α<0.5 → 防御偏向 (NormalTrack主导)     SwitchBonus: 方向正确+0.45 错误-0.15│
 └───────────────────────────────────────────────────────────────────┬────────────────────┘
                                                                     │
                         ┌───────────────────────────────────────────┤
                         │                                           │
                         ▼                                           ▼
 ┌──────────────────────────────────┐      ┌──────────────────────────────────┐
 │  failsafe/  风险熔断              │      │  WFO 调度器  schedules/           │
 │                                  │      │                                  │
 │  VetoSwitch: d3 > 85 → 仓位清零  │      │  Burn-in: Phase1(104周AE基座)     │
 │  FallbackSelector: LLM宕机→SQL   │      │         + Phase2(52周MAD铸造)     │
 │    选基(动量/波动率排序)         │      │  季度重训: 3/6/9/12月末触发        │
 └────────────────┬─────────────────┘      │  周频推断: 每周末 E_t 前向         │
                  │                        └──────────────────────────────────┘
                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                                    输  出  层                                          │
 │                                                                                       │
 │  回测模式:  nav_series.csv / weights_trajectory.csv / metrics.json / tearsheet.txt     │
 │             weights_data_generalbt.csv + GeneralBacktest 绘图 (figures/)               │
 │                                                                                       │
 │  实盘模式:  target_weights_{date}.json  ──►  AgentBase 标准格式  ──►  交易系统         │
 │             {date: {etf_code: weight, ...}}                                            │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心思想

传统强化学习（RL）在量化配置中常因**状态空间维度爆炸**和**相关性崩塌**导致策略过拟合。本方案的关键创新在于**三层解耦架构**：

- **AE (Regime AutoEncoder)**: 将 25 维宏观/资产特征压缩为 6 维潜在表征，通过重建误差 $E_t = ||X - \hat{X}||^2$ 量化"宏观压迫感"，作为市场异变的连续探测器。
- **LLM (大语言模型)**: 批量读取政策文本、市场新闻，输出 d1(流动性顺风)、d2(资金情绪)、d3(风险压力指数) 三组语义评分，充当可解释的宏观特征工程与尾部风险一票否决。
- **PPO (元控制器)**: 不直接选基，不直接生成下单权重，只输出两个标量——融合系数 α(攻防比) 和 门控阈值 τ(牛熊切换灵敏度)，使 RL 从高维动作空间降维为超参数调度问题。

**融合机制**: `w_final = α · w_event + (1-α) · w_normal`
- α > 0.5: 倾向进攻轨 (EventTrack)，LLM 信号驱动弹性配置
- α < 0.5: 倾向防御轨 (NormalTrack)，协方差加权防守配置
- Veto Switch: 当任意概念 d3 > 85，一票否决，强制降权

---

- **AE (Regime AutoEncoder)**: 将 25 维宏观/资产特征压缩为 6 维潜在表征，通过重建误差 $E_t = ||X - \hat{X}||^2$ 量化"宏观压迫感"，作为市场异变的连续探测器。
- **LLM (大语言模型)**: 批量读取政策文本、市场新闻，输出 d1(流动性顺风)、d2(资金情绪)、d3(风险压力指数) 三组语义评分，充当可解释的宏观特征工程与尾部风险一票否决。
- **PPO (元控制器)**: 不直接选基，不直接生成下单权重，只输出两个标量——融合系数 α(攻防比) 和 门控阈值 τ(牛熊切换灵敏度)，使 RL 从高维动作空间降维为超参数调度问题。

**融合机制**: `w_final = α · w_event + (1-α) · w_normal`
- α > 0.5: 倾向进攻轨 (EventTrack)，LLM 信号驱动弹性配置
- α < 0.5: 倾向防御轨 (NormalTrack)，协方差加权防守配置
- Veto Switch: 当任意概念 d3 > 85，一票否决，强制降权

---

## 目录结构

```
AE_LLM_RL_FOF-main/
├── config.yaml                  # 全局配置文件
├── pyproject.toml               # Python 项目依赖
├── .env                         # 环境变量 (API Key, 数据库连接)
│
├── data/
│   ├── raw/                     # 原始数据
│   ├── processed/               # ETL 输出: features_master.parquet
│   └── llm_cache/               # LLM 打分 SQLite 数据库
│
├── checkpoints/                 # 模型权重 & scaler
│   ├── ae_weights.pth           # AE 自编码器权重
│   ├── ae_scaler.pkl            # 标准化参数
│   └── actor_critic.pth         # PPO Actor-Critic 权重
│
├── src/
│   ├── features/                # 特征工程
│   │   ├── asset_features.py    # 资产特征 (动量/波动率/相关性/周收益)
│   │   ├── macro_features.py    # 宏观特征 (DR007/汇率/国债/利差/北向)
│   │   ├── normalizer.py        # 防穿越 Z-score 标准化
│   │   └── reconstruction_error.py  # AE 重建误差计算
│   │
│   ├── models/
│   │   └── regime_autoencoder.py    # AE 网络: 25→16→6→16→25
│   │
│   ├── llm_engine/              # LLM 语义引擎
│   │   ├── prompt_builder.py    # System/User Prompt 构建
│   │   ├── async_semantic_engine.py  # 异步并发打分引擎
│   │   ├── response_parser.py   # JSON 响应解析
│   │   ├── text_etl.py          # 文本数据提取 (政策/新闻)
│   │   └── concept_pools.py     # 概念池定义 (宽基/卫星/固收)
│   │
│   ├── compute/                 # 双轨测算引擎
│   │   ├── dual_track_engine.py # 异构双轨并发入口
│   │   ├── normal_track.py      # 防御轨: 协方差加权防守
│   │   └── event_track.py       # 进攻轨: LLM 信号驱动弹性配置
│   │
│   ├── env/                     # MDP 环境 (Gymnasium)
│   │   ├── mdp_environment.py   # 10维状态 × 2维动作 MDP
│   │   ├── reward_function.py   # 复合奖励函数
│   │   ├── state_assembler.py   # 10维状态空间组装
│   │   ├── action_mapper.py     # Action → Δα/Δτ 映射
│   │   ├── regret_engine.py     # 遗憾最小化引擎
│   │   └── metrics_utils.py     # Sharpe/MDD 计算
│   │
│   ├── ppo/                     # PPO 训练模块
│   │   ├── networks.py          # Actor-Critic 独立网络 (正交初始化)
│   │   ├── trainer.py           # PPO 训练器 (Clip + GAE)
│   │   ├── buffer.py            # Rollout 缓冲池
│   │   ├── gae.py               # GAE 优势估计
│   │   └── loss.py              # Actor/Critic/Entropy 损失
│   │
│   ├── inference/               # 推断与后处理
│   │   ├── ema_filter.py        # EMA 平滑滤波
│   │   └── panic_index_output.py # 恐慌指数输出
│   │
│   ├── failsafe/                # 风险熔断
│   │   ├── veto_switch.py       # LLM d3 一票否决
│   │   └── fallback_selector.py # 降级备选选择器
│   │
│   ├── selection/               # 选基 & 权重映射
│   │   ├── concept_to_etf_map.py
│   │   └── slot_weighting.py
│   │
│   ├── synthesis/               # 合成资产构建
│   │   └── covariance_weighter.py
│   │
│   ├── schedules/               # Walk-Forward 调度
│   │   └── wfo_scheduler.py     # 季度重训 + 周频推断
│   │
│   ├── training/                # 训练流程编排
│   │   ├── burn_in/             # 冷启动 (Phase1初始化 + Phase2 MAD校准)
│   │   └── dual_track/trainer.py
│   │
│   ├── data_pipeline/           # 数据管道
│   │   ├── track_b/fetcher.py   # ClickHouse 日频 ETF 数据
│   │   └── track_a/fetcher.py   # 其他数据源
│   │
│   └── penetration/             # 实盘信号下发
│       └── agentbase_formatter.py  # AgentBase 标准格式
│
├── scripts/                     # 可执行脚本
│   ├── run_data_etl.py          # Step 1: 数据清洗与特征构建
│   ├── run_llm_batch.py         # Step 2: LLM 批量推理 (断点续传)
│   ├── train_ae.py              # Step 3: AE 自编码器预训练
│   ├── train_ppo.py             # Step 4: PPO 元控制器沙盒训练
│   ├── run_backtest_wfo.py      # Step 5: Walk-Forward 滚动回测
│   └── run_inference_live.py    # Step 6: 实盘/准实盘信号下发
│
├── tests/                       # 单元测试
│   ├── test_mdp_environment_alignment.py
│   ├── test_event_track_prototypes.py
│   ├── test_regime_autoencoder.py
│   └── test_normalizer_no_lookahead.py
│
├── results/                     # 回测输出
│   └── wfo/                     # Walk-Forward 回测结果
│       └── <run_id>/
│           ├── nav_series.csv
│           ├── weights_trajectory.csv
│           ├── weights_data_generalbt.csv
│           ├── metrics.json
│           ├── gate_diagnostics.csv
│           ├── tearsheet.txt
│           └── figures/         # 可视化图表
│
├── logs/                        # 运行日志
└── docs/                        # 文档与数据字典
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA (可选, 用于 GPU 训练)

### 安装

```bash
pip install -e ".[dev]"
```

### 配置

1. 编辑 `.env` 配置 API Key 和数据库连接
2. 编辑 `config.yaml` 调整超参数 (默认配置可直接使用)

---

## 流水线

完整的 6 步操作流水线：

### Step 1: 数据 ETL

```bash
python scripts/run_data_etl.py --start-date 2015-01-01
```

从 ClickHouse 拉取 ETF 日频数据，计算 5 资产 × 4 特征 + 5 维宏观特征 = 25 维特征矩阵，执行防穿越 Z-score 标准化，输出 `data/processed/features_master.parquet`。

### Step 2: LLM 批量打分

```bash
python scripts/run_llm_batch.py --start-week 2022-01-07 --concurrency 5
```

批量调用 LLM 对每个周五的宏观文本进行 d1/d2/d3 三维语义评分，支持断点续传，结果写入 `data/llm_cache/llm_scores.db`。

### Step 3: AE 自编码器训练

```bash
python scripts/train_ae.py --epochs 50 --batch-size 256
```

训练 Regime AutoEncoder 将 25 维特征压缩为 6 维潜在表征，输出 `checkpoints/ae_weights.pth` 和 `checkpoints/ae_scaler.pkl`。

### Step 4: PPO 元控制器训练

```bash
python scripts/train_ppo.py --total-timesteps 100000
```

在 MDP 环境中训练 Actor-Critic 网络，输出 `checkpoints/actor_critic.pth`。支持 TensorBoard 监控 (`logs/tensorboard`)。

### Step 5: Walk-Forward 回测

```bash
python scripts/run_backtest_wfo.py --start-date 2015-01-01 --lookback-weeks 104
```

季度重训 + 周频推断的滚动回测，输出净值曲线、权重轨迹、GeneralBacktest 格式文件及可视化图表。

### Step 6: 实盘信号下发

```bash
python scripts/run_inference_live.py --week-end 2025-06-06
```

每周五盘后触发：增量更新数据 → 单周 LLM 打分 → 加载最新模型 → 前向传播 → 输出 `results/wfo/target_weights_*.json`（标准 AgentBase 格式）。

---

## 关键设计

### 10 维状态空间 S_t

| 维度 | 名称 | 含义 |
|------|------|------|
| s0 | AE 重建误差 (标准化) | 宏观压迫感强度 |
| s1 | 市场波动率 (标准化) | 风险水平 |
| s2 | LLM 宏观评分 d1 | 流动性顺风程度 |
| s3 | LLM 情绪评分 d2 | 市场情绪热度 |
| s4 | LLM 风险评分 d3 | 尾部风险压力 |
| s5 | 组合 Sharpe (20日) | 近期风险调整收益 |
| s6 | 当前最大回撤 | 组合回撤深度 |
| s7 | 遗憾 EMA (标准化) | Regret 跟踪信号 |
| s8 | 上一期 τ (标准化) | 牛熊切换阈值 |
| s9 | 上一期 α (标准化) | 攻防融合比 |

### 2 维动作空间

- **a₁ → Δα**: 攻防切换增量 (α ∈ [0, 1])
- **a₂ → Δτ**: 牛熊阈值增量 (τ ∈ [5, 50])

### 复合奖励函数

$$R_t = \underbrace{r_{rel}(t)}_{\text{相对收益}} + \lambda_{end} \cdot r_{end}(t) - \lambda_{turnover} \cdot C_{TO}(t) - \lambda_{TE} \cdot TE(t) - \lambda_{mdd} \cdot \kappa \cdot MDD(t) \pm \text{SwitchBonus}$$

### 5 资产类别

| 编号 | 类型 | 代表标的 | 配置角色 |
|------|------|---------|---------|
| 0 | 宽基指数 | 沪深300 (000300.SH) | 收益中枢 |
| 1 | 卫星资产 | 中证1000 (000852.SH) | Alpha 增强 |
| 2 | 固收 | 国债指数 (CBA02701.CS) | 防御底仓 |
| 3 | 避险 | 黄金 (AU9999.SGE) | 尾部对冲 |
| 4 | 现金 | 南华商品 (NH0100.NHF) | 流动性储备 |

### 风险熔断 (Veto Switch)

LLM 对每个概念输出 d3 风险压力指数 (0-100)。当任意 ETF 对应的概念 d3 > 85 时触发一票否决，该概念对应仓位被强制降至 0，由下一顺位替补。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch >= 2.0 |
| RL 环境 | Gymnasium >= 0.29 |
| LLM 调用 | OpenAI SDK (兼容 Qwen/Claude 等) |
| 数据处理 | Pandas, NumPy, PyArrow |
| 数据源 | ClickHouse (quantchdb), PostgreSQL |
| 监控 | TensorBoard |
| 测试 | Pytest, Pytest-Asyncio |
| 代码质量 | Ruff |

---

## 测试

```bash
pytest tests/ -v
```

- `test_regime_autoencoder.py` — AE 网络结构与前向传播验证
- `test_mdp_environment_alignment.py` — MDP 环境 reset/step 对齐测试
- `test_event_track_prototypes.py` — 双轨引擎 EventTrack 原型验证
- `test_normalizer_no_lookahead.py` — 标准化防穿越机制验证

---

## 模块数据流详解

### 1. features/ — 特征工程 (25维特征矩阵构建)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 1.1 资产特征 | `price_df` [date × 5资产] 从 ClickHouse `etf.etf_day` 拉取 | `compute_asset_features()` → 5资产 × 4特征(weekly_return / volatility_20d / momentum_20d / mean_corr_20d) = **20维**，所有窗口严格 [t-N, t-1] 防穿越 | DataFrame [date × 20列] 列名格式 `{code}__{feature}` | 1.4 特征拼接 |
| 1.2 宏观特征 | akshare API (`ak.macro_china_shibor_all`, `ak.currency_boc_safe`, `ak.bond_zh_us_rate`, `ak.macro_china_market_margin_sh/sz`) | `compute_macro_features()` → 5维: DR007 / CNY_USD_Offshore / Yield_10Y_CGB / Term_Spread / Northbound_Flow (两融20日动量), ffill填充缺失 | DataFrame [date × 5列] | 1.4 特征拼接 |
| 1.3 标准化 | 1.4输出的25维矩阵 | `normalize_dataframe()` → 严格滚动Z-score: mean/std用[t-252, t-1]窗口, 不包含t时刻, `shift(1)`确保防穿越, min_periods=60 | DataFrame [date × 25列] 标准化后 | 1.5 落盘 + AE训练 |
| 1.4 特征拼接 | 1.1 (20维) + 1.2 (5维) | `pd.concat([asset_feat, macro_feat], axis=1)` 按date索引对齐, dropna | DataFrame [date × 25列] | 1.3 标准化 |
| 1.5 落盘 | 1.3 标准化后的25维矩阵 | `to_parquet("data/processed/features_master.parquet")` | `features_master.parquet` | AE训练, PPO训练, WFO回测, 实盘推断 |
| 1.6 AE重建误差 | `RegimeAutoEncoder` (已训练) + 25维特征向量 | `compute_reconstruction_error()` → `E_t = \|\|X - Decoder(Encoder(X))\|\|_2^2` 逐样本计算, batch模式支持(N,25)→(N,) | float / np.ndarray (N,) | StateAssembler (s0), PanicIndexOutput |

### 2. models/ — AE 自编码器 (宏观压迫感提取器)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 2.1 网络定义 | `input_dim=25, latent_dim=6, hidden_dim=16` | `RegimeAutoEncoder`: Encoder(25→16→6→Tanh) + Decoder(6→16→25), LeakyReLU(0.01) | 模型实例 | train_ae.py, 重建误差计算 |
| 2.2 编码 | `X_t` shape (batch, 25) | `encoder(X)` → 25→16→6→Tanh → **6维潜在表征Z_t** | Tensor (batch, 6) | 诊断, 潜在空间分析 |
| 2.3 解码 | `Z_t` shape (batch, 6) | `decoder(Z)` → 6→16→25 → **重建X_hat** | Tensor (batch, 25) | 2.4 损失计算 |
| 2.4 损失 | X vs X_hat | `MSELoss(X_hat, X)` 最小化重建误差 | 标量 loss | 反向传播 |
| 2.5 权重保存 | 训练完成的model + scaler参数 | `torch.save(model.state_dict())` + `pickle.dump(scaler_state)` | `ae_weights.pth`, `ae_scaler.pkl` | PPO训练, WFO回测, 实盘推断 |

### 3. llm_engine/ — LLM 语义引擎 (d1/d2/d3三维评分)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 3.1 文本ETL | ClickHouse `text_db` (zgrmyh/csrc/govcn/eastmoney/sina), 当前周五日期 | `TextETL.extract_per_concept()` → MPC会议记录(最近1条) + CSRC标题([t-7,t]) + govcn政策([t-30,t]模糊匹配) + 财经新闻([t-7,t] Top20截断) | `etl_data` dict: `{shared: {mpc, csrc}, concepts: {concept: {govcn, news}}}` | 3.2 Prompt构建 |
| 3.2 Prompt构建 | 3.1的etl_data + concept_list (权益8概念/固收2概念) + prior_scores (上周d1/d2/d3) | `PromptBuilder.build()` → 组装System Prompt(评分指南+d3评分逻辑) + User Prompt(共享宏观+各概念政策/新闻+上周参考) | 两个Themed Prompt字符串 (equity + fixed_income) | 3.3 LLM调用 |
| 3.3 异步并发 | 3.2的两个Prompt | `AsyncSemanticEngine.evaluate()` → asyncio.gather 并发调用2次 AsyncOpenAI (temperature=0.0, response_format=json_object), 指数退避重试(max 3次) | 两个JSON字符串 | 3.4 解析 |
| 3.4 响应解析 | LLM返回的JSON字符串 | `ResponseParser.parse()` → json.loads → 验证每个concept含d1/d2/d3 ∈ [1.0, 100.0] | `dict[concept][d1/d2/d3]` (10概念×3维) | 3.5 SQLite落盘 |
| 3.5 断点续传 | 3.4的评分结果 + week_end日期 | `INSERT OR REPLACE INTO llm_scores` (week_end, concept, d1, d2, d3, completed_at) | `llm_scores.db` (SQLite) | StateAssembler (s2/s3/s4), WFO回测 |
| 3.6 概念池 | 硬编码定义 | `CONCEPT_POOLS`: wide_base(沪深300/中证1000) + satellite(6行业) + fixed_income(利率债/信用债) | 10个概念名称列表 | TextETL, PromptBuilder, slot_weighting |

### 4. compute/ — 双轨测算引擎 (防御轨+进攻轨)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 4.1 防御轨 NormalTrack | `returns_5d` shape (5, T≥5) | LedoitWolf协方差收缩 → ERC等风险贡献优化 (SLSQP, sum(w)=1, 各资产上下限) → 失败回退等权[0.2]*5 | `W_Normal` shape (5,) | 4.3 融合 |
| 4.2 进攻轨 EventTrack | `returns_5d` shape (5,T) + llm_macro/sentiment/risk ∈ [0,100] | 计算crisis/reflation/growth三原型得分(softmax) → event_intensity混合(base_neutral+proto_weights) → 归一化 | `W_Event` shape (5,) | 4.3 融合 |
| 4.3 权重融合 | W_Normal + W_Event + α (PPO输出) | `w_final = α * w_event + (1-α) * w_normal`, clip & normalize | `W_final` shape (5,) | MDP Environment (reward计算), 回测NAV, 实盘信号 |

### 5. env/ — MDP 环境 (10维状态 × 2维动作)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 5.1 状态组装 | 10个原始信号: ae_error, vol_mkt_20d, llm_macro/sentiment/risk, port_sharpe, mdd, regret_ema, tau_prev, alpha_prev | `StateAssembler.assemble()` → AE Z-score标准化, vol MinMax→[0,1], llm (x-50)/50→[-1,1], d3三阶段异常清洗(滚动中位数/周变化上限/硬截断), Sharpe硬截断[-3,3] | `S_t` shape (10,) np.float32 | Actor-Critic网络 |
| 5.2 动作映射 | Actor输出 (a1, a2) ∈ [-1,1]² | `ActionMapper.map()` → **非对称映射**: Δα: [-1,1]→[-0.5, 0.1] (砍仓快/加仓慢) + bias; Δτ: [-1,1]→[-0.1, 0.1] (对称) | `(Δα, Δτ)` | 5.3 环境step |
| 5.3 环境step | live_data注入 (ae_error/vol/llm/returns/w_normal/w_event) + action (a1,a2) | `MDPEnvironment.step()` → action→Δα/Δτ→α_new/τ_new → w_final融合 → r_port计算 → reward计算 → 状态更新 → next_state组装 | `(next_state, reward, terminated, truncated, info)` | PPO Trainer (rollout收集) |
| 5.4 复合奖励 | r_port, w_final_t, w_final_{t-1}, port_returns, bench_returns, equity_curve, regret_ema, regime_bull | `RewardFunction.compute()` → **牛熊双轨**: Bull(ae_error<τ): TE惩罚; Bear: 相对收益+MDD差惩罚 + turnover成本 + endpoint惩罚(α偏离0.5) + SwitchBonus(方向正确奖励/错误惩罚) + 一致性nudge | 标量 reward (float) | GAE优势估计 |
| 5.5 遗憾引擎 | w_final_prev (5,) + period_return (5,) | `RegretEngine.compute()` → 16个专家候选库 (含逆波动率/纯现金/纯黄金/纯债券/grid组合) → max(r_opt - r_actual, 0) → EMA(0.8)平滑 → 历史max归一化 | `(regret_ema, regret_ema_norm)` ∈ [0,1] | StateAssembler (s7) |
| 5.6 指标工具 | port_returns, bench_returns, equity_curve | `calculate_sharpe_ratio()` / `calculate_current_drawdown()` / `calculate_tracking_error()` | Sharpe / MDD / TE (float) | StateAssembler, RewardFunction |

### 6. ppo/ — PPO 训练模块 (Actor-Critic 优化)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 6.1 网络定义 | state_dim=10, action_dim=2, hidden_dim=64 | `ActorNetwork`: 10→64→64→2(Tanh) + 独立log_std; `CriticNetwork`: 10→64→64→1 (无激活); 正交初始化; Actor/Critic **不共享权重** | `ActorCritic` 模型 | 6.3 训练器 |
| 6.2 Rollout收集 | MDPEnvironment + live_data_list (历史数据) | `PPOTrainer.collect_rollout()` → env.reset → for T steps: inject_live_data → Actor前向(Normal采样+Tanh) → env.step → buffer.add(s,a,r,V(s),logπ,done) | RolloutBuffer (T×6数组) | 6.4 GAE + 6.5 更新 |
| 6.3 Buffer管理 | 每步的 (s,a,r,v,logp,done) | `RolloutBuffer` 固定容量T=100, 循环写入, is_full时触发更新, 支持shuffle+mini-batch split | 已填充buffer | 6.4 GAE |
| 6.4 GAE优势估计 | buffer中 (rewards, values, dones) + bootstrap_value V(s_T) | `GAEBuffer.compute()` → δ_t = r_t + γ·V(s_{t+1}) - V(s_t) → 反向累积: A_t = δ_t + γλ·A_{t+1} (dones截断) → 优势标准化 | `(advantages, value_targets)` | 6.5 PPO更新 |
| 6.5 PPO更新 | mini-batches: states, actions, advantages, value_targets, old_log_probs | `total_ppo_loss()` → Clip Loss: -min(ratio·A, clip(ratio,1-ε,1+ε)·A); Critic Loss: 0.5·MSE(V_pred, V_target); Entropy Bonus: -c_e·H; → Adam优化 → K=4 epochs | 更新的Actor-Critic权重 | 6.6 Checkpoint |
| 6.6 Checkpoint | 训练完成的ac权重 + optimizer状态 + step_count | `torch.save({ac, optimizer, step_count})` 每10个PPO iter保存一次 | `actor_critic.pth` | WFO回测, 实盘推断 |

### 7. inference/ — 推断与后处理 (恐慌指数输出)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 7.1 EMA滤波 | `E_t_raw` (AE重建误差) | `EMAFilter.step()` → E_smoothed = α·E_raw + (1-α)·E_prev, α=0.05, 状态机递推(仅保留上一时刻) | `E_smoothed` (float) | 7.2 Robust Z-score |
| 7.2 Robust Z-score | E_smoothed 序列 | `RobustZScore.step()` → 滚动窗口Z-score + MAD底层防线(mad_safe_floor=0.05防除零) | `E_zscore` (float) | 7.3 截断 |
| 7.3 状态截断 | E_zscore | `StateClipper.clip()` → hard clip到 [clip_min=-5.0, clip_max=5.0] | `E_clipped` ∈ [-5.0, 5.0] | 7.4 Burn-in |
| 7.4 Burn-in处理 | E_clipped + burn_in_weeks计数 | `BurnInHandler.handle()` → 冷启动期(Phase1+Phase2=156周)返回0.0(中性), 之后放行 | `Final_State` ∈ [-5.0, 5.0] | StateAssembler (s0标准化后) |
| 7.5 整合输出 | E_raw + config | `PanicIndexOutput.step()` → 串联 7.1→7.2→7.3→7.4 | `Final_State` (float) | MDP环境 |

### 8. failsafe/ — 风险熔断 (一票否决+降级备选)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 8.1 一票否决 | LLM评分 dict (concept→d1/d2/d3) + ranked_concepts列表 | `VetoSwitch.apply_veto()` → 遍历d3>85的概念→从排序列表中移除→由下一顺位替补 | 过滤后的concept列表 | 选基权重映射 |
| 8.2 降级备选 | current_date + ClickHouse配置 | `FallbackSelector.select_8()` → LLM宕机时纯SQL选基: 宽基(20日动量Top1)/卫星(20日动量Top3去重)/固收(20日波动率升序Top2)/避险(固定518880)/现金(固定511850) | `dict[slot→etf_code]` (8只ETF) | 权重映射 & 实盘下发 |

### 9. selection/ — 选基 & 权重映射 (概念→ETF→权重)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 9.1 概念→ETF映射 | concept名称 + 配置override | `get_etf_pool_by_concept()` → 查 DEFAULT_ETF_POOLS (如"人工智能"→["159819","515070"], "沪深300"→["510300"]) | ETF代码列表 | 选基 & 下单 |
| 9.2 插槽权重评分 | d1/d2/d3 + pool_type (wide_base/satellite/fixed_income) | `compute_slot_score()` → Score = p1·d1 + p2·d2 - p3·d3, P向量: wide_base[0.6,0.1,0.3], satellite[0.2,0.6,0.2], fixed_income[0.5,0.0,0.5] | 标量score | 概念排序 |
| 9.3 固定插槽 | 概念分类映射 | `CONCEPT_CATEGORY_MAP` → hedging(黄金→518880) + cash(货币→511850) 固定不参与评分 | 固定ETF代码 | 权重分配 |

### 10. synthesis/ — 合成资产构建 (协方差权重)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 10.1 协方差计算 | `returns_5d` DataFrame (5资产×T日) | `CovarianceWeighter.compute_covariance()` → 日收益协方差 ×252 年化 | 5×5协方差矩阵 (np.ndarray) | 组合优化 |
| 10.2 等权基准 | — | `equal_weight()` → [0.2, 0.2, 0.2, 0.2, 0.2] | 5维向量 | 优化回退 |

### 11. schedules/ — WFO Walk-Forward 调度器 (季度重训 + 周频推断)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 11.1 冷启动 Phase1 | AkShare公网数据 (2019.01-2020.12, 104周) | `Phase1Initializer.run()` → 104周→25维特征→滚动归一化→MSE训练AE(50 epochs, Adam lr=1e-3) | `RegimeAutoEncoder` (基座模型) | 11.2 Phase2 |
| 11.2 冷启动 Phase2 | Phase1的AE模型 + 历史数据 | `Phase2MADCalibrator.run()` → 用Phase1模型在历史数据上计算AE误差→75%分位数波动率过滤→铸造MAD标尺(median基准+安全边际) | `(Vol_filtered_model, median基准, mad_safe)` | 11.3 并网 |
| 11.3 季度重训 | 季度末日期 (3/6/9/12月末) | `DualTrackTrainer.train_quarter()` → 拉取过去104周AkShare数据→75%分位数波动率过滤离群值→重置权重→重训AE(30 epochs)→保存 | `ae_weights_{year}Q{quarter}.pth` | WFO回测 |
| 11.4 周频推断 | 当前周五日期 | `WeeklyInferrer.infer()` → 单周数据→AE前向→计算E_t_raw | `E_t_raw` (float) | PanicIndexOutput, StateAssembler |
| 11.5 调度编排 | WFOScheduler + 当前日期 | 判断是否季度末→触发重训; 每周末→触发周频推断 | E_t_raw (float) | WFO回测主循环 |

### 12. training/ — 训练流程编排 (冷启动 + 低频轨)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 12.1 Phase1初始化 | config + device | `Phase1Initializer`: 拉取AkShare→25维特征→滚动归一化→MSE训练AE(50 epochs) | `RegimeAutoEncoder` (收敛权重) | Phase2 |
| 12.2 低频轨重训 | 季度末日期 + config | `DualTrackTrainer.train_quarter()` → 104周数据→25维特征→75%分位数波动率过滤→权重重置→重训→保存 | `.pth` 权重文件 | WFO调度器 |

### 13. data_pipeline/ — 数据管道 (ClickHouse + AkShare)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 13.1 Track B ETF日频 | ClickHouse `etf.etf_day`: close/open/adj_factor | `fetch_track_b()` → SQL查询5只ETF(510300/159919/511010/518880/159985)→pivot(code×date)→CODE_MAP映射为资产代码 | DataFrame [date × 5资产] (单列) 或 [date × 15列] (多列) | asset_features计算 |
| 13.2 Track A 其他数据 | AkShare API (公网数据,物理隔离ClickHouse) | `TrackAFetcher.fetch_weekly()` → 周频宏观/资产数据 | DataFrame [date × feature] | 低频轨训练 |

### 14. penetration/ — 实盘信号下发 (AgentBase 格式)

| 步骤 | 输入数据 / 来源 | 核心处理 | 输出数据 | 下游消费者 |
|------|-----------------|---------|---------|-----------|
| 14.1 格式转换 | etf_weights dict (slot→weight) + etf_codes dict (slot→code) + current_date | `AgentBaseFormatter.format()` → {current_date: {etf_code: weight, ...}} | 嵌套dict (AgentBase标准格式) | 实盘交易系统 |

---

## 端到端流水线数据流

### Step 1: 数据 ETL (`scripts/run_data_etl.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 数据拉取 | ClickHouse `etf.etf_day` + start/end日期 | `fetch_track_b_safe()` SQL查询 | price_df [date × 5资产] |
| 资产特征 | price_df | `compute_asset_features()` 5资产×4特征 | 20维 DataFrame |
| 宏观特征 | akshare API 5个数据源 | `compute_macro_features()` 5个宏观指标 | 5维 DataFrame |
| 时间对齐 | 20维 + 5维 DataFrame | `pd.concat` 按date索引交集对齐 | 25维特征矩阵 (T×25) |
| 标准化 | 25维特征矩阵 | `normalize_dataframe()` 滚动Z-score [t-252, t-1] | **`features_master.parquet`** |
| 增量检测 | 已有parquet + 新start_date | 增量模式: 跳过已有日期, 仅计算新数据 | 全量parquet (追加) |

### Step 2: LLM 批量打分 (`scripts/run_llm_batch.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 周频切片 | start_week → end_date | 构建每周五日期列表 | fridays list |
| 断点过滤 | SQLite已完成周 set | 跳过已打分周, 仅处理pending | pending fridays |
| 文本ETL | ClickHouse text_db (zgrmyh/csrc/govcn/news) | `TextETL.extract_per_concept()` 30天窗口 | etl_data dict |
| Prompt构建 | etl_data + 上周prior_scores | `PromptBuilder.build_equity/fixed_income()` | 2个Themed Prompt |
| 并发打分 | 2个Prompt + AsyncOpenAI | `asyncio.gather` 2路并发 + 信号量限流 | JSON评分 |
| 解析校验 | LLM JSON | `ResponseParser.parse()` 验证d1/d2/d3∈[1,100] | dict[concept][d1/d2/d3] |
| 断点写入 | 评分结果 + week_end | `INSERT OR REPLACE INTO llm_scores` | **`llm_scores.db`** (SQLite) |

### Step 3: AE 自编码器训练 (`scripts/train_ae.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 数据加载 | `features_master.parquet` | `pd.read_parquet` → 去NaN | X shape (N, 25) |
| 数据划分 | X (N,25) | 随机80/20 split (seed=42) | train_loader + val_loader |
| 模型初始化 | input_dim=25, latent=6, hidden=16 | `RegimeAutoEncoder` 随机初始化 | model |
| 训练循环 | train_loader + val_loader | MSE Loss + Adam(lr=1e-3) + ReduceLROnPlateau + EarlyStopping(patience=10) | best model |
| 权重保存 | best model + scaler (mean/std/columns) | `torch.save` + `pickle.dump` | **`ae_weights.pth`**, **`ae_scaler.pkl`** |

### Step 4: PPO 元控制器训练 (`scripts/train_ppo.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 历史数据准备 | `features_master.parquet` + `llm_scores.db` + `ae_weights.pth` + `ae_scaler.pkl` | `inject_live_data_from_history()` → 预计算每个时间步: ae_error, vol, llm_macro/sentiment/risk, DualTrackEngine(w_normal, w_event), port_returns, equity_curve, rolling_sharpe, mdd, returns_window_5d | `live_data_list` (T个dict) |
| LLM日频填充 | llm_scores_df (周频) | smart_agg: d1/d2全市场均值, d3=宽基均值×0.7+satellite最大×0.3 → ffill到日频 | 日频LLM评分 |
| 环境初始化 | config | `MDPEnvironment` (Gymnasium): state_dim=10, action_dim=2, episode_max_steps=252 | env |
| 网络初始化 | state_dim=10, action_dim=2 | `ActorCritic` 正交初始化, Actor(10→64→64→2) + Critic(10→64→64→1) | ac |
| 训练循环 | env + ac + buffer(size=100) | for N updates: collect_rollout(T=100步, 每步注入live_data) → compute_GAE(γ=0.99, λ=0.95) → 4 epochs mini-batch SGD(Adam lr=3e-4, clip_ε=0.2) → TensorBoard打点 | updated ac |
| Checkpoint | ac + optimizer + step_count | 每10 iter + 最终保存 | **`actor_critic.pth`** |

### Step 5: Walk-Forward 回测 (`scripts/run_backtest_wfo.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 冷启动 | WFOScheduler + config | Phase1(104周基座AE训练) + Phase2(52周MAD标尺铸造) | 基座模型 + MAD参数 |
| WFO主循环 | features_master + llm_scores + ae_weights + actor_critic + 周五列表 | 每周: WFO调度(E_t)→双轨权重(w_normal, w_event)→PPO Actor前向(α决策)→融合(w_final = α·w_event + (1-α)·w_normal)→Regret计算→Veto检查→净值更新 | records list (NAV, α, τ, regret, weights) |
| 季度重训 | 季度末周五 (3/6/9/12月末) | `DualTrackTrainer.train_quarter()` → 104周AkShare数据→重训AE | ae_weights_Q.pth |
| 指标计算 | NAV序列 | `compute_wfo_metrics()` → 总收益/年化收益/最大回撤/Sharpe/年化波动率 | metrics.json |
| 输出落盘 | records + gate_diagnostics | nav_series.csv + weights_trajectory.csv + weights_data_generalbt.csv + metrics.json + gate_diagnostics.csv + tearsheet.txt + figures/ | **`results/wfo/{run_id}/`** |
| GeneralBacktest | weights_data + price_data(ClickHouse日频OHLC) | 集成GeneralBacktest: run_backtest(adj_factor, tcost=0.0003, slippage=0.0001) + plot_dashboard | 业绩图 (nav/excess/drawdown/heatmap) |

### Step 6: 实盘信号下发 (`scripts/run_inference_live.py`)

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 增量ETL | ClickHouse (最新2周) | `fetch_track_b_safe()` | 最新ETF价格 |
| 单周LLM | 当前周五 + AsyncSemanticEngine | 单次 `engine.evaluate(week_end)` | llm_scores dict |
| 模型加载 | `ae_weights.pth` + `ae_scaler.pkl` + `actor_critic.pth` | torch.load → eval模式 | ae_model + ac_model |
| 状态组装 | ae_error (AE前向) + vol_mkt_20d + llm_scores (聚合d1/d2/d3) + regret_ema | `StateAssembler.assemble()` → 10维S_t | S_t shape (10,) |
| Actor前向 | S_t tensor | `ac.actor(S_t)` → action_mean → ActionMapper → (Δα, Δτ) → α_new, τ_new | α_new (float), τ_new (float) |
| 权重融合 | α_new + DualTrackEngine.compute() → w_normal, w_event | w_target = α·w_event + (1-α)·w_normal → clip → normalize | w_target shape (5,) |
| 信号下发 | w_target + week_end + α + τ + ae_error | JSON格式化 → `target_weights_{date}.json` + `target_weights_latest.json` (软链接) | **`results/wfo/target_weights_*.json`** |
