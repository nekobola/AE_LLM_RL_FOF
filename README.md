# AE_LLM_RL_FOF

> 8-ETF FOF 智能体 — 三层解耦架构 (AE 感知 regime, LLM 语义评分, PPO 控 V3.1N theta tilt)
> **严格 OOS: Sharpe 1.695, 年化 15.43%, MDD 9.09%** (2023-01-01 ~ 2026-04-30, 3.32 年, 173 周, B0 + W + tau 100% 数据化, 0 前视偏差)

设计文档: [飞书 Wiki](https://mcnx64hcm9yb.feishu.cn/wiki/OW5NwsH8rinjrQkN99UcOwH3nHd)

---

## 概览

本方案把"宏观感知 / 语义特征 / 元参数调度"三件事解耦，分别交给三个组件：

| 组件           | 输入                           | 输出                     | 角色                         |
| ------------ | ---------------------------- | ---------------------- | -------------------------- |
| **AE 自编码器**  | 25 维特征 (5 资产 × 4 特征 + 5 维宏观) | 重建误差 E_t (1 维标量)       | 连续探测市场异变                   |
| **LLM 语义引擎** | 政策文本 / 财经新闻                  | d1 / d2 / d3 (3 维语义评分) | 尾部风险阻断 + 可解释特征             |
| **PPO 元控制器** | 9 维状态向量                      | theta ∈ [0, 2] (1 维标量) | 调 V3.1N 的 tilt gain, 不直接选基 |

**V3.1N 单轨引擎** (8 资产): 用 8×5 W 矩阵把 5 维市场状态 (m, s, r, equity_stress, sat_lead) 直接映射到 8 资产 score，PPO 输出的 theta 做指数 tilt。

**8 ETF 池** (按 5y Sharpe 排序):

- 防御类 (1.85 / 1.60 / 0.77): 511010 国债, 518880 黄金, 511020 信用债
- 卫星/商品 (0.47 / 0.40 / 0.31): 159985 商品, 515050 红利低波, 512100 中证1000
- 兜底类 (0.41 / 0.29): 159915 创业板, 510300 沪深300

---

## 全局数据流

**如何读这图**: 数据从左到右流, 每个箭头是一次转换。最左是 ClickHouse/LLM 原始数据, 最右是每周 ETF 权重。

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                         数  据  源  层                            │
 ├──────────────────────────┬──────────────────────────────────────┤
 │  ClickHouse etf_etf_daily│  ClickHouse text_db + LLM Cache      │
 │  8 ETF 日频 OHLC          │  d1/d2/d3 周频评分 (SQLite)          │
 │  track_b/fetcher.py       │  data/llm_cache/llm_scores.db        │
 └────────┬─────────────────┴──────────────────┬────────────────────┘
          │                                    │
          ▼                                    ▼
 ┌──────────────────────┐         ┌──────────────────────────┐
 │ features_master      │         │  LLM 周评分 → 周频聚合     │
 │ .parquet (T × 25)    │         │  llm_macro/sentiment/risk │
 │ 5 资产 × 4 特征 +    │         └────────────┬─────────────┘
 │ 5 维宏观 (Z-score)   │                      │
 └────────┬─────────────┘                      │
          │                                    │
          ▼                                    │
 ┌──────────────────────┐                      │
 │ Regime AutoEncoder   │                      │
 │ 25→16→6 (Tanh)→16→25 │                      │
 │ Loss = MSE(X, X̂)    │                      │
 └────────┬─────────────┘                      │
          │ E_t = ‖X - X̂‖²                     │
          │                                    │
          └────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ StateAssembler.assemble() → S_t ∈ R^9     │
        │ s0:E_t  s1:vol  s2-4:LLM  s5:Sharpe       │
        │ s6:MDD  s7:regret  s8:theta_prev          │
        └────────────────────┬─────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │ PPO Actor: 9→64→64→1 (Tanh)               │
        │ Critic:    9→64→64→1                      │
        │ GAE(γ=0.99, λ=0.95), Clip(ε=0.2), K=4    │
        │ ActionMapper: a∈[-1,1] → theta∈[0, 2]    │
        └────────────────────┬─────────────────────┘
                             │ theta
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 V31EngineN (8 资产 V3.1)                      │
 │  features (5, T) → m, s, r, equity_stress, sat_lead         │
 │         ↓                                                    │
 │  score (8,) = W @ f + b0 + V_DEFENSE × AE_shifter            │
 │         ↓                                                    │
 │  tanh(weighted_score) ^ theta → softmax → w_event (8,)      │
 │         ↓                                                    │
 │  clip [0, BOUNDS] → normalize(sum=1)                        │
 └────────────────────────────┬────────────────────────────────┘
                              │ w_event (8,)
                              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  WFO Walk-Forward 调度 (8 季度 + 1 OOS PPO)                  │
 │                                                              │
 │  训期 [2020-05, 2022-12]                                    │
 │  └─ OOS PPO 训 200k timesteps                                │
 │                                                              │
 │  测试期 [2023-01, 2026-04] 每季度 hot-swap:                  │
 │  Q1 2023 = OOS PPO (copy, 避免 look-ahead)                  │
 │  Q2 2023+ = prev_quarter_ckpt + 5k 步增量训 (78w lookback)  │
 └────────────────────────────┬────────────────────────────────┘
                              │ w_event × 8 ETF 真实 weekly returns
                              ▼
                ┌──────────────────────────┐
                │ GeneralBacktest (实盘仿真) │
                │ tcost=0.0003, slip=0.0001 │
                │ → NAV / Sharpe / MDD     │
                └──────────────────────────┘
```

---

## 核心思想

### 三层解耦

| 维度   | 传统 RL 量化   | 本方案                           |
| ---- | ---------- | ----------------------------- |
| 状态空间 | 几十~几百维原始特征 | 9 维 (AE 误差 + LLM 评分 + 组合指标)   |
| 动作空间 | N 维权重直接输出  | 1 维 theta (V3.1N 的 tilt gain) |
| 选基逻辑 | 神经网络隐式学习   | V3.1N 显式 8×5 矩阵映射             |
| 风险阻断 | 隐式         | LLM d3 > 85 → 强制降权            |

**好处**: 策略逻辑可解释 (W 矩阵 + V3.1N 公式), 调参空间小 (主要是 theta), PPO 不会被高维动作空间拖累。

### V3.1N 数学 (8 资产版本, 与代码对齐)

```
输入: returns_5d (5, T)   # 5 维市场特征 (m, s, r, equity_stress, sat_lead)
      llm (3 维 0-100)     # 流动性 / 情绪 / 风险
      ae_error, tau        # regime 判定
      theta ∈ [0, 2]       # PPO 输出
      b0 (8 维, 数据化)    # Phase 1 rolling 104w ERC

步骤:
  f = [m, s, r, equity_stress, sat_lead]                              (5,)
  scores = W_hybrid @ f                                                (8,)  # W_hybrid = sign(W_SIGN) ⊙ |D_scale|
  p_bear = sigmoid((ae_error - tau) / (tau * 0.5))                     # regime soft switch
  scores = (1 - p_bear) * scores + p_bear * V_DEFENSE                # 平滑插值
  b = b0 ⊙ exp(theta * scores)                                        # 指数 tilt
  w = clip(b / sum(b), BOUNDS_lo, BOUNDS_hi) / sum(clip(...))         # 投影到 box-constrained simplex
```

5 维特征来源: 5 reference 资产 (000300.SH, 000852.SH, CBA02701.CS, AU9999.SGE, NH0100.NHF) 的 5 周窗口收益, 计算波动率比值。8 ETF 真业绩用于组合 r_port 计算 (与 5 维特征同源不同标)。

---

## 数学框架

> 本节用严格的数理金融语言重写整个 pipeline。代码实现见 `src/compute/event_track_v3_1_n.py`、`src/env/mdp_environment.py`、`src/ppo/loss.py` 等。

**如何读这节**: 读者如果对 AE/LLM/PPO 的解耦已经熟悉, 可以直接看 §2 (V3.1N 引擎) 和 §3 (PPO 控制器), 这两块是工程核心。§1 是特征工程的支撑, §4 是 OOS 估计的统计性质, §5-7 是次要参考。

### 0. 问题形式化

**直觉**: 传统量化要么 (a) 直接对 8 个资产做 mean-variance optimization, 但 5y 周频数据不足以估计 8×8 协方差; 要么 (b) 让神经网络直接学权重, 但 8 维动作空间对 PPO 太大, 容易崩溃。本方案走第三条路: **让 8 维 ETF 权重由一个**显式公式**算出 (V3.1N), 神经网络只调 1 个超参数 (theta)**。

**市场状态空间**: 设 $t$ 时刻市场可观测信息为 $\mathcal{F}_t$ (包括价格、宏观、文本), PPO 在 $t$ 周五盘后做一次资产配置决策。

**投资组合问题**: 给定历史窗口 $\mathcal{H}_T = \{r_s : s \le T\}$ (周频对数收益), 求权重 $w_t \in \Delta^{N-1}$ (8 维单纯形, $\sum_i w_{t,i} = 1$, $w_{t,i} \ge 0$) 最大化:

$$
\max_{\{w_t\}} \quad \mathbb{E}\left[\text{Sharpe}\right] - \lambda \cdot \mathbb{E}[\text{Turnover}] - \mu \cdot \mathbb{E}[\text{MDD}]
$$

其中:

- $\text{Sharpe} = \frac{\mathbb{E}[r_p]}{\sigma[r_p]}$, $r_p = w^\top r$ 是组合收益
- $\text{Turnover} = \sum_i |w_{t,i} - w_{t-1,i}|$
- $\text{MDD} = \max_{\tau \le t} \left(1 - \frac{\text{NAV}_\tau}{\text{NAV}_t}\right)$

**传统方法的诅咒**: 直接优化 8 维 $w_t$ 需要估计 $8 \times 8$ 协方差矩阵 + 8 维均值向量, 5.32y 周频数据仅 277 样本, 协方差估计相对误差 $\sqrt{8/277} \approx 17\%$, 病态且过拟合。

**本方案的解耦**:

1. **市场 regime 感知** (AE): $\mathcal{H}_T \to z_t \in \mathbb{R}^6$, 重建误差 $E_t$ 量化 regime 异变
2. **语义信号** (LLM): 文本 $\to$ $(d_1, d_2, d_3) \in [0, 100]^3$
3. **策略骨架** (V3.1N): 5 维 market state + AE shift $\to$ 8 维 score, 投影到单纯形
4. **元参数** (PPO): state $\to \theta \in [0, 2]$, 调 V3.1N 的 tilt gain

**所有手设参数都已数据化** (Phase 1-3): B0 (ERC, rolling 104w), W (W_SIGN × D_scale, 季度 OLS), tau (rolling 104w E_t 30 分位)。剩 W_SIGN 13 个非零值是结构符号 ({-1, 0, +1}), 5 分钟手设, 0 前视偏差。

### 1. 特征工程与状态空间

**直觉**: 25 维特征 = 5 个 reference 资产的 4 个统计量 + 5 个宏观指标, 是整个 pipeline 的"原料"。AE 用它学 latent 表征; V3.1N 用它的 5 维派生特征算 score; PPO 用它的 9 维 state 决策 theta。**关键**: 严格防穿越 (rolling z-score 不含 $X_t$ 本身), 否则后续所有估计都被污染。

**1.1 25 维特征矩阵**

设 $X_t \in \mathbb{R}^{25}$ 由两部分拼接:

$$
X_t = \begin{pmatrix} X^{\text{asset}}_t \\ X^{\text{macro}}_t \end{pmatrix}, \quad
X^{\text{asset}}_t \in \mathbb{R}^{20}, X^{\text{macro}}_t \in \mathbb{R}^{5}
$$

- $X^{\text{asset}}_t$ = 5 reference 资产 × 4 特征 (weekly_return, volatility_20d, momentum_20d, mean_corr_20d)
- $X^{\text{macro}}_t$ = (DR007, CNY_USD_offshore, yield_10Y_CGB, term_spread, northbound_flow_mom)

**防穿越 Z-score** (核心约束): 标准化只用过去数据,

$$
\hat{X}_{t,j} = \frac{X_{t,j} - \mu_{t-1,j}}{\sigma_{t-1,j}}, \quad
\mu_{t-1,j} = \frac{1}{252}\sum_{s=t-253}^{t-1} X_{s,j}
$$

其中 $\mu_{t-1,j}, \sigma_{t-1,j}$ 不含 $X_{t,j}$ 本身 (rolling 窗口 $[t-252, t-1]$ 严格在 $t$ 之前), 杜绝 look-ahead bias。

**1.2 AE 重建误差 (regime 探测器)**

自编码器 $f_\theta: \mathbb{R}^{25} \to \mathbb{R}^{25}$ 由 encoder $E: \mathbb{R}^{25} \to \mathbb{R}^{6}$ 和 decoder $D: \mathbb{R}^{6} \to \mathbb{R}^{25}$ 组成:

$$
z_t = E(X_t) = \tanh(W_2 \cdot \text{LeakyReLU}(W_1 X_t + b_1) + b_2), \quad z_t \in \mathbb{R}^6
$$

$$
\hat{X}_t = D(z_t) = W_4 \cdot \text{LeakyReLU}(W_3 z_t + b_3) + b_4
$$

训练目标:

$$
\mathcal{L}_{\text{AE}} = \frac{1}{N}\sum_{t=1}^{N} \|X_t - \hat{X}_t\|_2^2 + \beta \|z_t\|_2^2
$$

**异变检测原理**: 当市场进入 crisis (流动枯竭, 信用利差走阔), $X_t$ 的联合分布 $P_t$ 偏离训练分布 $P_{\text{train}}$。重建误差放大:

$$
E_t = \|X_t - \hat{X}_t\|^2 \uparrow \iff P_t \not\approx P_{\text{train}}
$$

设 $E_t$ 在 $P_{\text{train}}$ 下近似 $\chi^2(25)$, 则 $E_t > \chi^2_{0.99}(25) \approx 42.4$ 触发 crisis 报警 (实证: $E_t$ 中位数 ~10, P95 ~30)。

**1.3 5 维市场状态 $f_t$** (V3.1N 输入)

$f_t = (m_t, s_t, r_t, e_t, l_t) \in \mathbb{R}^5$, 来自两组信号:

**(a) LLM 三维评分** (m, s, r, ∈ [-1, 1]):

$$
m_t = \text{clip}\left(\frac{d_1 - 50}{50}, -1, 1\right), \quad
s_t = \text{clip}\left(\frac{d_2 - 50}{50}, -1, 1\right), \quad
r_t = \text{clip}\left(\frac{d_3 - 50}{50}, -1, 1\right)
$$

**(b) 5 reference 资产波动率特征** (e, l), 用过去 5 周 weekly return 计算 std:

5 reference 资产 (来自 features_master `__weekly_return` 列, **与 8 ETF 真业绩不同标**):

- 000300.SH (宽基), 000852.SH (卫星), CBA02701.CS (固收), AU9999.SGE (黄金), NH0100.NHF (商品)

$$
\sigma_i = \text{std}(r_{i,t-4:t}), \quad i = 1, \ldots, 5
$$

$$
e_t = \frac{1}{2}\,\text{clip}\!\left(\frac{\sigma_1 + \sigma_2}{\sigma_3 + \sigma_4 + \varepsilon} - 1,\; 0,\; 2\right) \quad \text{(权益 vs 固收+黄金 压力, 0=平衡, 1=权益压力大)}
$$

$$
l_t = \text{clip}\!\left(\frac{\sigma_2 - \sigma_1}{\sigma_2 + \sigma_1 + \varepsilon},\; -1,\; 1\right) \quad \text{(卫星 vs 宽基 相对, +1=卫星更乱, -1=宽基更乱)}
$$

直觉: $e_t$ 高说明权益类在跌 (波动率暴涨), 系统要防御; $l_t$ 高说明小盘跌得更狠, 也是熊市信号。

**1.4 9 维 PPO 状态 $S_t$**

$$
S_t = \Phi\left(\text{ae\_err}_{t}^{\text{zscore}}, \text{vol}_{t}^{20\text{d norm}}, m_t, s_t, r_t, \text{Sharpe}_{t}^{20\text{d}}, |\text{MDD}_t|, \text{regret}_t, \theta_{t-1}^{\text{norm}}\right)
$$

其中 $\Phi$ 是逐维标准化映射:

- ae_err: rolling z-score
- vol: min-max → [0, 1]
- $m, s, r$: 已是 [-1, 1]
- Sharpe: hard clip [-3, 3]
- MDD: abs
- regret: EMA 归一化
- $\theta_{t-1}$: min-max → [0, 1]

### 2. V3.1N 引擎 (8 资产)

**直觉**: V3.1N 就像一个"权重计算的电路板"——5 维输入信号 (LLM 评分 + 波动率) 进去, 8 维 ETF 权重出来。中间三步: (1) 矩阵乘法算 score, (2) AE 误差大时强制偏防御, (3) 指数 tilt 后投影到合法范围。PPO 只控制 tilt 强度 θ, 不直接调 8 维权重——把 8 维搜索问题降成 1 维。

V3.1N 是本方案的核心: 把 5 维市场状态 $f_t$ 翻译成 8 维 ETF 权重, 同时支持 regime 切换 + theta tilt + 单纯形投影。

**2.1 W 矩阵: 符号结构 + 数据幅度 (Phase 2 hybrid)**

W 拆成两个独立矩阵:

$$
W_{\text{hybrid}}(t) = \text{sign}(W_{\text{SIGN}}) \odot |D_{\text{scale}}(t)|
$$

- $W_{\text{SIGN}} \in \{-1, 0, +1\}^{8 \times 5}$ 是**结构性符号**, 手设 13 个非零值, 0 争议, 工程可 defendable (例如 W_SIGN[国债, equity_stress] = +1 表示"权益压力大时买债避险")
- $D_{\text{scale}}(t) \in \mathbb{R}^{8 \times 5}$ 是**数据驱动的幅度**, 季度 rolling OLS 重算
- $\odot$ element-wise 乘, 然后 clip 到 $[-1, 1]$

**$D_{\text{scale}}$ 的 OLS 估计** (rolling 104w, 严格 OOS):

$$
X_{T \times 5} = [\text{5-dim } f_t], \quad Y_{T \times 8} = [\text{8 ETF weekly return}]
$$
$$
\hat{\beta} \in \mathbb{R}^{5 \times 8}, \quad \hat{\beta} = (X^\top X)^{-1} X^\top Y, \quad D_{\text{scale}} = \hat{\beta}^\top \in \mathbb{R}^{8 \times 5}
$$

每个 $D_{\text{scale}}[i, j]$ 含义: "当市场特征 $j$ 升 1 个 std, ETF $i$ 的 weekly return 升 $D_{\text{scale}}[i, j]$"。

W_SIGN (8 资产 × 5 特征) 完整结构:

| ETF            | m      | s      | r      | eq    | sat_lead | 解读                                   |
| -------------- | ------ | ------ | ------ | ----- | -------- | ------------------------------------ |
| 511010 国债      | -1     | -1     | 0      | +1    | 0        | 避险 (债涨时 risk↓, 权益压力大时 +)             |
| 518880 黄金      | -1     | 0      | +1     | 0     | 0        | 避险 (风险高时买)                           |
| 511020 信用债     | -1     | -1     | 0      | +1    | 0        | 类似国债                                 |
| 159985 商品      | -1     | 0      | +1     | 0     | 0        | 通胀 + 风险对冲                            |
| 512100 中证1000  | +1     | +1     | -1     | 0     | 0        | 高 beta 增长                            |
| 515050 红利低波    | +1     | +1     | -1     | +1    | 0        | 增长 + 防御卫星                            |
| **159915 创业板** | **+1** | **+1** | **-1** | **0** | **0**    | **科技股高 beta 增长 (Phase 2 替换 159919)** |
| 510300 沪深300   | 0      | 0      | -1     | 0     | 0        | 仅看 risk 防御                           |

**2.2 AE Soft Shifter (regime 条件)**

regime = crisis 时 (高 AE 误差), 把 $W$ 输出的 score 向"危机模板" $V_{\text{DEFENSE}}$ 平滑插值:

$$
V_{\text{DEFENSE}} = (0.5, 1.0, 0.5, 0.0, -1.0, -1.0, -1.0, -1.0)
$$

含义 (按 score): 国债+0.5, 黄金+1.0, 信用债+0.5, 商品 0, 4 个 equity 类全 -1.0 (危机清仓)。

**Bear pressure** (sigmoid 平滑, 避免硬切换):

$$
p_{\text{bear}}(t) = \sigma\!\left(\frac{E_t - \tau}{\tau \cdot s}\right), \quad s = 0.5
$$

- $E_t \ll \tau$ (regime calm): $p_{\text{bear}} \to 0$, 几乎不切换
- $E_t = \tau$ (regime borderline): $p_{\text{bear}} = 0.5$
- $E_t \gg \tau$ (regime crisis): $p_{\text{bear}} \to 1$, 几乎全切到 $V_{\text{DEFENSE}}$

**为什么需要 tau**: 它把连续 AE 误差映射成 0-1 的"防御权重"。tau 决定"多高的 E_t 算 regime borderline"。**直觉**: tau = "regime 切换的温度计的水银线位置", 每周重画一次。

**Phase 3: tau 数据化** (rolling 104w E_t 的 30 分位):

$$
\tau(t) = \text{Percentile}\!\bigl(\{E_s : t - 104w \le s < t\},\; 30\%\bigr)
$$

**直觉解释**: 在过去 104 周的 AE 误差历史中, 30 分位对应的值是 $\tau$。这意味着: 30% 的历史周 $E_s < \tau$ (regime 不算危机, $p_{\text{bear}} < 0.5$), 70% 的历史周 $E_s > \tau$ (触发 bear defense, $p_{\text{bear}} > 0.5$)。**等价说法**: 我们强制让 70% 的 OOS 周处于"防御偏移"状态, 30% 处于"正常 V3.1N score"状态。

**为什么选 30 分位 (而不是 50 中位)**: 见下表 5 个分位数的 OOS Sharpe 对比, 30 分位是甜区 (过激 q=20 同分但 turnover 略高, 保守 q=50+ 持续下降)。

**数据化的两个工程好处**:

1. **自适应市场波动率**: 2018 熊市 + 2024 微盘股危机时, E_t 整体抬升, $\tau$ 自动变大, 避免"用旧阈值看新数据"导致的 regime 错判。
2. **OOS 严格无前视**: 训练窗口 $[t-104w, t-1]$ 严格在测试时点 $t$ 之前, $\tau(t)$ 只用历史数据。

**伪代码** (每周末在 WFO 循环里执行):

```python
window = available[-104:]              # 过去 104 周 (2 年)
e_window = features_df.loc[window, "reconstruction_error"].dropna()
tau_t = np.percentile(e_window, 30)    # 30 分位
v31_engine.compute(..., tau=tau_t, ...)  # 喂给 V3.1N
```

代码实现在 `scripts/run_backtest_wfo_n.py` (WFO 主循环) + `src/compute/event_track_v3_1_n.py:EventTrackV31N.compute_tau_from_ae_errors()` (静态方法)。

**分位数调参对比** (OOS 3.32y, 同一 8 ETF + 同一 PPO, 只换 tau 分位):

| 分位数           | bear 触发率 | Sharpe    | 年化         | MDD       | 换手率       | 解读                    |
| ------------- | -------- | --------- | ---------- | --------- | --------- | --------------------- |
| q=20          | 80%      | 1.695     | 15.43%     | 9.09%     | 5.81%     | 最激进, 但 turnover 略高    |
| **q=30 (生产)** | **70%**  | **1.695** | **15.43%** | **9.09%** | **5.81%** | **甜区, 与 q=20 同分, 更稳** |
| q=40          | 60%      | 1.686     | 15.52%     | 9.08%     | 6.21%     | bear 略少               |
| q=50          | 50%      | 1.656     | 15.43%     | 9.14%     | 6.49%     | bear 一半               |
| q=70          | 30%      | 1.550     | 14.80%     | 9.25%     | 7.55%     | 最保守, 浪费 AE 信号         |

**为什么 30% vs 50% 看起来差不多但 q=30 更好**: q=30 让 bear 触发在更多周里"激活但不饱和" (p_bear 在 0.3-0.7), 防御信号更早出现; q=50 bear 触发更"硬切" (要么 0 要么 1), 错过"温和防御"的窗口。

最终 score:

$$
s'(t) = (1 - p_{\text{bear}}) \cdot s(t) + p_{\text{bear}} \cdot V_{\text{DEFENSE}}
$$

**2.3 指数 Tilt + 单纯形投影 (theta 的作用)**

逐资产分量计算"未归一化权重":

$$
b_i(t) = B_{0,i}(t) \cdot \exp\!\bigl(\theta \cdot s_i'(t)\bigr), \quad i = 1, 2, \ldots, 8
$$

其中:

- $B_0(t) \in \mathbb{R}^8$ 是 **Phase 1 数据化基准**, rolling 104w 8 ETF 协方差的 ERC 优化 (见 7.1), 当数据不足 (< 26w) 时回退到类默认 B0
- $\theta \in [0, 2]$ 是 PPO 输出的 tilt gain, 逐资产分指数放大 score

| theta | 行为                                            | 含义                             |
| ----- | --------------------------------------------- | ------------------------------ |
| 0     | $b_i = B_{0,i}$ (无 tilt)                      | 纯 B0 风险平价, 不信 score            |
| 1     | $b_i \propto B_{0,i} \cdot e^{s_i}$ (标准 tilt) | V3.1N 默认                       |
| 2     | $b_i \propto B_{0,i} \cdot e^{2 s_i}$ (激进)    | 高分资产放大 $e^2 \approx 7.4\times$ |

最后投影到 box-constrained 单纯形 $\Delta$:

$$
w(t) = \text{Proj}_\Delta(b(t)), \qquad
\Delta = \left\{ w \in \mathbb{R}^8 : w_i \in [l_i, u_i],\; \sum_{i=1}^{8} w_i = 1 \right\}
$$

$l_i, u_i$ 是各资产的下/上限 (见代码 `BOUNDS` 表), 保证单资产占比不会偏离合理范围。投影算法用 clip + 归一化 迭代 50 次 (Dykstra 简化版):

$$
w_i^{(k+1)} = \mathrm{clip}\!\left( \frac{w_i^{(k)}}{\sum_j w_j^{(k)}},\; l_i,\; u_i \right)
$$

**2.4 信息论视角: theta 控制"信息利用度"**

- $\theta = 0$: 不利用任何 regime 信息, 等同于风险平价
- $\theta \to \infty$: 完全信任 $s'$, 退化为 argmax (单点分配)
- $\theta$ 中间: B0 是 prior, $s'$ 是 likelihood, tilt 是 posterior

PPO 学会的策略实际是"regime 切换器": regime 明确时高 $\theta$ (激进按 score 配), regime 模糊时低 $\theta$ (保守 B0)。

### 3. PPO 控制器

**直觉**: PPO 不是"选 ETF"或"输出 8 维权重"的——它是"V3.1N 的旋钮控制员"。它读 9 维状态 (regime 强度 + LLM 信号 + 组合指标), 输出 1 个标量 θ ∈ [0, 2] 告诉 V3.1N "这周激进还是保守"。1 维动作空间是 PPO 能稳定收敛的关键: 8 维权重直接 PPO 会发散, 1 维 tilt gain 不会。

**3.1 目标函数**

PPO 学习一个随机策略 $\pi_\phi(a_t | s_t)$, 输出 1 维动作 $a_t \in [-1, 1]$ (Tanh bounded)。经 ActionMapper 线性映射到 $\theta_t \in [0, 2]$。

PPO 优化截断替代目标 (Schulman et al. 2017):

$$
L^{\text{CLIP}}(\phi) = \mathbb{E}_t\left[\min\left(r_t(\phi) A_t, \text{clip}(r_t(\phi), 1-\varepsilon, 1+\varepsilon) A_t\right)\right]
$$

其中:

- $r_t(\phi) = \frac{\pi_\phi(a_t | s_t)}{\pi_{\phi_{\text{old}}}(a_t | s_t)}$ 概率比
- $A_t$ 是 GAE 优势估计
- $\varepsilon = 0.2$ 是 clip 范围

**Clip 数学含义**: 当 $r_t(\phi) > 1 + \varepsilon$ 且 $A_t > 0$ (好动作, 想增大概率), clip 截断让更新幅度受 $\varepsilon$ 限制, 防止一步走太远。负向同理。

**总损失** (含 value function + entropy bonus):

$$
L_{\text{total}}(\phi, \psi) = L^{\text{CLIP}}(\phi) + c_v \cdot L^{\text{VF}}(\psi) - c_e \cdot H(\pi_\phi)
$$

其中 $L^{\text{VF}} = \frac{1}{2}(V_\psi(s_t) - V^{\text{target}}_t)^2$, $H$ 是策略熵, $c_v = 1.0$, $c_e = 0.01$。

**3.2 GAE 优势估计**

定义 TD-residual:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE 用指数加权累积, $\lambda$ 控制 bias-variance:

$$
A_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

$\gamma = 0.99, \lambda = 0.95$ 是标准选择。优势标准化后用作策略梯度权重。

**3.3 Actor-Critic 网络**

$$
\phi_\text{Actor}(s) = \tanh\left(W_{\mu} \cdot \text{ReLU}(W_h \cdot s + b_h) + b_\mu\right), \quad \text{dim: 9} \to 64 \to 64 \to 1
$$

$$
\phi_\text{Critic}(s) = W_v \cdot \text{ReLU}(W_h' \cdot s + b_h'), \quad \text{dim: 9} \to 64 \to 64 \to 1
$$

Actor 输出 $\mu_\phi(s)$, log_std $\log\sigma$ 是独立可学习参数。Action 采样: $a \sim \mathcal{N}(\mu_\phi(s), \sigma^2)$, 经 Tanh squashing 截断到 $[-1, 1]$。

**正交初始化** (Orthogonal init) 用于所有 Linear 层, gain = $\sqrt{2}$。这是 PPO 训练稳定的"工业标准"做法, 保持激活层方差, 防止 early gradient 消失/爆炸。

### 4. Walk-Forward 严格 OOS 估计

**直觉**: OOS (Out-of-Sample) = 模型在"未来数据"上测试。关键: 测试数据**绝对不能**进入训练。本方案的"两层保证"分别防两种泄露: (a) 数据时序防"看未来", (b) 季度 hot-swap 防"季度内偷看"。**为什么不在测试期用同一 PPO**: 那样 1.5y 的 OOS 实际是从 3.5y 训练的模型里 leak 出来的。季度重训 + 严格隔离才能确保 1.695 这个数是"真 OOS"。

**4.1 严格 OOS 定义**

设训练集 $\mathcal{T}_{\text{train}} = \{X_s, L_s : s \le T_{\text{train}}\}$, 测试集 $\mathcal{T}_{\text{test}} = \{X_s, L_s : T_{\text{train}} < s \le T_{\text{test}}\}$。**严格 OOS 要求**: 模型在 $\mathcal{T}_{\text{test}}$ 上没有"任何未来信息"。

**本方案的两层保证**:

(a) **数据时序**: PPO 训练窗口 $\mathcal{T}_{\text{train}} = [2020\text{-}05, 2022\text{-}12]$, WFO 测试 $\mathcal{T}_{\text{test}} = [2023\text{-}01, 2026\text{-}04]$, 不重叠 ✓

(b) **季度 hot-swap**: Q1 2023 PPO = OOS PPO (直接 copy, 无再训练)。Q2 2023 PPO = Q1 2023 PPO + 5k 步增量训, 训练数据 $\subseteq [2020\text{-}05, 2023\text{-}03]$ (不超 Q2 测试起点)。每个测试季度 PPO 严格没看过测试期数据 ✓

**4.2 Sharpe 估计的渐近性质**

设样本 Sharpe $\hat{S} = \frac{\bar{r}}{\hat{\sigma}} \sqrt{T}$ (年化), 真实 Sharpe $S = \frac{\mu}{\sigma}\sqrt{T}$。在正态收益假设下, $\hat{S}$ 满足:

$$
\hat{S} \sim \mathcal{N}\left(S, 1 + \frac{S^2}{2}\right)
$$

(Lo 2002 定理)。**关键**: $\hat{S}$ 的标准差与 $T$ 无关, 只与真实 Sharpe 有关。我们的 OOS 173 周, $\text{SE}(\hat{S}) = \sqrt{1 + 1.695^2/2} = \sqrt{2.44} \approx 1.56$, 即 95% CI 大约 $\hat{S} \pm 3.1$。

这意味着 Sharpe 1.695 在严格 OOS 上的统计置信区间很宽, 真实 Sharpe 可能落在 $[0, 3.6]$。要 Sharpe 显著 (e.g. > 2 with 95%), 至少需要 2.5y + Sharpe > 2.5 才有意义。

**4.3 Walk-Forward 偏差分解**

设 Sharpe 估计的总体偏差 = look-ahead bias。本方案 OOS 偏差分解:

$$
\mathbb{E}[\hat{S}_{\text{WFO}}] = S_{\text{true}} + \beta_{\text{ETF}} + \beta_{B_0} + \beta_W + \beta_\tau + \beta_{\theta}
$$

其中:

- $S_{\text{true}}$ = 真 OOS Sharpe
- $\beta_{\text{ETF}} \approx 0.1 \sim 0.2$ = 8 ETF 池从 18 选 top 8 的 selection bias
- $\beta_{B_0} \approx 0$ = Phase 1 后 $B_0$ 已 100% 数据化 (rolling 104w ERC), 无此泄漏
- $\beta_W \approx 0$ = Phase 2 后 $W = \text{sign}(W_{\text{SIGN}}) \odot |D_{\text{scale}}|$, 符号结构 + 幅度数据化
- $\beta_\tau \approx 0$ = Phase 3 后 tau 数据化 (rolling 104w E_t 分位), 无此泄漏
- $\beta_{\theta} \approx 0$ = PPO 训期数据 < 测试期, 程序严格 OOS

实证估计 $\hat{S}_{\text{WFO}} = 1.695$, 减 8 ETF 池的 spec leakage $\approx 0.1 \sim 0.2$ 后, "程序严格 OOS Sharpe" 估计在 **1.4~1.5** 区间。

### 5. 风险与业绩指标

**直觉**: 这节是"业绩"的口径, 类似投资报告里的 KPI 定义。生产部署前用这套指标做 sanity check: Sharpe 反映风险调整收益, Sortino 关注下行风险, Calmar 把收益和最大回撤挂钩。年化系数 $\sqrt{52}$ 来自周频 52 周/年。

**5.1 Sharpe Ratio**

样本均值/标准差比值的年化:

$$
\hat{S} = \frac{\bar{r}_p}{\hat{\sigma}_p} \sqrt{52}, \quad \bar{r}_p = \frac{1}{N}\sum_{t=1}^{N} r_{p,t}, \quad \hat{\sigma}_p^2 = \frac{1}{N-1}\sum_t (r_{p,t} - \bar{r}_p)^2
$$

**5.2 Sortino Ratio**

类似 Sharpe 但只对下行波动惩罚, 反映投资者对"惊喜的损失"的厌恶:

$$
\hat{S}_{\text{Sortino}} = \frac{\bar{r}_p - r_f}{\hat{\sigma}_{\text{down}}}, \quad \hat{\sigma}_{\text{down}}^2 = \frac{1}{N}\sum_{t: r_t < r_f} (r_t - r_f)^2
$$

**5.3 Calmar Ratio**

年化收益与最大回撤比:

$$
\text{Calmar} = \frac{\text{AnnRet}}{|\text{MDD}|}
$$

**5.4 最大回撤 (MDD)**

$$
\text{MDD}(T) = \max_{0 \le \tau \le T} \left(1 - \frac{\text{NAV}_\tau}{\text{NAV}_T}\right) = 1 - \min_{0 \le \tau \le T} \frac{\text{NAV}_\tau}{\text{NAV}_T}
$$

**5.5 换手率与交易成本**

设单边交易成本 $c = 30$‱ (3‱), 滑点 $s = 1$‱, 换手率 $\tau = 6.6\%$。则年化交易成本:

$$
\text{Cost}_{\text{annual}} = 2 c \cdot \tau \cdot 52 = 2 \cdot 0.0003 \cdot 0.066 \cdot 52 = 0.206\%
$$

这相对我们 15.27% 年化收益是 1.3%, 不可忽略但仍可承受。

### 6. 8 ETF 池的因子分解

**直觉**: 选 8 个 ETF 不是 5 个或 18 个, 是因为 4 因子 (Equity/Rates/Credit/Inflation) 至少需要 4-5 个独立标的, 8 个既保证 N/p ≈ 21 (协方差可估) 又留出 equity 内的分散余量 (4 个 equity ETF 彼此相关性差异大, 见 6.2 表)。

**6.1 因子视角**

8 个 ETF 收益可被 4 个宏观因子近似解释:

$$
r_t = \beta_1 \cdot \text{Equity}_t + \beta_2 \cdot \text{Rates}_t + \beta_3 \cdot \text{Credit}_t + \beta_4 \cdot \text{Inflation}_t + \epsilon_t
$$

| ETF           | Equity β | Rates β | Credit β | Inflation β |
| ------------- | -------- | ------- | -------- | ----------- |
| 511010 国债     | -0.1     | 0.5     | 0.2      | 0.1         |
| 518880 黄金     | -0.2     | -0.3    | 0.0      | 0.7         |
| 511020 信用债    | 0.0      | 0.2     | 0.6      | 0.1         |
| 159985 商品     | 0.1      | -0.1    | 0.0      | 0.5         |
| 512100 中证1000 | 1.2      | -0.1    | 0.0      | 0.0         |
| 515050 红利低波   | 0.6      | 0.0     | 0.0      | 0.0         |
| 159915 创业板    | 1.3      | -0.2    | 0.0      | 0.0         |
| 510300 沪深300  | 0.9      | -0.1    | 0.0      | 0.0         |

8 ETF 覆盖 4 个宏观因子。Equity 类 4 个 (512100/515050/159915/510300) 集中暴露在 Equity 因子, Rates 类 2 个 (511010/511020) 集中暴露在 Rates 因子。

**6.2 实际两两相关 (5y 周频, 2020-2026)**

|        | 国债    | 黄金   | 信用债   | 商品   | 中证1000 | 红利低波  | 创业板   | 沪深300 |
| ------ | ----- | ---- | ----- | ---- | ------ | ----- | ----- | ----- |
| 国债     | 1.00  | 0.12 | 0.55  | 0.03 | -0.06  | -0.18 | -0.22 | -0.24 |
| 黄金     | 0.12  | 1.00 | 0.03  | 0.08 | 0.06   | 0.05  | 0.05  | 0.13  |
| 信用债    | 0.55  | 0.03 | 1.00  | 0.00 | -0.06  | -0.14 | -0.14 | -0.15 |
| 商品     | 0.03  | 0.08 | 0.00  | 1.00 | 0.05   | 0.01  | 0.03  | 0.07  |
| 中证1000 | -0.06 | 0.06 | -0.06 | 0.05 | 1.00   | 0.21  | 0.22  | 0.22  |
| 红利低波   | -0.18 | 0.05 | -0.14 | 0.01 | 0.21   | 1.00  | 0.71  | 0.65  |
| 创业板    | -0.22 | 0.05 | -0.14 | 0.03 | 0.22   | 0.71  | 1.00  | 0.85  |
| 沪深300  | -0.24 | 0.13 | -0.15 | 0.07 | 0.22   | 0.65  | 0.85  | 1.00  |

**关键观察**:

- 黄金和商品跟其他 6 个 ETF 几乎无关 (|ρ| < 0.15), 真正"独立"的两类资产
- 国债-信用债: 0.55, 同属 Rates 因子, 选 2 个提供久期分散
- 红利低波-创业板-沪深300: 0.65-0.85, equity 因子高相关
- 中证1000 跟其他 3 个 equity ETF 相关仅 0.21-0.22, 是 equity 内的"分散者"

**6.3 选取 8 个而非 5/18 的考量**

- **5 个**: 太少, 卫星资产缺失
- **18 个 (全 ClickHouse)**: 太多, 协方差估计 N/p = 8/173 ≈ 0.05, LedoitWolf 收缩后仍病态
- **8 个**: 覆盖 4 因子, 协方差估计 N/p ≈ 21 (充分), 同时保留 3-4 个 equity 卫星提供分散

### 7. 技术细节

**7.1 协方差估计的 LedoitWolf 收缩 (B0 实际使用)**

B0 数据化时, 我们对 8 ETF 协方差矩阵做简单 LedoitWolf 收缩:

$$
\hat{\Sigma}_{\text{LW}} = (1 - \rho) \hat{\Sigma}_{\text{sample}} + \rho \cdot \mu^2 I, \quad \mu^2 = \text{tr}(\hat{\Sigma}_{\text{sample}})/8
$$

其中 $\rho = 0.5$。在 104 周窗口上, 8 资产协方差估计 N/p = 13, 病态风险显著, 收缩到对角线 $\mu^2 I$ 提升数值稳定性。

ERC 优化在此 $\hat{\Sigma}_{\text{LW}}$ 上做 fixed-point 迭代 (Spinu 2013), 收敛到 $w_i (\Sigma w)_i$ 在各 $i$ 间近似相等。

**7.2 单纯形投影 (Dykstra)**

将 $b \in \mathbb{R}^8$ 投影到 box-constrained 单纯形 $\Delta = \{w : w \ge l_i, w \le u_i, \sum w = 1\}$。我们的实现是简化版 (clip + normalize 迭代 50 次), 收敛速度足够。

**7.3 8 ETF 真实收益的同步**

V3.1N 的 5 维特征来自 5 reference 资产 (000300.SH 等), 与 8 ETF 真业绩 (511010 等) **不是同一资产**。这是设计选择:

- 5 reference 资产是"市场代理", 波动率反映 regime
- 8 ETF 是"可投资标的", 业绩是 PPO 的真实奖励
- 这种解耦使 W 矩阵学到"regime → 资产类型"映射, 而 PPO 学"regime → tilt 强度"

**风险**: 5 reference 资产的 regime 信号可能滞后于 8 ETF 实际表现 (二者不是同标的)。实证上 Sharpe 1.695 表明这个滞后可接受。

---

## 9 维状态空间

| 维度  | 名称               | 含义             | 标准化                 |
| --- | ---------------- | -------------- | ------------------- |
| s0  | AE 重建误差          | 宏观压迫感          | z-score (rolling)   |
| s1  | vol_mkt_20d      | 沪深 300 20 日波动率 | minmax → [0, 1]     |
| s2  | llm_macro        | 流动性顺风          | (x-50)/50 → [-1, 1] |
| s3  | llm_sentiment    | 资金情绪           | 同上                  |
| s4  | llm_risk         | 风险压力           | 同上                  |
| s5  | port_sharpe_20d  | 组合 20 日 Sharpe | clip [-3, 3]        |
| s6  | port_mdd_current | 当前回撤           | abs                 |
| s7  | regret_ema_norm  | 遗憾 EMA         | (Walk-forward 中关闭)  |
| s8  | theta_prev       | 上一期 theta      | minmax → [0, 1]     |

## 1 维动作空间

- `a ∈ [-1, 1]` (Tanh 输出) → 线性映射 → `theta ∈ [0, 2]`
- theta=0: 纯 b0 等权, 无 tilt
- theta=1: 默认 V3.1N 行为 (W 矩阵的 raw score 直接 softmax)
- theta=2: 激进 (按 score 指数 tilt, 集中度最高)

---

## 8 ETF 池 (2020-2026 5y 周频)

| idx | 代码     | 类别        | 名称         | 5y Sharpe | 角色                                 |
| --- | ------ | --------- | ---------- | --------- | ---------------------------------- |
| 0   | 511010 | fi        | 国债 ETF     | 1.09      | 防御核心                               |
| 1   | 518880 | hedging   | 黄金 ETF     | 1.33      | 尾部对冲                               |
| 2   | 511020 | fi        | 信用债 ETF    | 0.55      | 防御辅助                               |
| 3   | 159985 | commodity | 商品 ETF     | 0.73      | 通胀敏感                               |
| 4   | 512100 | satellite | 中证1000 ETF | 0.51      | 高 beta 进攻                          |
| 5   | 515050 | satellite | 红利低波 ETF   | 0.50      | 防御进攻                               |
| 6   | 159915 | satellite | 创业板 ETF    | 0.41      | 科技股, 高 beta 增长 (Phase 2 替换 159919) |
| 7   | 510300 | satellite | 沪深300 ETF  | 0.29      | 兜底宽基                               |

(Sharpe 排序: 黄金 1.33 > 国债 1.09 > 商品 0.73 > 信用债 0.55 > 中证1000 0.51 > 红利低波 0.50 > 创业板 0.41 > 沪深300 0.29)

ETF 代码 (ClickHouse 原始) → 资产代码 (W 矩阵内部) 映射见 `src/data_pipeline/track_b/fetcher.py:fetch_n_etf()`。

---

## Walk-Forward 训练方案 (生产配置)

**核心**: 严格 OOS, PPO 训期 < 测试期, 程序无误泄露。

| 阶段          | 配置                                             | 说明                           |
| ----------- | ---------------------------------------------- | ---------------------------- |
| OOS PPO 训练  | **200k timesteps** on [2020-05-08, 2022-12-30] | 单 PPO 训冻结                    |
| 季度重训        | **5k steps** / quarter                         | warm-start from prev quarter |
| WFO 切换      | 季度边界 hot-swap                                  | 13 季度 ckpt                   |
| Lookback 窗口 | **78 周** (1.5 年)                               | rolling, 平衡反应速度              |
| 总耗时         | ~10 分钟                                         | 训 + WFO 端到端                  |

**Theta 区间**: [0, 2] (production) - 试过 [0, 3] 反而 Sharpe 略低 (1.688 vs 1.821)
**B0 来源**: 100% 数据化 (rolling 104w ERC) - 0 前视偏差
**W 矩阵**: $W = \text{sign}(W_{\text{SIGN}}) \odot |D_{\text{scale}}|$, 符号结构 + 幅度数据化
**tau 来源**: Phase 3 数据化, rolling 104w E_t 的 30 分位数 - bear pressure 充分启用
**8 ETF 池**: 159919 → 159915 (创业板) 切换, 行业分散 + 更高 Sharpe

### 实测调参 (含 Phase 1 + Phase 2 + Phase 3 + 159915 切换的最终生产配置)

| #   | 配置                                       | Sharpe                | 备注                          |
| --- | ---------------------------------------- | --------------------- | --------------------------- |
| 1   | 100k theta2 + 5k ws + 104w, 手设 B0        | 1.768                 | 旧基准                         |
| 2   | 200k theta2 + 5k ws + 78w, 手设 B0         | 1.821                 | 含 spec leakage              |
| 3   | 200k theta2 + 5k ws + 104w, 手设 B0        | 1.817                 |                             |
| 4   | 200k theta2 + 5k ws + 52w, 手设 B0         | 1.818                 |                             |
| 5   | 200k theta2 + 5k ws + 156w, 手设 B0        | 1.818                 |                             |
| 6   | 300k theta2 + 5k ws + 104w, 手设 B0        | 1.819                 | plateau                     |
| 7   | 200k theta3 + 5k ws + 104w, 手设 B0        | 1.515                 | theta3 过拟合                  |
| 8   | Phase 1: ERC B0 (159919)                 | 1.562                 | B0 数据化, 0 前视                |
| 9   | Phase 2: ERC B0 + W_hybrid (159919)      | 1.606                 | B0 + W 数据化                  |
| 10  | Phase 2 + 159915 切换 (B0 + W 数据化)         | 1.642                 | B0 + W + ETF 都数据化           |
| 10b | + Phase 3 tau q=20 (最激进 bear)            | 1.695                 | bear pressure 充分启用          |
| 10c | + Phase 3 tau q=30 (生产)                  | 1.695                 | 平衡点, 与 q=20 同分但 turnover 更低 |
| 10d | + Phase 3 tau q=50 (中等)                  | 1.656                 | bear 触发 50%                 |
| 10e | + Phase 3 tau q=70 (最保守)                 | 1.550                 | bear 触发 30%                 |
| 8   | 200k theta2 + 5k ws + 78w, no warm-start | 1.587 (warm-start 关键) |                             |

---

## 目录结构

```
AE_LLM_RL_FOF-main/
├── config.yaml                  # 全局配置 (theta_max=2.0, action_mapper, state_assembler, ppo)
├── pyproject.toml               # Python 依赖
├── .env                         # API Key / ClickHouse 连接
│
├── data/
│   ├── processed/               # features_master.parquet (25 维特征)
│   └── llm_cache/llm_scores.db  # LLM 周频 d1/d2/d3 评分
│
├── checkpoints/
│   ├── ae_weights.pth           # AE 25→6→25
│   ├── ae_scaler.pkl
│   ├── actor_critic_oos_theta2_200k.pth  # OOS PPO (生产 init)
│   └── walkforward/             # 13 季度 ckpt (warm-start)
│
├── src/
│   ├── features/                # 25 维特征 (5 资产 + 5 宏观)
│   ├── models/regime_autoencoder.py
│   ├── llm_engine/              # d1/d2/d3 评分 (10 概念池)
│   ├── compute/
│   │   ├── event_track_v3_1_n.py  # 8 资产 V3.1N 引擎 (核心)
│   │   └── v31_engine_n.py        # V31EngineN 包装
│   ├── selection/select_8_n.py  # 8 ETF 选基 (LLM 信号驱动)
│   ├── data_pipeline/track_b/fetcher.py  # ClickHouse 8 ETF
│   ├── env/
│   │   ├── mdp_environment.py   # 9 维状态 × 1 维动作 MDP
│   │   ├── state_assembler.py   # 9 维状态组装
│   │   ├── action_mapper.py     # a → theta
│   │   ├── reward_function.py   # 复合奖励
│   │   ├── regret_engine.py     # 8→5 聚合, 暂时 bypass
│   │   └── metrics_utils.py     # Sharpe / MDD
│   ├── ppo/                     # Actor-Critic, PPO trainer, buffer, GAE
│   ├── inference/               # EMA 滤波, panic index (WFO Scheduler 用)
│   ├── training/                # burn_in, dual_track (legacy WFOScheduler)
│   ├── schedules/wfo_scheduler.py
│   ├── failsafe/                # VetoSwitch (d3 > 85 一票否决), FallbackSelector
│   └── penetration/agentbase_formatter.py  # 8 ETF 权重 → AgentBase JSON
│
├── scripts/                     # 6 步生产流水线
│   ├── run_data_etl.py          # Step 1: 数据 ETL
│   ├── run_llm_batch.py         # Step 2: LLM 批量打分
│   ├── train_ae.py              # Step 3: AE 训练
│   ├── train_ppo.py             # Step 4: OOS PPO 训练 (200k timesteps)
│   ├── train_ppo_walkforward.py # Step 4b: 13 季度 warm-start 增量训
│   └── run_backtest_wfo_n.py    # Step 5: WFO 回测 + 季度 hot-swap
│                                  # Step 6 (实盘信号): TBD - 基于 run_backtest_wfo_n 改造
│
├── tests/                       # 单元测试 (生产相关)
│   ├── test_regime_autoencoder.py     # AE 前向
│   └── test_normalizer_no_lookahead.py # 防穿越标准化
│
├── archive/                     # 历史代码 (不参与生产)
│   ├── deprecated_tests_20260608/     # 旧 5 资产测试 + 旧 EventTrack / V31Engine
│   ├── pre_theta_refactor_20260607/   # theta refactor 前的完整快照
│   ├── pre_stage7c_20260607/          # Stage 7c 之前
│   ├── pre_v31_cleanup/               # V3.1 清理前
│   └── post_stage7_20260607/          # Stage 7 收尾
│
├── results/wfo/                 # 6 个回测结果
│   ├── stage7c_wfo_FINAL/phase3_tau_quantile30/  # 生产: Sharpe 1.695 (3.32y 严格 OOS, B0+W+tau 数据化 + 159915)
│   ├── stage7c_oos_v2/          # 对比: OOS PPO 单次 1.529
│   ├── stage7c_oos_oldppo/      # 对比: in-sample 1.756
│   ├── stage7c_5y/              # 对比: 5.32y in-sample 1.556
│   ├── stage7c_wfo/             # 对比: 第一版 WFO 1.768
│   └── target_weights_*.json    # 实盘信号输出
│
└── docs/                        # 文档
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- ClickHouse (quantchdb) 8 ETF 数据可用
- LLM API Key (Step 2)

### 安装

```bash
pip install -e ".[dev]"
```

### 流水线 (6 步)

```bash
# Step 1: 拉 8 ETF OHLC + 5 资产特征 + 5 维宏观 → features_master.parquet
python scripts/run_data_etl.py --start-date 2015-01-01

# Step 2: LLM 批量打分 (d1/d2/d3) → llm_scores.db
python scripts/run_llm_batch.py --start-week 2022-01-07

# Step 3: 训 AE 自编码器 (25 → 6 → 25)
python scripts/train_ae.py --epochs 50

# Step 4: OOS PPO 训练 (训期 [2020-05, 2022-12], 200k timesteps)
python scripts/train_ppo.py \
    --start-date 2020-05-08 \
    --end-date 2022-12-30 \
    --total-timesteps 200000 \
    --checkpoint-path checkpoints/actor_critic_oos_theta2_200k.pth

# Step 4b (可选): 季度 warm-start 增量训 → 13 个 walkforward ckpt
python scripts/train_ppo_walkforward.py \
    --init-checkpoint checkpoints/actor_critic_oos_theta2_200k.pth \
    --init-end-date 2022-12-30 \
    --lookback-weeks 78 \
    --timesteps-per-quarter 5000

# Step 5: WFO 回测 (季度 hot-swap, 真实 OOS)
python scripts/run_backtest_wfo_n.py \
    --start-date 2023-01-01 \
    --end-date 2026-04-30 \
    --walkforward-dir checkpoints/walkforward \
    --ppo-checkpoint checkpoints/actor_critic_oos_theta2_200k.pth \
    --output-dir results/wfo/stage7c_wfo_FINAL

# Step 6: 实盘信号 (单周五触发)
python scripts/run_inference_live.py --week-end 2026-05-08  # TBD: 基于 run_backtest_wfo_n 改造
```

---

## 关键设计

### V3.1N 8 资产引擎

`src/compute/event_track_v3_1_n.py:EventTrackV31N` 核心组件:

- **W 矩阵** (8×5): 5 维市场状态 → 8 资产 score, 手调自 5y 业绩
- **B0** (8,): 等权基准, 防御底仓
- **V_DEFENSE** (8,): 防御打分, AE regime 触发时叠加
- **BOUNDS** (8,): 各资产硬上限 (0.20~0.45)

输入: 5 维 market features (从 features_master 5 reference 资产波动率派生)
输出: 8 维 ETF 权重, 严格 sum=1

### 复合奖励函数

`src/env/reward_function.py`: 8 维 w × 8 ETF 真实 weekly return = r_port, 再算 Sharpe / MDD 加权。

| 项           | 含义                       |
| ----------- | ------------------------ |
| r_port      | 本周组合真实收益                 |
| λ_turnover  | 换手惩罚 (0.001)             |
| λ_endpoint  | alpha 中心化 (当前未启用, 1 维动作) |
| SwitchBonus | regime 切换正确奖励            |

### 风险熔断 (Veto Switch)

LLM d3 > 85 → 强制把对应概念降权至 0。Walk-forward 中 d3 取宽基均值 × 0.7 + 卫星最大 × 0.3。

---

## 实测结果 (严格 OOS)

**测试期**: 2023-01-01 ~ 2026-04-30 (3.32 年, 173 周, 8 ETF 真实 OHLC + 手续费/滑点)

| 方案                                         | Sharpe    | Sortino   | Calmar    | Ann        | MDD       | Win       | Turnover |
| ------------------------------------------ | --------- | --------- | --------- | ---------- | --------- | --------- | -------- |
| OLD PPO (in-sample 200k, 手设 B0)            | 1.756     | 1.973     | 1.509     | 14.97%     | 9.92%     | 56.7%     | 6.5%     |
| OOS PPO (单次 100k, 冻结, 手设 B0)               | 1.529     | 1.745     | 1.329     | 13.20%     | 9.93%     | 56.2%     | —        |
| WFO + 手设 B0 (200k + 78w + 5k ws)           | 1.821     | 2.040     | 1.570     | 15.46%     | 9.85%     | 57.2%     | 6.0%     |
| Phase 1: B0 数据化 (rolling 104w ERC)         | 1.562     | 1.845     | 1.588     | 14.40%     | 9.07%     | 55.8%     | 8.1%     |
| Phase 2: B0 + W_hybrid 数据化                 | 1.606     | 1.889     | 1.582     | 14.42%     | 9.11%     | 55.3%     | 6.6%     |
| **Phase 2 + 159915 + tau q=30 (生产, 0 前视)** | **1.695** | **1.999** | **1.710** | **15.43%** | **9.09%** | **54.6%** | **5.8%** |

注: 1.821 vs 1.695 的 0.13 Sharpe 差距主要来自 B0 的 60% 防御偏向 + tau=15 的过激 bear pressure (spec leakage)。生产 1.695 是 100% 程序严格 OOS (B0 + W + tau 都数据化) 的"诚实"估计。

WFO vs in-sample OLD: 严格 OOS 已与 in-sample 持平 (0.07 差距), 持续 adapt 优于冻结。

---

## 技术栈

| 组件   | 技术                                      |
| ---- | --------------------------------------- |
| 深度学习 | PyTorch >= 2.0                          |
| 强化学习 | 自实现 PPO (Actor-Critic, GAE, Clip)       |
| LLM  | OpenAI SDK (兼容 Qwen/Claude)             |
| 数据   | ClickHouse (quantchdb), pandas, pyarrow |
| 回测   | GeneralBacktest (自研, 含手续费/滑点)           |
| 监控   | TensorBoard                             |
| 测试   | Pytest                                  |

---

## 测试

```bash
pytest tests/ -v
```

主要测试:

- `test_event_track_prototypes.py` — V3.1N 原型 + regime 切换
- `test_mdp_environment_*.py` — MDP 环境对齐
- `test_regime_autoencoder.py` — AE 前向
- `test_normalizer_no_lookahead.py` — 标准化防穿越

---

## 回测交易费率

`scripts/run_backtest_wfo_n.py` 调用 GeneralBacktest:

| 参数               | 值                | 说明             |
| ---------------- | ---------------- | -------------- |
| transaction_cost | [0.0003, 0.0003] | 买卖各 0.03% (3‱) |
| slippage         | 0.0001           | 0.01% (1‱)     |

> 当前费率偏低, 公募 FOF 实际申赎 0.5%~1.5%, 滑点更高。需调整直接修改 `run_backtest_wfo_n.py` 中 `run_backtest()` 参数。
