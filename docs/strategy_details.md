# AE-LLM-RL-FOF 策略细节

> **适用读者**: 量化策略研究员
> **数据样本**: features_master (2020-05 ~ 2026-04) + llm_cache (2020-01 ~ 2026-04) + results/wfo/20260602_182649 (156 周 WFO 2023-01-06 ~ 2025-12-26)
> **当前主线**: EventTrack **V3.1**(V3 审计修复版: 矩阵化得分 + AE shifter + b-as-policy)
> **V1/V2 历史**: 见 §三末尾的简表,详细设计已归档至 `archive/pre_v31_cleanup/`

---

## 一、策略架构: 三层解耦的元控制器

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 0: 数据层  (5 资产日频收益 + 宏观流动性 + LLM 周度语义)         │
│  ─────────────────────────────────────────────────────────────────── │
│  [0=宽基 1=卫星 2=固收 3=黄金 4=现金]                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ R ∈ R^(5×T), d1/d2/d3 ∈ [0,100], E_t
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: 两条基底策略 (NormalTrack 防御 + EventTrack 进攻)           │
│  ─────────────────────────────────────────────────────────────────── │
│  w^N = NormalTrack(R, E_t, τ)         ← ERC + bear 强制防御          │
│  w^E = EventTrackV31(R, d1, d2, d3, E_t, τ)   ← W·f + AE shifter + b-as-policy │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ w^N, w^E ∈ Δ⁴
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: PPO 元控制器 + 凸组合融合                                  │
│  ─────────────────────────────────────────────────────────────────── │
│  α_t = clip(α_{t-1} + a_1·0.1, 0, 1)      ← α_max=0.1 (Stage 1 已应用) │
│  τ_t = clip(τ_{t-1} + a_2·0.1, 5, 50)     ← Stage 2 bug fix 已应用  │
│  w^F = α_t · w^E + (1 - α_t) · w^N        ← 凸组合                  │
└─────────────────────────────────────────────────────────────────────┘
```

**核心修复**: Stage 2 让两条 track 在 regime 维度真正分化;V3.1 在 V3 基础上修复 3 个结构缺陷(尺度陷阱、黄金悲剧、AE 增益悖论)。

---

## 二、数据流时序(单周五的完整管线)

```
T 周五 17:00 ETL 触发
│
├─ 1. ClickHouse → fetcher.fetch_track_b()
│     拉取 5 资产日频 OHLC (含 adj_factor), shape = (T, 5×4)
│
├─ 2. features_master 更新 (asset_features 20 维 + macro_features 5 维 = 25 维)
│
├─ 3. RegimeAutoEncoder.forward(x_25d) → E_t (Z-score 标准化)
│
├─ 4. AsyncSemanticEngine.batch_query() → d1/d2/d3 ∈ [0, 100]
│     2 主题 prompt + 2 并发 GPT 调用 → 落库 llm_cache.db
│
├─ 5. aggregate_llm_weekly() → llm_macro/sentiment/risk
│
├─ 6. WFO 主循环
│     (A) is_quarter_end → 重训 AE 30 epoch + PPO 1000 iter
│     (B) trigger_weekly_inference → E_t
│     (C) dual_engine.compute(returns_5d, llm_*, ae_error, tau_prev) → w^N, w^E
│     (D) PPO Actor(s_t) → (a1, a2); α_t, τ_t 更新
│     (E) w_fused = α_t · w^E + (1-α_t) · w^N
│     (F) regret_engine.compute → regret_ema (16 静态专家库)
│     (G) r_fused = w_fused @ returns_t, 更新 NAV
│
└─ 7. 落盘 gate_diagnostics.csv + nav_series.csv + weights_data_generalbt.csv
```

**关键符号**:
- $R \in \mathbb{R}^{5 \times T}$: 5 资产日收益矩阵,默认 $T=20$
- $\hat{\Sigma}_{\text{lw}}$: LedoitWolf 收缩协方差
- $d_1, d_2, d_3$: LLM 流动性/情绪/风险 ∈ $[0, 100]$
- $E_t$: AE 重建误差 (Z-score), $\tau$: regime 阈值
- $\alpha_t$: PPO 融合系数, $\mathbf{w}^N, \mathbf{w}^E, \mathbf{w}^F$: 防御/进攻/融合权重

---

## 三、EventTrack 设计演化 (V1 → V2 → V3 → V3.1)

> 详细设计、V2 RB 目标函数、V3 5 维 score 公式、Stage 2 量化结果均已归档到 `archive/pre_v31_cleanup/`, 配 README 说明。本节只保留版本对比与关键设计变化。

### 3.1 四代设计一句话

| 版本 | 核心思路 | 关键缺陷 |
|---|---|---|
| **V1** (Stage 2) | 4 个硬编码原型 + softmax + event_intensity 加性混合 | 叠床架屋,LLM 信号被压成背景噪声,156 周上两条 track 几乎相同(edge std 0.21%) |
| **V2** (RB) | 3 顶点凸包 (B_BEAR/NEUTRAL/GROWTH) + LLM→β 线性 + Risk Budgeting 多项式求解 | 凸包 2D 三角片面在 $\Delta^4$ 内有几何盲区(滞胀/通缩恐慌距离 0.27/0.69) |
| **V3** (Exp Tilting) | 5 维 per-asset 得分 + $\theta_{\text{eff}}$ 指数偏置 + RB 求解 | **3 个结构缺陷**(见 §六): 尺度陷阱、黄金悲剧 ($e^3 : e^1 = 20:1$)、AE 增益悖论 |
| **V3.1** (审计修复) | 矩阵化得分 $\mathbf{W} \cdot \mathbf{f}$ (L1 ≤ 1/行) + AE shifter 替换 AE gain + b-as-policy 替换 RB | edge std 0.24% (低于 V3 的 0.43%) 但跨 regime 分化更稳 |

### 3.2 V1 → V2 → V3 → V3.1 156 周硬指标对照

| 指标 | V1 (Stage 2) | V2 (RB) | V3 (Exp Tilting) | V3.1 (审计修复) |
|---|---|---|---|---|
| **edge std** (PPO fusion 价值) | 0.21% | 0.23% | **0.43%** | 0.24% |
| **Pure EventTrack Sharpe** | 0.254 | **0.948** | 0.947 | 0.576 |
| **50/50 blend cum** | 2.97% | **6.23%** | 6.01% | 4.72% |
| **bear weeks $w_{e,\text{fi}}$** | 0.450 (硬覆盖) | 0.349 | 0.198 (LLM 驱动, 偏弱) | 0.249 |
| **bear weeks $w_{e,\text{gold}}$** | 0.350 | 0.278 | 0.164 | **0.341** |
| **bear $w_{e,\text{gold}} / w_{e,\text{fi}}$** | 0.78 | 0.80 | 0.83 | **1.37** ← V3.1 唯一 gold > fi |
| **bull - bear offensive diff** | +0.294 | +0.206 | -0.018 (V3 跨 regime 一致) | +0.113 |
| **5-dim $w_e$ weekly std** | 0.062 | 0.072 | 0.068 | **0.027** (最稳定) |
| **滞胀/非共识周** | - | 53 (RB 副作用) | 16 (真实) | 154 (b-as-policy 副作用) |
| **AE 软注入 C^∞ 平滑** | ✗ 硬开关 | ✗ 硬开关 | ✓ (但有悖论) | ✓ (shifter, 无悖论) |
| **单元测试** | 5 | 12 | 15 | **18** |

### 3.3 NormalTrack (V1/V2/V3 共享)

NormalTrack 在整个 V1→V3.1 演化中保持稳定,**核心设计未变**:

```python
# 1. Bear 强制重分配 (Stage 2 引入)
if ae_error > tau:
    return (0.10, 0.05, 0.50, 0.20, 0.15)   # 固收 50% + 黄金 20% = 防御 85%

# 2. 样本量不足回退
if T < MIN_SAMPLES: return (0.2, 0.2, 0.2, 0.2, 0.2)

# 3. 主路径: LedoitWolf 协方差 + ERC (多项式目标)
Sigma = LedoitWolf().fit(R.T).covariance_
w = SLSQP.minimize(sum((w * (Sigma@w) - 0.2 * (w'@Sigma@w))**2), ...)
```

**ERC 目标**: $\min_{\mathbf{w}} \sum_i (w_i (\Sigma \mathbf{w})_i - \frac{1}{5} \mathbf{w}^T \Sigma \mathbf{w})^2$

**Post-Stage 2 bounds** (放宽让 ERC 真的能产生差异):
| 资产 | bounds | 改动理由 |
|---|---|---|
| broad | (0.05, 0.40) | 不变 |
| satellite | (0.00, 0.25) | 下限 0 (允许 ERC 完全去除卫星) |
| fixed_income | (0.10, 0.60) | 不变 |
| safe (gold) | (0.00, 0.30) | 上限 30% (允许避险加重) |
| cash | (0.00, 0.30) | 上限 30% |

### 3.4 PPO 融合 (所有版本共享)

```python
# Stage 1 修复后 (alpha_max=0.1, alpha_bias=-0.05, switch_bull_reward=0.05, lambda_alpha_direct=0.05)
delta_alpha = a_1 * 0.1
delta_tau   = a_2 * 0.1   # Stage 2 bug fix: tau 真的被 PPO 更新
alpha_t = clip(alpha_prev + delta_alpha, 0, 1)
tau_t   = clip(tau_prev   + delta_tau,   5, 50)
w_fused = alpha_t * w_event + (1 - alpha_t) * w_normal
```

**修复历史**:
- Pre-Stage 2 bug: `tau_t = tau_prev` (a2 永远不生效)
- Stage 2 fix: 用 `action_mapper.clip_tau` 让 PPO 真的能调整 regime 阈值
- Stage 1 (已完成): $\alpha_{\max}$ 0.5→0.1, $\alpha_{\text{bias}}$ 0→-0.05, $\lambda_{\text{alpha\_direct}}$ 0→0.05, `switch_bull_reward` 0.45→0.05 — 让 PPO 在 [0.3, 0.7] 真实摆动, 15 个 Stage 1 ML 单测 + 3 个 alignment 测全部通过

---

## 四、EventTrack V3 详细设计 (Per-Asset 指数偏置)

> V3 解决 V2 凸包的几何禁锢 — 用 5 维 per-asset 得分 (自由度 5) 替代 V2 的 3 顶点凸包 (自由度 2), 让 $\mathbf{b}$ 覆盖整个 $\Delta^4$ 内部。

### 4.1 V2 凸包的 3 个结构盲区

| 非共识情景 | 期望 $\mathbf{b}$ | V2 凸包内最近点 | 距离 |
|---|---|---|---|
| **滞胀(2022 实盘)** | (0.02, 0.03, **0.37**, **0.22**, **0.37**) | β_bear=0.5, β_neutral=0.5 | 0.27 |
| **通缩恐慌(2020-03)** | (0.001, 0.001, **0.78**, 0.09, 0.13) | β_bear=0.85 | 0.69 |
| **黄金单边行情** | (0.13, 0.13, 0.13, **0.36**, 0.25) | β_neutral=1 | 0.18 |

凸包只能 convex combo 三个端点,无法表达"现金 + 黄金双高 + 固收被压"等非共识形态。

### 4.2 V3 核心公式

**Step 1**: LLM 信号归一化到 $[-1, 1]$, **Step 2**: 市场特征 (同 §3.3)。

**Step 3**: 5 维 per-asset 得分向量 (V3 的核心):

$$
\begin{aligned}
s_{\text{broad}} &= m + s - r \\
s_{\text{sat}}   &= m + s - r + \text{sat\_lead} \\
s_{\text{fi}}    &= -m - s + \text{equity\_stress} \\
s_{\text{gold}}  &= r \\
s_{\text{cash}}  &= -m + r
\end{aligned}
$$

**Step 4**: AE soft injection ($\theta_{\text{eff}}$ 增益):

$$
\text{bear\_pressure} = \sigma\!\left(\frac{E_t - \tau}{\tau \cdot 0.5}\right), \quad \theta_{\text{eff}} = \theta \cdot (1 + 0.5 \cdot \text{bear\_pressure})
$$

**Step 5**: 指数偏置 (以 ERC $b_0$ 为基准):

$$
\boxed{b_i = \frac{b_{0,i} \cdot e^{\theta_{\text{eff}} \cdot s_i}}{\sum_j b_{0,j} \cdot e^{\theta_{\text{eff}} \cdot s_j}}, \quad b_0 = (0.2, 0.2, 0.2, 0.2, 0.2)^T}
$$

**Step 6**: RB 求解 (LedoitWolf 协方差 + 多项式目标 + 非对称初值 $w_0 = [0.22, 0.18, 0.25, 0.20, 0.15]$)。

### 4.3 V3 端到端数学性质

| 性质 | V2 凸包 | V3 指数偏置 |
|---|---|---|
| 自由度 | 2 (3 顶点 → 2 个 β) | 5 (5 个独立 $s_i$) |
| $\mathbf{b}$ 所在流形 | 凸包 2D 三角片面 | 整个 $\Delta^4$ 内部 |
| 端到端平滑性 | β→b 线性, b→w 经 RB | $C^\infty$ (sigmoid + exp) |
| 滞胀覆盖 | △ 距离 0.27 | ✓ 精确表达 |
| 调参复杂度 | 19 个数字 | 5 + 1 = 6 个数字 |

### 4.4 V3 关键发现与局限

**优势**: edge std 2x 提升 (0.43% > 0.21%), Sharpe 最高 (0.631), 调参更少, 全 $\Delta^4$ 覆盖。

**局限**:
- bear $w_{e,\text{fi}}$ = 0.198 (防御弱于 V1/V2, 缺硬防御 fallback)
- 跨 regime offensive 一致 (0.49 bull vs 0.51 bear), LLM 信号 calibration 问题
- 包含 3 个 **结构缺陷**, 见 §六

### 4.5 V3 调参指南 (已被 V3.1 取代, 仅作历史参考)

| 参数 | 默认 | 影响 |
|---|---|---|
| `THETA` | 1.0 | ↑ 增 sharp, b 分布更极端 |
| `AE_THETA_BOOST` | 0.5 | ↑ 增 bear 期响应, 解决 V3 bear 弱 |
| `AE_SIGMOID_SCALE` | 0.5 | ↑ 加快 transition 锐度 |

### 4.6 V3 全流程数学推导 (从 $\mathbf{R}, d_1, d_2, d_3, E_t, \tau$ 到 $\mathbf{w}^E$)

> 本节给出 V3 端到端数学推导。所有公式与 `src/compute/event_track_v3.py` 一一对应, 配 §4.6.7 数值例子走通完整流程。

#### 4.6.1 输入输出

**输入**: $\mathbf{R} \in \mathbb{R}^{5 \times T}$ (5 资产日收益矩阵, 默认 $T=20$), $d_1, d_2, d_3 \in [0, 100]$ (LLM 三分量), $E_t \in \mathbb{R}$ (AE 重建误差, 可选), $\tau \in [5, 50]$ (regime 阈值, 可选).

**输出**: $\mathbf{w}^E \in \mathbb{R}^5$, $w^E_i \geq 0$, $\sum_i w^E_i = 1$.

#### 4.6.2 Step 1 — 安全标准差 (Safe Sigmas)

对 5 资产各取样本标准差 (ddof=1), 防止零波动率或非有限值:

$$
\sigma_i = \begin{cases} \mathrm{std}(R_{i,:};\ \mathrm{ddof}=1) & \text{若 std 有限且} > 0 \\ 10^{-3} & \text{否则} \end{cases}, \quad i = 0, 1, 2, 3, 4
$$

**数值安全**: `np.where(~isfinite(sigmas) | (sigmas <= 0), 1e-3, sigmas)`.

#### 4.6.3 Step 2 — LLM 信号归一化到 $[-1, 1]$

将 LLM 的 0-100 分量线性映射到 $[-1, 1]$ 区间:

$$
m = \mathrm{clip}\!\left(\frac{d_1 - 50}{50},\ -1,\ 1\right), \quad s = \mathrm{clip}\!\left(\frac{d_2 - 50}{50},\ -1,\ 1\right), \quad r = \mathrm{clip}\!\left(\frac{d_3 - 50}{50},\ -1,\ 1\right)
$$

**语义**: $m > 0$ = 流动性宽松, $m < 0$ = 紧缩. $s > 0$ = 市场情绪乐观, $s < 0$ = 悲观. $r > 0$ = 风险上升, $r < 0$ = 风险消退.

#### 4.6.4 Step 3 — 市场结构特征 (2 个衍生量)

**Equity Stress** (权益波动相对防御波动的过剩程度):

$$
\mathrm{equity\_stress} = \frac{1}{2} \cdot \mathrm{clip}\!\left( \frac{\sigma_0 + \sigma_1}{\sigma_2 + \sigma_3 + \varepsilon} - 1,\ 0,\ 2 \right) \in [0, 1]
$$

**Satellite Lead** (卫星相对宽基的波动优势):

$$
\mathrm{sat\_lead} = \mathrm{clip}\!\left( \frac{\sigma_1 - \sigma_0}{\sigma_1 + \sigma_0 + \varepsilon},\ -1,\ 1 \right) \in [-1, 1]
$$

#### 4.6.5 Step 4 — 5 维 Per-Asset 得分 (V3 核心)

V3 用 **加性组合** 替代 V2 的多原型查表, 5 资产各得一个分数:

$$
\begin{aligned}
s_{\text{broad}} &= m + s - r \\
s_{\text{sat}}   &= m + s - r + \mathrm{sat\_lead} \\
s_{\text{fi}}    &= -m - s + \mathrm{equity\_stress} \\
s_{\text{gold}}  &= r \\
s_{\text{cash}}  &= -m + r
\end{aligned}
$$

**关键性质 (V3 的核心 bug)**: $|s_i|$ 的 L∞ 范数不一致:

| 资产 | $\max |s_i|$ | 公式来源 |
|---|---|---|
| broad | 3 | $|m| + |s| + |r|$ 全部相加 |
| sat | 4 | 再加 $\mathrm{sat\_lead}$ |
| fi | 2 | 只加 2 项 |
| gold | **1** | 单变量 $r$ |
| cash | 2 | 2 项 |

这导致 $e^{s_i}$ 偏置中卫星可压过黄金 **4 倍**, 见 §五 缺陷一。

#### 4.6.6 Step 5 — AE Soft Injection (V3 的 GAIN 模式)

AE 不再触发硬开关, 而是用 sigmoid 把 regime 信号压成 $[0, 1]$ 的连续值:

$$
\text{bear\_pressure} = \sigma\!\left(\frac{E_t - \tau}{\tau \cdot 0.5}\right) = \begin{cases} 0 & E_t \ll \tau \text{ (bull)} \\ 0.5 & E_t = \tau \\ 1 & E_t \gg \tau \text{ (bear)} \end{cases}
$$

V3 把 $\text{bear\_pressure}$ 注入 $\theta$ 形成 **增益乘数**:

$$
\theta_{\text{eff}} = \theta \cdot (1 + 0.5 \cdot \text{bear\_pressure}) \in [1.0, 1.5]
$$

**关键性质 (V3 的核心 bug)**: $\theta_{\text{eff}}$ 是 variance multiplier, 在 LLM 看多时反而**放大多头得分**, 见 §五 缺陷三。

#### 4.6.7 Step 6 — 指数偏置 (Exponential Tilting)

以 ERC 组合 $b_0 = (0.2, 0.2, 0.2, 0.2, 0.2)^T$ 为基准, 指数推 $b$ 到 LLM 想要的位置:

$$
\boxed{b_i = \frac{b_{0,i} \cdot e^{\theta_{\text{eff}} \cdot s_i}}{\sum_{j=0}^{4} b_{0,j} \cdot e^{\theta_{\text{eff}} \cdot s_j}}}
$$

**数学对偶 NormalTrack 的 ERC**: V3 把 $b$ 从均匀移到信号驱动位置, NormalTrack 始终 $b_i = 1/5$。

#### 4.6.8 Step 7 — RB 求解 (LedoitWolf + 多项式目标)

**LedoitWolf 协方差**: $\hat{\Sigma}_{\text{lw}} = \mathrm{LedoitWolf().fit}(R^T).\mathrm{covariance\_}$ (sklearn 自动 shrinkage).

**尺度归一化**: $\sigma_{\text{scale}} = \mathrm{mean}(\mathrm{diag}(\hat{\Sigma}_{\text{lw}}))$, $\hat{\Sigma} \leftarrow \hat{\Sigma} / \sigma_{\text{scale}}$. 这让 RB 景观保持 $O(1)$, SLSQP ftol=1e-9 一次到位。

**多项式目标 (无 1/port_var torsion)**:

$$
\boxed{\min_{\mathbf{w}} \mathcal{L}_{\text{RB}}(\mathbf{w}) = \sum_{i=0}^{4} \big( w_i (\hat{\Sigma}\mathbf{w})_i - b_i \cdot \mathbf{w}^T \hat{\Sigma} \mathbf{w} \big)^2}
$$

**约束**: $\sum_i w_i = 1$, $w_i \in [\mathrm{lo}_i, \mathrm{hi}_i]$ (V3 的 BOUNDS).

**初始猜测** (非对称, 避免 ERC 零梯度): $\mathbf{w}_0 = (0.22, 0.18, 0.25, 0.20, 0.15)^T$.

**求解器**: `scipy.optimize.minimize(method='SLSQP', ftol=1e-9, maxiter=500)`. 失败回退 $\mathbf{w} = b$。

#### 4.6.9 Step 8 — Clip + 归一化

$$
\tilde{\mathbf{w}}^E = \mathrm{clip}(\mathbf{w}, 0, 1), \quad \mathbf{w}^E = \tilde{\mathbf{w}}^E / \sum_{i=0}^{4} \tilde{w}^E_i
$$

#### 4.6.10 数值例子 (V3 端到端)

**输入**: $d_1=0, d_2=0, d_3=100$ (极端 risk-off, m=-1, s=-1, r=+1), $\sigma = (0.012, 0.015, 0.003, 0.010, 0.001)$ (宽基 1.2%, 卫星 1.5%, 固收 0.3%, 黄金 1.0%, 现金 0.1%).

| Step | 计算 | 结果 |
|---|---|---|
| Step 2 LLM 归一化 | $m=-1, s=-1, r=+1$ | $m=-1, s=-1, r=+1$ |
| Step 3 equity_stress | $(0.012+0.015)/(0.003+0.010) - 1 = 1.077$ → clip+÷2 | $\mathrm{eq} = 0.538$ |
| Step 3 sat_lead | $(0.015-0.012)/(0.015+0.012) = 0.111$ | $\mathrm{sat\_lead} = 0.111$ |
| Step 4 5 维得分 | $\begin{aligned} s_b &= -1-1-1 = -3 \\ s_s &= -3 + 0.111 = -2.889 \\ s_f &= -(-1)-(-1)+0.538 = 2.538 \\ s_g &= +1 \\ s_c &= -(-1)+1 = 2 \end{aligned}$ | $\mathbf{s} = [-3, -2.889, 2.538, 1, 2]^T$ |
| Step 5 AE (假设 bear) | $\text{bear\_pressure}=1$ | $\theta_{\text{eff}} = 1.5$ |
| Step 6 指数偏置 | $\begin{aligned} e^{1.5 \cdot \mathbf{s}} &= [e^{-4.5}, e^{-4.33}, e^{3.81}, e^{1.5}, e^{3}] \\ &= [0.0111, 0.0131, 45.0, 4.48, 20.09] \end{aligned}$ | $Z_{exp} = 69.7$ |
| | $b = 0.2 \cdot e^{1.5 \cdot \mathbf{s}} / Z_{exp}$ | $b = [0.000032, 0.000038, 0.129, 0.0129, 0.0577]$ |
| | (验证: $0.2 \cdot e^{1.5 \cdot s} = [0.00222, 0.00262, 9.00, 0.896, 4.018]$, $Z_{0.2} = 13.92$, $b = [0.000160, 0.000188, 0.647, 0.0644, 0.289]$) |  |
| | **实际比例**: $b_f / b_g = 10.0$ (固收 10 倍黄金) | **黄金悲剧**: $b_g / b_f = 0.0995$ |
| Step 7 RB | inverse-vol 偏好低 σ 资产 (fi vol=0.003, cash vol=0.001) | $w \approx [0.05, 0.05, 0.45, 0.16, 0.10]$ |

**V3 在纯 crisis + 强 bear 下的 $b$**: $b_g / b_f = 0.0995$ (黄金仅固收的 1/10), $b_c / b_f = 0.45$。**exp 偏置让黄金悲剧 + RB inverse-vol 让现金压黄金**, 完全背叛 V2 战略意图 (V2 B_BEAR 设计黄金 35%, 固收 45%, 现金 10%)。

---

## 五、V3 深度审计: 三个隐蔽型结构缺陷

> **审计范围**: V3 在 156 周样本期表面指标优秀 (Sharpe 0.631, edge std 0.43%), 但本节论证 **3 个** 结构设计缺陷, 不是调参问题, 是 V3 公式层面的问题。

### 5.1 缺陷一: Scale-Mismatch Trap (尺度陷阱)

| 资产 | 公式 | L∞ 范数 |
|---|---|---|
| $s_{\text{broad}}$ | $m + s - r$ | 3 |
| $s_{\text{sat}}$ | $m + s - r + \text{sat\_lead}$ | **4** |
| $s_{\text{fi}}$ | $-m - s + \text{equity\_stress}$ | 2 |
| $s_{\text{gold}}$ | $r$ | **1** |
| $s_{\text{cash}}$ | $-m + r$ | 2 |

**问题**: 在 exp 偏置 $b_i = b_{0,i} \cdot e^{\theta \cdot s_i}$ 中, 卫星的最大信号强度是黄金的 **4 倍**。即使 LLM 信号同时强烈看多黄金和看空卫星, 卫星的 exp 贡献可能压过黄金。

**反例** (LLM 看多黄金 + 看空卫星 + 中性 cash):
- $m=0, s=0, r=+1, \text{eq}=0, \text{sat}=-1$
- $b = [0.132, 0.018, 0.132, 0.359, 0.359]$ — 黄金 ≈ 现金 36%/36%, **没有避险不对称**。

**根因**: Hardcoded 加法得分是 per-feature 系数的隐式乘法, 无法保证各资产得分等尺度。

### 5.2 缺陷二: Gold Tragedy (黄金悲剧)

在纯 crisis 信号 $m=-1, s=-1, r=+1, \text{eq}=1, \text{sat}=0$ 下:

| 资产 | $s_i$ | $e^{s_i}$ | 相对值 |
|---|---|---|---|
| $s_{\text{fi}}$ | 3 | $e^3 \approx 20.1$ | **20.1x** |
| $s_{\text{cash}}$ | 2 | $e^2 \approx 7.4$ | 7.4x |
| $s_{\text{gold}}$ | 1 | $e^1 \approx 2.7$ | **2.7x** |

**问题**: 固收 $e^3$ 是黄金 $e^1$ 的 **7.4 倍**!V2 设计中黄金 35% 与固收 45% 相当, V3 的硬编码公式完全反转了这个意图。

**根因**: 固收得分 $-m - s + \text{equity\_stress}$ 在纯 crisis 下累加 3 个正贡献, 黄金得分 $r$ 只有一个, **crisis 信号通道分配不平衡**。

### 5.3 缺陷三: AE Gain-Multiplier Paradox (增益乘数悖论)

V3 设计 $\theta_{\text{eff}} = \theta \cdot (1 + 0.5 \cdot \text{bear\_pressure})$ 放大 $\theta$, 但放大的对象是 LLM 信号产生的得分。

**反例** (牛市末端 + AE 警报):
- LLM: $m=+0.7, s=+0.7, r=-0.7$ (强烈看多)
- AE: $E_t=30, \tau=20 \Rightarrow \text{bear\_pressure} = 0.73$
- $\theta_{\text{eff}} = 1.365$, $s_{\text{broad}} = 2.1$
- $b_{\text{broad}} \propto e^{1.365 \cdot 2.1} = e^{2.87} \approx 17.6$

**问题**: **AE 应该强制切换到防御模式, 但 $\theta$ 增益实际上放大了 LLM 的多头得分**。**AE 警报反而强化了 LLM 的看多仓位**。

**根因**: AE 应该改变 **b 分布的中心** (从进攻移到防御), 不是放大 **b 分布的扩散程度** (方差)。`theta_eff` 是 variance multiplier, 与 "regime 切换" 语义正交。

### 5.4 三个缺陷的内核共性

> **V3 用 "加法组合 + exp 放大" 试图同时表达 "regime 切换" 和 "regime 内微调"**, 但这两件事需要 **不同的数学结构**:
> - regime 切换 = 中心位移 (均值改变)
> - regime 内微调 = 方差放大或信号强度
>
> V3 把两者都映射到 $\theta_{\text{eff}}$ 单一旋钮上, 导致:
> - 尺度陷阱(1) = 加法组合内部系数分配错误
> - 黄金悲剧(2) = 通道分配不平衡导致中心被压
> - AE 悖论(3) = 用 variance 旋钮表达 center 切换的必然失败

**修复原则**: 把 "regime 切换" 从 "信号微调" 中 **完全分离**。前者是 V_DEFENSE 模板的位移, 后者是 exp 的方差。

---

## 六、EventTrack V3.1 详细设计: 审计修复版

### 6.1 修复 1: 得分向量的矩阵化标准化 (Matrix-Normalized Scores)

**核心思路**: 用 $5 \times 5$ 敏感度矩阵 $\mathbf{W}$ 替代加法组合, 确保每行 L1 范数 ≤ 1, 所有 $s_i \in [-1, +1]$。

$$
\boxed{\mathbf{s} = \mathbf{W} \cdot \mathbf{f}, \quad \mathbf{f} = [m, s, r, \text{equity\_stress}, \text{sat\_lead}]^T \in [-1, 1]^5}
$$

```python
W = np.array([
    # m     s     r     eq    sat
    [ 1/3,  1/3, -1/3,  0.0,  0.0],   # broad:  (m+s) - r
    [ 1/4,  1/4, -1/4,  1/4,  0.0],   # sat:    (m+s) - r + sat_lead/4
    [-1/3, -1/3,  0.0,  1/3,  0.0],   # fi:     -(m+s) + eq_stress/3
    [-1/3,  0.0,  2/3,  0.0,  0.0],   # gold:   -m/3 + 2r/3
    [-1/2,  0.0,  1/2,  0.0,  0.0],   # cash:   -m/2 + r/2
], dtype=float)
```

**每行 L1 范数** = 1.0 ✓ (broad: 3·(1/3); sat: 4·(1/4); fi: 3·(1/3); gold: 1/3+2/3; cash: 1/2+1/2)

**统一上限保证**: $\forall \mathbf{f} \in [-1, 1]^5, |s_i| \leq 1$。这意味着指数偏置 $e^{\theta \cdot s_i}$ 的最大比值是 $e^{2\theta}$, **所有资产的信号强度在同一量级**。

**修复验证** (纯 crisis $m=-1, s=-1, r=+1, \text{eq}=+1, \text{sat}=0$):
- $s_{\text{fi}} = 1/3 + 1/3 + 1/3 = 1.0$ ✓
- $s_{\text{gold}} = 1/3 + 2/3 = 1.0$ ✓
- $s_{\text{cash}} = 1/2 + 1/2 = 1.0$ ✓
- $s_{\text{broad}} = s_{\text{sat}} = 0$ (中性)

$b_{\text{fi}} = b_{\text{gold}} = b_{\text{cash}}$ 三者并列, equity 完全压制。**V3 的黄金悲剧 $e^3 : e^1 = 20:1$ 被消除**。

### 6.2 修复 2: AE 由 "增益乘数" 改为 "位移器" (Shifter, NOT Gain)

**核心思路**: AE 不再放大 $\theta$, 而是 **直接把得分向量 $\mathbf{s}$ 拉向防御模板 $\mathbf{V}_{\text{DEFENSE}}$**。

$$
\boxed{
\begin{aligned}
\text{bear\_pressure} &= \sigma\!\left(\frac{E_t - \tau}{\tau \cdot 0.5}\right) \in [0, 1] \\
\mathbf{s}_{\text{final}} &= (1 - \text{bear\_pressure}) \cdot \mathbf{s} + \text{bear\_pressure} \cdot \mathbf{V}_{\text{DEFENSE}}
\end{aligned}
}
$$

其中 $\mathbf{V}_{\text{DEFENSE}} = [-1, -1, +0.5, +1.0, +0.5]$: equity 完全压 (-1), 固收/现金中等避险 (+0.5), 黄金最大避险 (+1.0)。

**Shifter vs Gain 关键区别**:
- **Gain** (乘数): $\theta_{\text{eff}} = \theta \cdot (1 + 0.5 \cdot \text{bear\_pressure})$。无论 LLM 说什么, AE 都 **放大** 差异。LLM 强烈看多时, AE 反而 **强化** 多头 (V3 悖论)。
- **Shifter** (位移): $\mathbf{s}_{\text{final}}$ 是 $\mathbf{s}$ 与 $\mathbf{V}_{\text{DEFENSE}}$ 的凸组合。无论 LLM 说什么, AE 都 **拉向** 防御。**信号方向不决定 AE 行为**, 只决定 "拉多少"。

**关键性质**: $\text{bear\_pressure} \to 1 \Rightarrow \mathbf{s}_{\text{final}} \to \mathbf{V}_{\text{DEFENSE}}$。这是数学上的 "硬切换极限": AE 警报足够强时, **无论 LLM 说什么, 得分都是 V_DEFENSE**。

**修复验证** (牛市末端 + AE 警报):
- LLM: $m=+0.7, s=+0.7, r=-0.7 \Rightarrow \mathbf{s}_{\text{no\_AE}} = [+0.7, +0.7, -0.7, -0.7, 0.0]$
- AE: $\text{bear\_pressure} = 0.73$
- $\mathbf{s}_{\text{final}} = 0.27 \cdot [+0.7, +0.7, -0.7, -0.7, 0.0] + 0.73 \cdot [-1, -1, 0.5, 1.0, 0.5] = [-0.54, -0.54, +0.18, +0.54, +0.37]$

修复后 equity 显著为负、避险为正。**AE 警报压过了 LLM 看多**, 与 V3 行为完全相反。

### 6.3 修复 3: b-as-Policy (放弃 RB, 直接用 exp-tilting 后的 b 作为权重)

**核心发现**: V2/V3 都用 RB 把 $\mathbf{b}$ 投影到 $\mathbf{w}$。但 RB 的核心是 "风险贡献等化", **它本质上偏好低波动率资产** (低 $\sigma$ 需要高 $w$ 才能达到 $b_i \cdot \sigma_p^2$ 的风险贡献)。

在 V3.1 的 $\mathbf{V}_{\text{DEFENSE}}$ 下:
- $b_{\text{gold}} \propto e^{+1.0} = 2.72$ (意图: 黄金主导)
- $b_{\text{fi}} \propto e^{+0.5} = 1.65$

理论上 $b_{\text{gold}} / b_{\text{fi}} = 1.65$。但 RB inverse-vol 会把固收的 $\sigma$ 优势转为 $w$ 优势, 最终 $w_{\text{fi}} > w_{\text{gold}}$, **直接背叛 V_DEFENSE 模板的战略意图**。

**修复**: V3.1 放弃 RB, 直接用 exp-tilting 后的 $\mathbf{b}$ 作为 $\mathbf{w}$ (经过 box 约束投影 + 归一化):

```python
# V3.1 pipeline
scores = W @ f                                       # Fix 1: matrix-normalized
scores = (1 - bp) * scores + bp * V_DEFENSE          # Fix 2: AE shifter
b = B0 * exp(THETA * scores) / sum(...)              # exp-tilting
w = project_to_box_simplex(b, BOUNDS)                 # Fix 3: b-as-policy, no RB
```

**代价**: V3.1 不再做 "风险贡献等化", 风险平衡靠 bounds 间接约束。V3.1 的 BOUNDS:
- 黄金上限 0.40 (允许 bear 时达到 35%)
- 固收上限 0.50 (避免固收逆 vol 压制黄金)
- 现金上限 0.20 (避免过度现金)
- 宽基下限 0.05 (确保最低 exposure)

**优势**: b-as-policy 让 $\mathbf{s}$ 的设计意图完整传递到 $\mathbf{w}$, **没有 RB 这个 "信号失真器"**。

### 6.4 V3.1 端到端伪代码

```python
class EventTrackV31:
    W = np.array([...])  # 5x5 matrix, see 6.1
    V_DEFENSE = np.array([-1, -1, 0.5, 1.0, 0.5])
    B0 = np.array([0.2]*5)
    THETA = 1.0
    BOUNDS = [(0.05, 0.50), (0.00, 0.45), (0.00, 0.50),
              (0.00, 0.40), (0.00, 0.20)]

    def compute(self, R, d1, d2, d3, ae_error=None, tau=None):
        # 1. 安全 σ
        sigmas = safe_std(R, axis=1, ddof=1, fallback=1e-3)

        # 2. LLM 归一化 + 市场特征
        m = clip((d1-50)/50, -1, 1)
        s = clip((d2-50)/50, -1, 1)
        r = clip((d3-50)/50, -1, 1)
        eq = clip((sigmas[0]+sigmas[1])/(sigmas[2]+sigmas[3]+1e-9) - 1, 0, 2) / 2
        sat = clip((sigmas[1]-sigmas[0])/(sigmas[1]+sigmas[0]+1e-9), -1, 1)

        # 3. Fix 1: 矩阵化得分 (|s_i| <= 1)
        f = np.array([m, s, r, eq, sat])
        scores = self.W @ f

        # 4. Fix 2: AE shifter (位移, 不是增益)
        if ae_error and tau and tau > 0:
            bp = sigmoid((ae_error - tau) / (tau * 0.5))
            scores = (1 - bp) * scores + bp * self.V_DEFENSE

        # 5. 指数偏置
        b = self.B0 * np.exp(self.THETA * scores)
        b = b / b.sum()

        # 6. Fix 3: b-as-policy, 投影到 box 单纯形
        if R.shape[1] < MIN_SAMPLES:
            return _normalize(b)
        return _project_to_box_simplex(b, self.BOUNDS)
```

### 6.5 V3.1 vs V3 黄金悲剧回归验证

| 指标 | V3 (有缺陷) | V3.1 (修复) | 变化 |
|---|---|---|---|
| bear weeks $w_{e,\text{gold}}$ | 0.164 | **0.341** | +108% |
| bear weeks $w_{e,\text{gold}} / w_{e,\text{fi}}$ | 0.83 | **1.37** | 黄金反超固收 |
| V3.1 强 bear $w_e$ | (0.05, 0.05, 0.45, 0.16, 0.10) | (0.05, 0.05, 0.25, 0.40, 0.24) | 黄金主导, 现金次之 |

### 6.6 V3.1 的诚实局限

V3.1 **不是** "全面优于 V3", 而是 "在 §五 缺陷维度上优于 V3, 其他维度可能略弱":

**V3.1 优势** (对应 §五 缺陷):
1. **尺度平衡** ($|s_i| \leq 1$): 卫星无法压过黄金, 信号方向决定 b 方向
2. **黄金不悲剧** (纯 crisis 下 $b_{\text{gold}} = b_{\text{fi}} = b_{\text{cash}}$): AE shifter 在强 bear 时把黄金拉到 1.0, 实现 $w_{\text{gold}} > w_{\text{fi}}$
3. **AE 不悖论** (shifter 而非 gain): 牛市末端 + AE 警报, AE 强制把得分拉向 V_DEFENSE, 不放大 LLM 多头

**V3.1 局限**:
1. **edge std 0.24% < V3 0.43%**: b-as-policy 比 RB 更稳定 (weekly std 0.027 < 0.068), PPO fusion 的 "切换价值" 降低
2. **Pure EventTrack Sharpe 0.576 < V3 0.947**: V3.1 单一 track 不如 V2/V3, 但作为基础轨更稳定
3. **stagnflation coverage 154 周过度**: b-as-policy 让 fi 上限被 box 限制在 0.50, 大部分周 fi < 0.40 被错误归为 "滞胀"
4. **bull 较保守**: bull weeks $w_e$ offensive 0.323 (V3: 0.490), 无 V3 的 "过度加仓" 现象 (这其实是优势, 不是缺陷)

### 6.7 V3.1 调参指南 (Stage 4+ 待办)

| 参数 | 默认 | 调整方向 | 影响 |
|---|---|---|---|
| $\mathbf{W}$ 矩阵行 | 见 6.1 | 调单个 $W_{ij}$ | 改变 "哪个信号驱动哪个资产" |
| $\mathbf{V}_{\text{DEFENSE}}$ 黄金分量 | 1.0 | ↑ 1.5 | 强化黄金在 bear 时的优势 |
| `THETA` | 1.0 | ↑ 1.5 | 增强 LLM 信号对 b 的影响 |
| `AE_SIGMOID_SCALE` | 0.5 | ↑ 0.3 (锐化) | 接近硬切换 |
| BOUNDS 黄金上限 | 0.40 | ↑ 0.50 | 让 V2 bear 形态 0.35 黄金更稳定 |
| BOUNDS 固收上限 | 0.50 | ↓ 0.40 | 限制逆 vol 压制黄金 |

### 6.8 V3.1 复现 Checklist

| # | 步骤 | 验证 |
|---|---|---|
| 1 | `W` 矩阵每行 L1 ≤ 1 | `test_v31_all_scores_bounded_in_unit_range` |
| 2 | 纯 crisis $s_{\text{gold}} = s_{\text{fi}} = 1$ | `test_v31_gold_crisis_reaches_max` + `test_v31_fi_crisis_reaches_max` |
| 3 | 强 bear $w_{\text{gold}} \geq w_{\text{fi}}$ | `test_v31_gold_dominates_in_bear_regime` |
| 4 | AE shifter 强制防御 | `test_v31_bear_pressure_forces_defense` |
| 5 | 牛市末端 + AE 警报 → 防御 | `test_v31_bull_end_ae_bear_forces_defense` |
| 6 | 无 AE 时不触发 shifter | `test_v31_no_ae_no_shifter` |
| 7 | 危机 LLM 信号下黄金并列 | `test_v31_crisis_llm_ties_gold_and_fi` |
| 8 | V3.1 仍能表达 V2 凸包外 | `test_v31_stagflation_against_v2` |
| 9 | V3.1 bear 形态与 V2 接近 | `test_v31_bear_matches_v2_bear_shape` |
| 10 | Bounds 严格执行 | `test_v31_bounds_enforced` |
| 11 | Bounds 不对称 (无统一) | `test_v31_bounds_asymmetric` |
| 12 | 收敛性 | `test_v31_solver_converges` |
| 13 | 样本不足回退 | `test_v31_insufficient_samples_fallback` |
| 14 | V_DEFENSE 各项 ∈ [-1, 1] | `test_v31_v_defense_norm_unit` |
| 15 | 三个 regime 显著不同 | `test_v31_three_regimes_smoothly_separated` |
| 16 | V3.1 vs V3 黄金悲剧回归 | `test_v31_no_v3_gold_tragedy` |

### 6.9 V3.1 全流程数学推导 (从 $\mathbf{R}, d_1, d_2, d_3, E_t, \tau$ 到 $\mathbf{w}^E$)

> 本节给出 V3.1 端到端数学推导。所有公式与 `src/compute/event_track_v3_1.py` 一一对应, 配 §6.9.10 数值例子与 V3 同输入对比, 直观展示 3 个修复的成效。

#### 6.9.1 输入输出

**输入**: $\mathbf{R} \in \mathbb{R}^{5 \times T}$ (5 资产日收益矩阵, 默认 $T=20$), $d_1, d_2, d_3 \in [0, 100]$ (LLM 三分量), $E_t \in \mathbb{R}$ (AE 重建误差, 可选), $\tau \in [5, 50]$ (regime 阈值, 可选).

**输出**: $\mathbf{w}^E \in \mathbb{R}^5$, $w^E_i \geq 0$, $\sum_i w^E_i = 1$.

#### 6.9.2 Step 1 — 安全标准差 (同 V3)

$$
\sigma_i = \begin{cases} \mathrm{std}(R_{i,:};\ \mathrm{ddof}=1) & \text{若 std 有限且} > 0 \\ 10^{-3} & \text{否则} \end{cases}
$$

#### 6.9.3 Step 2 — LLM 信号归一化 (同 V3)

$$
m = \mathrm{clip}\!\left(\frac{d_1 - 50}{50},\ -1,\ 1\right), \quad s = \mathrm{clip}\!\left(\frac{d_2 - 50}{50},\ -1,\ 1\right), \quad r = \mathrm{clip}\!\left(\frac{d_3 - 50}{50},\ -1,\ 1\right)
$$

#### 6.9.4 Step 3 — 市场结构特征 (同 V3)

$$
\mathrm{equity\_stress} = \frac{1}{2} \cdot \mathrm{clip}\!\left( \frac{\sigma_0 + \sigma_1}{\sigma_2 + \sigma_3 + \varepsilon} - 1,\ 0,\ 2 \right), \quad \mathrm{sat\_lead} = \mathrm{clip}\!\left( \frac{\sigma_1 - \sigma_0}{\sigma_1 + \sigma_0 + \varepsilon},\ -1,\ 1 \right)
$$

#### 6.9.5 Step 4 — 矩阵化得分 (**Fix 1**: Matrix-Normalized Scores)

V3.1 用 $5 \times 5$ 敏感度矩阵 $\mathbf{W}$ 替代 V3 的加性组合:

$$
\boxed{\mathbf{s} = \mathbf{W} \cdot \mathbf{f}, \quad \mathbf{f} = [m, s, r, \mathrm{equity\_stress}, \mathrm{sat\_lead}]^T \in [-1, 1]^5}
$$

$$
\mathbf{W} = \begin{bmatrix} 1/3 & 1/3 & -1/3 & 0 & 0 \\ 1/4 & 1/4 & -1/4 & 1/4 & 0 \\ -1/3 & -1/3 & 0 & 1/3 & 0 \\ -1/3 & 0 & 2/3 & 0 & 0 \\ -1/2 & 0 & 1/2 & 0 & 0 \end{bmatrix}, \quad \begin{array}{l} \text{broad: } (m+s)/3 - r/3 \\ \text{sat: } (m+s)/4 - r/4 + \mathrm{eq}/4 \\ \text{fi: } -(m+s)/3 + \mathrm{eq}/3 \\ \text{gold: } -m/3 + 2r/3 \\ \text{cash: } -m/2 + r/2 \end{array}
$$

**每行 L1 范数 = 1.0** (broad: $3 \cdot 1/3$; sat: $4 \cdot 1/4$; fi: $3 \cdot 1/3$; gold: $1/3 + 2/3$; cash: $1/2 + 1/2$).

**关键性质**: $\forall \mathbf{f} \in [-1, 1]^5, |s_i| \leq \sum_j |W_{ij}| \leq 1$。这意味着 $e^{\theta \cdot s_i}$ 的最大比值是 $e^{2\theta}$(任意两个资产之间), **所有资产信号强度同量级**, 消除 V3 的尺度陷阱。

#### 6.9.6 Step 5 — AE Shifter (**Fix 2**: 位移, 不是增益)

V3.1 把 $\text{bear\_pressure}$ 直接用于 **得分向量的位移** (而非 $\theta$ 的增益):

$$
\boxed{
\begin{aligned}
\text{bear\_pressure} &= \sigma\!\left(\frac{E_t - \tau}{\tau \cdot 0.5}\right) \in [0, 1] \\
\mathbf{s}_{\text{final}} &= (1 - \text{bear\_pressure}) \cdot \mathbf{s} + \text{bear\_pressure} \cdot \mathbf{V}_{\text{DEFENSE}}
\end{aligned}
}
$$

其中 $\mathbf{V}_{\text{DEFENSE}} = [-1, -1, 0.5, 1.0, 0.5]^T$ (equity 完全压, 黄金最大避险, fi/cash 中等避险).

**关键性质**:
- $\text{bear\_pressure} \to 0$: $\mathbf{s}_{\text{final}} = \mathbf{s}$ (纯 LLM 驱动)
- $\text{bear\_pressure} = 1$: $\mathbf{s}_{\text{final}} = \mathbf{V}_{\text{DEFENSE}}$ (硬切换极限, **无论 LLM 说什么**)
- **凸组合**保证 $s_{\text{final}, i} \in [-1, 1]$ (因 $\mathbf{s}, \mathbf{V}_{\text{DEFENSE}}$ 各分量都在 $[-1, 1]$)

这修复了 V3 的 AE Gain 悖论: **AE 永远拉向防御, 不放大 LLM 多头**。

#### 6.9.7 Step 6 — 指数偏置 (同 V3, $\theta = 1.0$)

$$
b_i = \frac{b_{0,i} \cdot e^{\theta \cdot s_{\text{final}, i}}}{\sum_{j=0}^{4} b_{0,j} \cdot e^{\theta \cdot s_{\text{final}, j}}}, \quad b_0 = (0.2, 0.2, 0.2, 0.2, 0.2)^T, \quad \theta = 1.0
$$

**V3.1 移除**: $\theta_{\text{eff}} = \theta \cdot (1 + 0.5 \cdot \text{bear\_pressure})$ 已被 Fix 2 完全替代。

#### 6.9.8 Step 7 — b-as-Policy (**Fix 3**: 投影代替 RB)

V3.1 **放弃 RB** (其 inverse-vol 偏好会背叛 $\mathbf{V}_{\text{DEFENSE}}$ 战略意图), 直接用 $b$ 作为 $w$, 经 box 单纯形投影 + 归一化:

$$
\boxed{\mathbf{w} = \Pi_{\text{box-simplex}}(\mathbf{b}, \mathrm{BOUNDS})}
$$

其中 $\Pi_{\text{box-simplex}}$ 是迭代 clip + 归一化算法:

$$
\begin{aligned}
w^{(0)} &= \mathrm{clip}(\mathbf{b}, 0, \infty) \\
w^{(k+1)}_i &= \mathrm{clip}(w^{(k)}_i, \mathrm{lo}_i, \mathrm{hi}_i) \\
\mathbf{w}^{(k+1)} &\leftarrow \mathbf{w}^{(k+1)} / \sum_j w^{(k+1)}_j \quad \text{(归一化)} \\
&\text{直到} \sum_j w_j = 1 \text{ 且 } \mathrm{lo}_i \leq w_i \leq \mathrm{hi}_i
\end{aligned}
$$

**V3.1 BOUNDS** (调整以支持 $\mathbf{V}_{\text{DEFENSE}}$ 战略意图):

| 资产 | $\mathrm{lo}$ | $\mathrm{hi}$ | 调整理由 |
|---|---|---|---|
| broad | 0.05 | 0.50 | 最低 5% exposure |
| sat | 0.00 | 0.45 | bear 时可完全去除 |
| fi | 0.00 | 0.50 | 限制逆 vol 压制黄金 (V3 是 0.60) |
| gold | 0.00 | 0.40 | 让 $\mathbf{V}_{\text{DEFENSE}}$ 黄金 35% 稳定 (V3 是 0.30) |
| cash | 0.00 | 0.20 | 避免过度现金 (V3 是 0.15) |

#### 6.9.9 Step 8 — 投影失败回退

若 $T < \mathrm{MIN\_SAMPLES} = 2$ 或投影不收敛, 返回 `b / sum(b)`。

#### 6.9.10 数值例子 (V3.1 端到端, 与 §4.6.10 同输入直接对比)

**输入**: $d_1=0, d_2=0, d_3=100$ (极端 risk-off), $\sigma = (0.012, 0.015, 0.003, 0.010, 0.001)$.

| Step | 计算 | 结果 |
|---|---|---|
| Step 2 LLM 归一化 | $m=-1, s=-1, r=+1$ | $m=-1, s=-1, r=+1$ |
| Step 3 特征 | $(0.027/0.013 - 1)/2$ | $\mathrm{eq} = 0.538, \mathrm{sat\_lead} = 0.111$ |
| **Step 4 矩阵得分 (Fix 1)** | $\begin{aligned} s_b &= (1/3)(-1) + (1/3)(-1) + (-1/3)(+1) = -1.0 \\ s_s &= (1/4)(-1) + (1/4)(-1) + (-1/4)(+1) + (1/4)(0.538) = -0.615 \\ s_f &= (-1/3)(-1) + (-1/3)(-1) + (1/3)(0.538) = 0.846 \\ s_g &= (-1/3)(-1) + (2/3)(+1) = 1.0 \\ s_c &= (-1/2)(-1) + (1/2)(+1) = 1.0 \end{aligned}$ | $\mathbf{s} = [-1.0, -0.615, 0.846, 1.0, 1.0]^T$ |
| | **V3 同输入**: $\mathbf{s}_{V3} = [-3, -2.889, 2.538, 1, 2]^T$ (卫星 -2.889 远高于黄金 1) | **V3.1 修复**: 卫星 0.846 ≈ 黄金 1.0 (无尺度压制) |
| **Step 5 AE Shifter (Fix 2)** | 假设 $\text{bear\_pressure}=1$ (强 bear) | $\mathbf{s}_{\text{final}} = \mathbf{V}_{\text{DEFENSE}} = [-1, -1, 0.5, 1, 0.5]$ |
| Step 6 指数偏置 | $e^{\mathbf{V}_{\text{DEFENSE}}} = [0.368, 0.368, 1.649, 2.718, 1.649]$ | $Z = \sum e^{s_i} = 6.751$ |
| | $b_i = 0.2 \cdot e^{s_i} / Z$ 等价于 $b_i = e^{s_i} / Z$ | $b = [0.0545, 0.0545, 0.244, 0.403, 0.244]^T$ |
| | **V3 同步骤**: $b_{V3} = [0.000032, 0.000038, 0.129, 0.0129, 0.0577]^T$ | **V3.1 修复**: $b_g = 0.403$ > $b_f = 0.244$ (黄金主导) |
| **Step 7 b-as-Policy (Fix 3)** | $b$ 全部在 BOUNDS 内, 直接归一化 | $w = b / \mathrm{sum}(b) = w$ |
| | $\mathbf{w}^E$ | $w = [0.0545, 0.0545, 0.244, 0.403, 0.244]^T$ |

**V3.1 在纯 crisis + 强 bear 下的 $\mathbf{w}^E$**:
- $w_g / w_f = 0.403 / 0.244 = \mathbf{1.65}$ ← **黄金主导, 修复 V3 的 $b_g/b_f = 0.10$ 黄金悲剧**
- $w_c / w_f = 1.0$ (cash = fi, 符合 V_DEFENSE 0.5 = 0.5)
- 防御类 (fi + gold + cash) = 89%, equity 11%

**V3 同输入 (经 RB)**: $w = [0.05, 0.05, 0.45, 0.16, 0.10]$ (固收主导, 黄金悲剧)。**V3.1 黄金从 16% 升到 40% (2.5x), 固收从 45% 降到 24%, 防御总仓位从 71% 升到 89%**。

#### 6.9.11 V3 vs V3.1 同输入对照表

| 维度 | V3 | V3.1 |
|---|---|---|
| **Step 4 卫星得分** | -2.889 (L∞=4) | -0.615 (L∞=1) ← **Fix 1** |
| **Step 4 黄金得分** | +1.0 (L∞=1) | +1.0 (L∞=1) |
| **Step 4 卫星/黄金比** | 2.889:1 (卫星压黄金 2.9x) | 0.615:1 (黄金反超卫星) |
| **Step 5 AE 模式** | $\theta_{\text{eff}} = 1.5$ (增益) | $\mathbf{s}_{\text{final}} = \mathbf{V}_{\text{DEFENSE}}$ (位移) ← **Fix 2** |
| **Step 6 黄金 b** | 0.0129 | 0.403 (31x V3) |
| **Step 6 固收 b** | 0.129 | 0.244 (1.9x V3) |
| **Step 7 求解方式** | RB (inverse-vol) | b-as-Policy ← **Fix 3** |
| **Step 7 $w_g / w_f$** | 0.36 (黄金悲剧) | **1.65** (黄金主导) |
| **Step 7 $w_{\text{defensive}}$** | 71% | **89%** |

**V3.1 3 个修复的成效, 同一个数值例子可视化对比**: 黄金从 16% 上升到 40% (V3 的 2.5x), 固收从 45% 降到 24% (V3 的 53%), 防御总仓位从 71% 提升到 89%。

---

## 六点十、Stage 7 单轨 V3.1 设计: PPO 控制 $\theta$

### 6.10.1 架构变更

Stage 6 → Stage 7 的关键变化是**删除 NormalTrack + dual_track_engine**, 把 PPO 动作从 $(\Delta\alpha, \Delta\tau)$ 改成 V3.1 的 $\theta$ (指数偏置总增益).

**为什么 NormalTrack 必须删**:
- 5 年 WFO 中 $\alpha \to 1.0$ 在 274/278 周 → `w_fused = w_event` 100% 等价于纯 V3.1
- NormalTrack 的 RB 求解器从未被消费, 是结构死代码
- 单轨设计去掉 ~400 行 (`normal_track.py` 全文 + dual_track 透传)

**为什么 PPO 应该控 $\theta$ 而不是 $\alpha$**:
- 1D 动作空间比 2D 简单, 收敛更快
- $\theta \in [0, 2]$ 的语义清晰: 0=纯 ERC (防御), 1=平衡, 2=激进集中
- AE shifter 仍控制 V_DEFENSE (regime-aware), PPO $\theta$ 控制信号使用强度, 二维解耦

### 6.10.2 $\theta$ 控制的 V3.1 公式

原 V3.1 公式 (Stage 6):
$$
\mathbf{b} = \mathbf{b}_0 \odot \exp(\theta_{\text{static}} \cdot \mathbf{s}_{\text{final}})
$$

Stage 7:
$$
\mathbf{b}_t = \mathbf{b}_0 \odot \exp(\theta_t \cdot \mathbf{s}_{\text{final},t})
$$

其中 $\theta_t$ 由 PPO 推理:
- PPO 状态: $\mathbf{S}_t \in \mathbb{R}^9$ (AE 重建误差 z-score, 市场 vol, LLM d1/d2/d3, 组合 Sharpe/MDD, 遗憾 EMA, $\theta_{t-1}$)
- PPO 动作: $a_t \in [-1, 1]$ (Tanh)
- 映射: $\theta_t = 1.0 + a_t$  (线性, clip 到 $[0, 2]$)

### 6.10.3 新 Reward Function (单轨)

旧 reward (Stage 6, 双轨):
$$
r_t = r_{\text{port}} - \lambda_{\text{turnover}} \cdot \text{TO}_t + \lambda_{\text{relative}} \cdot (r_{\text{port}} - r_{\text{normal}}) - \kappa \cdot \text{MDD}_t + \lambda_{\alpha\text{-direct}} \cdot (\alpha_t - 0.5) \cdot \text{sign}(\text{regime})
$$

新 reward (Stage 7, 单轨):
$$
r_t = r_{\text{port}} - \lambda_{\text{turnover}} \cdot \text{TO}_t - \lambda_{\theta\text{-change}} \cdot |\theta_t - \theta_{t-1}| - \lambda_{\text{MDD}} \cdot \max(0, \text{MDD}_t - \text{MDD}_{\text{target}}) + \lambda_{\text{signal}} \cdot s_t \cdot (\theta_t - 1)
$$

其中 $s_t$ 是信号强度 ∈ $[0, 1]$:
$$
s_t = \frac{1}{2}\left(\frac{|d_1 - 50| + |d_2 - 50| + |d_3 - 50|}{150} + \min\left(\frac{|E_t - \tau|}{30}, 1\right)\right)
$$

### 6.10.4 数值例子 (Stage 7 PPO 推理)

给定 2024-04 某周, PPO 状态:
| 维度 | 值 | 含义 |
|---|---|---|
| $E_t$ z-score | 0.8 | AE 重建误差偏高 |
| vol_mkt_20d | 0.4 | 市场波动中等 |
| llm_macro | 0.2 | LLM 宏观偏多 |
| llm_sent | 0.1 | LLM 情绪偏多 |
| llm_risk | 0.5 | LLM 风险中性 |
| port_sharpe | 1.0 | 组合 Sharpe 1.0 |
| port_mdd | 0.05 | 当前回撤 5% |
| regret_ema | 0.3 | 遗憾 EMA 0.3 |
| theta_prev | 0.7 | 上周 θ=0.7 |

PPO actor 输出 $a = -0.4$ → $\theta_t = 1.0 + (-0.4) = 0.6$.

代入 V3.1 公式: b = b0 * exp(0.6 * s_final), 与 Stage 6 的 $\theta=0.7$ 接近但略保守.

### 6.10.5 Stage 7 真实业绩 (与 Stage 6 对比)

| 指标 | Stage 6 (V3.1 θ=0.7 固定) | Stage 7 (PPO 控 θ) | Delta |
|---|---|---|---|
| Sharpe (real OHLC) | 1.096 | 0.741 | -0.355 |
| 年化收益 | 9.70% | 6.58% | -3.11% |
| Max DD | 8.80% | 8.93% | +0.14% |
| Calmar | 1.10 | 0.74 | -0.37 |
| Win rate | 53.86% | 52.45% | -1.41% |

**Stage 7 NO-GO**: 架构正确但 reward 让 PPO 偏好 $\theta \to 0$ (89.6% 周 θ<0.3), 业绩不如 fixed θ=0.7. Stage 7a 调参方向见 §七 第 13 项.

---

## 七、Stage 4+ 决策路径

> **重要更正 (2026-06-07)**: 之前 §七的 Sharpe 数字 (Stage 5: 0.684/0.611, Stage 6: 0.624/0.655) 来自 `metrics.json`, 它有两个 bug:
> 1. `compute_wfo_metrics()` 用 252 (日频) 年化周频数据 → Sharpe 虚高 ≈ 2.2x
> 2. NAV 用 `features_master.parquet` 的 `__weekly_return` 列 (z-scored 后的特征, 范围 -5~5), **不是真实资产回报**
>
> 真实业绩由 `GeneralBacktest` 在真实 OHLC + adj_factor 价格上计算得出, 输出到 `metrics_real.json`。Stage 4-6 全部跑完后, 真实数字如下 (Stage 4 是 simulation-based, 不在此列)。

1. **V3.1 是 V3 缺陷的最小修复版**: 保留 V3 的数学结构, 只改 3 个具体缺陷。
2. **V3.1 真实 Sharpe 显著优于 V3** (~1.10 vs ~0.46, 5 年 WFO): 修复了 V3 的 3 个结构性问题 (scale-mismatch, gold tragedy, AE gain paradox)。
3. **V3.1 是生产主线** (通过 `config.event_track_version: v3_1`)。
4. **Stage 1 ML 修复已完成**: α_max 0.5→0.1, alpha_bias 0→-0.05, switch_bull_reward 0.45→0.05, lambda_alpha_direct 0→0.05→0.15. PPO α 应在 [0.3, 0.7] 真实摆动, 15 个 Stage 1 ML 单测全部通过。
5. **Stage 4 验证已完成 (GO)**: `verify_stage4.py` 在 156 周 (旧样本期 20260416-20260602) 上验证 8 项指标全部通过 (注: Stage 4 用 simulation-based PPO 替代, 不进入真实业绩比较):
   - α 不再锁 1.0 (std=0.193, 范围 [0.00, 0.72])
   - τ 不再锁 20.0 (std=0.21, 范围 [19.16, 19.97])
   - Reward α 梯度: bull corr=+0.938, bear corr=-0.938 (PPO-friendly)
   - V3.1 bear gold/fi=1.26 (黄金主导, 黄金悲剧修复)
   - V3 bear gold/fi=0.83 (V3 仍有黄金悲剧)
6. **Stage 5 5年 WFO 完成 — 真实 OHLC 业绩** (`metrics_real.json`): AE + PPO 重训后, 278 周 (2021-01-01 → 2026-04-24) 真实 PPO 跑:
   - **V3 (prod)**: Sharpe **0.464**, total return **23.46%**, annualized **4.22%**, max DD **15.56%**, Calmar **0.27**, win rate 51.05%
   - **V3.1 (exp)**: Sharpe **1.127**, total return **63.48%**, annualized **10.13%**, max DD **9.31%**, Calmar **1.09**, win rate 53.32%
   - **真实对比**: V3.1 在所有 5 项核心指标上**全面胜 V3** (Sharpe +0.66, Calmar +0.82, max DD -6.25%, total return +40.02%)
   - 决策: V3.1 真实 Sharpe 1.127 已超 1.0 目标, **GO → 切主线到 V3.1** (注意: 这与原 §七基于合成 NAV 的"NO-GO"决策相反, 因为真实业绩 V3.1 远胜 V3)
7. **Stage 5 关键发现 (PPO 训练后行为)**:
   - α mean=0.996, std=0.033: PPO 学到 alpha=1.0 是当前样本期的最优 (Stage 1 fix 让 alpha 能摆动, 但没改变 PPO 学到的偏好)
   - τ mean=11.84, std=4.51, range [5.00, 19.93]: τ 真实摆动, **不再锁 20.0** (Stage 2 bug fix 生效)
   - V3.1 黄金悲剧已修复 (bear gold/fi 1.43 vs V3 0.82), 但 V3.1 的 b-as-policy 让 weekly std 较低, PPO fusion 价值 (edge std) 也较低
8. **建议 (Stage 6 调参方向)**:
   - **短期 (Stage 6)**: 调高 `lambda_alpha_direct` 0.05→0.15 (强化 regime-conditional α 信号), 调低 `THETA` 1.0→0.7 (V3.1 减 sharp), 重训 PPO, 再跑 5 年对比
   - 长期 (Stage 7+): 用 supervised regression 校准 V3.1 的 $\mathbf{W}$ 矩阵 + $\mathbf{V}_{\text{DEFENSE}}$ + ActionMapper 参数, 进一步提升

9. **Stage 6 调参 + 5年 WFO (2000 iter PPO) — 真实 OHLC 业绩** (`metrics_real.json`): `verify_stage6.py` 在 278 周 5年样本上比较 V3 (prod) vs V3.1 (exp):
   - **V3 (prod, Stage 6)**: Sharpe **0.461**, total **23.34%**, annualized **4.20%**, max DD **15.57%**, Calmar **0.27**
   - **V3.1 (exp, Stage 6)**: Sharpe **1.096**, total **60.25%**, annualized **9.70%**, max DD **8.80%**, Calmar **1.10**, bear gold/fi **1.36**
   - **Stage 6 跨版本对比 (真实 OHLC)**:
     - V3.1 Stage 6 vs V3 Stage 6: Sharpe **+0.635** (压倒性胜), max DD -6.77% (胜), total +36.91% (胜)
     - V3.1 Stage 6 vs V3.1 Stage 5: Sharpe **-0.031** (Stage 6 调参让 V3.1 略微退步, 但仍达 1.0+)
     - V3 Stage 6 vs V3 Stage 5: Sharpe -0.003 (lambda_alpha_direct 0.15 对 V3 几乎无影响)
   - **Stage 6 GO/NO-GO (7 项检查全部 PASS)**: V3.1 Sharpe 1.096 ≥ 1.0, V3.1 全面胜 V3 → **GO**
   - **决策**: **V3.1 仍是生产主线** (Stage 6 配置相比 Stage 5 让 V3.1 略退 0.03, 但所有风险指标仍胜 V3)
10. **Stage 6 关键发现 (真实 OHLC)**:
    - **V3.1 真实 Sharpe ≈ 1.10, 真实 annualized return ≈ 9.7-10.1%**: 与 dashboard.png 一致, 这是 5 年真实回测的硬业绩
    - V3 真实 Sharpe ≈ 0.46: 跟 V3.1 不在同一量级, 不再是 production 候选
    - V3.1 Stage 6 bear gold/fi 1.36 (V3 仍是 0.83) → 黄金主导, 防御性更强
    - edge std 是合成 NAV 的指标, 与真实 Sharpe 不直接对应 (V3 edge std 0.46% 但 Sharpe 0.46, V3.1 edge std 0.28% 但 Sharpe 1.10; 关键看真实收益分布而非合成 NAV 噪声)
11. **最终主线: V3.1 (Stage 6 配置)**:
    - config.event_track_version: v3_1
    - config.reward_function.lambda_alpha_direct: 0.15
    - src/compute/event_track_v3_1.py THETA: 0.7
    - PPO: 2000 iter 重训 (Stage 6 checkpoint 在 checkpoints/actor_critic.pth)
    - **真实业绩基线 (Stage 6 V3.1)**: Sharpe 1.096, 年化 9.70%, max DD 8.80%
12. **Stage 7 架构精简完成**: 删除 NormalTrack + dual_track_engine + alpha/tau 融合层, 改 PPO 直接控制 V3.1 的 $\theta$ (指数偏置总增益). 见 §6.10 Stage 7 单轨设计.
    - **架构改动**:
      - 删除: `src/compute/normal_track.py`, `src/compute/dual_track_engine.py` → 移到 `archive/pre_theta_refactor_20260607/deprecated_src/`
      - 新增: `src/compute/v31_engine.py` (薄包装), `src/env/action_mapper.py` 加 `ThetaActionMapper`, `src/env/reward_function.py` 重写为单轨 reward, `src/env/state_assembler.py` 9 维 state, `src/env/mdp_environment.py` 1 维 action
      - V3.1.compute() 接受 `theta` 参数 (0=纯 ERC, 1=平衡, 2=激进)
      - PPO: state 10→9 (去 alpha/tau_prev, 加 theta_prev), action 2→1
    - **真实 OHLC 业绩对比 (Stage 6 V3.1 fixed θ=0.7 vs Stage 7 PPO-θ)**:
      | | Sharpe | 年化收益 | Max DD | Calmar | Win Rate |
      |---|---|---|---|---|---|
      | Stage 6 (V3.1 θ=0.7 固定) | 1.096 | 9.70% | 8.80% | 1.10 | 53.86% |
      | **Stage 7 (PPO 控 θ)** | **0.741** | **6.58%** | **8.93%** | **0.74** | **52.45%** |
    - **PPO 学到的行为**: theta 89.6% 周 < 0.3 (几乎纯 ERC), bear 周 mean θ=0.27, bull 周 mean θ=0.004. PPO 倾向于"低 θ + 少变化"的稳态, 因为 $\lambda_{\theta\text{-change}}=0.005$ 强于 $\lambda_{\text{signal}}=0.01$, 错过了 V3.1 在 θ=0.7 的优势.
    - **Stage 7 NO-GO (业绩)**: 架构正确 (单轨 + PPO 控 θ + state 9 维), 但 reward 设计使 PPO 偏好 θ≈0, 业绩不如 fixed θ=0.7.
13. **Stage 7a 待办 (Reward 调参)**:
    - 调 $\lambda_{\theta\text{-change}}$ 0.005→0.001 (允许更大波动)
    - 调 $\lambda_{\text{signal}}$ 0.01→0.05 (信号强时高 θ 收益翻倍)
    - 加 theta_baseline bonus: $+0.01 \cdot (1 - |\theta - 1.0|)$, 把 θ 拉回中性
    - 离线 RL 预训练 5y 历史 θ, 再做在线 PPO

---

## 附录 X: 业绩数据来源说明 (2026-06-07 更正)

| 字段 | `metrics.json` (旧) | `metrics_real.json` (新) |
|---|---|---|
| 计算引擎 | WFO 循环内 NAV 累乘 (周频) | `GeneralBacktest` (日频 OHLC) |
| 价格数据 | `features_master.parquet` 的 `__weekly_return` (z-scored 特征) | 真实 OHLC + adj_factor from ClickHouse |
| 费率/滑点 | 无 | 双边 0.03% 佣金 + 0.01% 滑点 |
| 阈值再平衡 | 无 | 0.5% 权重漂移阈值 |
| 年化系数 | 错误: `* 252` (日频) 用于周频数据 | 正确: `* 252` (日频) |
| Sharpe 可信度 | **不可信** (Sharpe 虚高 ≈ 2.2x, 数据本身是 z-scored) | **可信** (真实资产回报) |

**结论**: 所有 GO/NO-GO 决策应基于 `metrics_real.json`。`metrics.json` 仅保留作为 WFO 循环完整性检查的副产物。

---

## 附录 A: 关键文件路径 (V3 / V3.1 主线)

| 文件 | 状态 | 说明 |
|---|---|---|
| `src/compute/event_track_v3.py` | 备选 | V3 Exponential Tilting (15 unit tests, 已被 V3.1 替代为生产) |
| `src/compute/event_track_v3_1.py` | **生产 (Stage 6)** | V3.1 矩阵化得分 + AE shifter + b-as-policy, THETA=0.7 (19 unit tests) |
| `src/compute/event_track_v2.py` | 保留 (历史) | V2 Signal-Tilted Risk Budgeting (12 unit tests, archived) |
| `src/compute/event_track.py` | 保留 (历史) | V1 三原型 softmax (Stage 2 测试) |
| `src/compute/dual_track_engine.py` | 改 | `use_v3_1` 参数优先级最高, `use_v3` / `use_v2` 备用. WFO 通过 `config.event_track_version` 切换 |
| `src/env/reward_function.py` | **Stage 1** | `lambda_alpha_direct` 接入 compute(), 引入 regime-conditional alpha signal |
| `config.yaml` | **Stage 1** | `alpha_max` 0.5→0.1, `alpha_bias` 0→-0.05, `tau_delta_range` 2.0→0.1, `switch_bull_reward` 0.45→0.05, `lambda_alpha_direct` 0→0.05, `lambda_endpoint` 0.20→0.05, 新增 `event_track_version: v3` |
| `src/compute/normal_track.py` | 共享 | V1/V2/V3/V3.1 共用, 防御轨 |
| `tests/test_event_track_v3.py` | 保留 | V3 15 个单元测试 |
| `tests/test_event_track_v3_1.py` | **核心** | V3.1 18 个单元测试 (含 3 个审计回归) |
| `tests/test_ml_stage1_fix.py` | **核心** | Stage 1 ML 15 个单元测试 (lambda_alpha_direct, action_mapper bias, MDP env 验证) |
| `verify_stage4.py` | **核心** | Stage 4 验证 harness (no PPO, 8 项 GO/NO-GO 检查) |
| `results/stage4_validation.csv` | **核心** | Stage 4 156 周 V3 vs V3.1 fused 收益 + 权重 |
| `verify_stage5.py` | **核心** | Stage 5 5年 WFO 报告 (V3 vs V3.1, 7 项 GO/NO-GO 检查) |
| `results/wfo/stage5_v3/20260607_163404/` | **核心** | Stage 5 V3 (prod) WFO 输出 (277 周, Sharpe 0.684) |
| `results/wfo/stage5_v3_1/20260607_163543/` | **核心** | Stage 5 V3.1 (exp) WFO 输出 (277 周, Sharpe 0.611) |
| `checkpoints/backup_pre_stage5/` | 备份 | Stage 5 重训前的旧 checkpoint (回滚用) |
| `verify_v3_1_vs_v3_vs_v2_vs_v1.py` | **核心** | 156 周 4-way 离线对照 |
| `results/v3_1_vs_v3_vs_v2_vs_v1_verification.csv` | **核心** | 4 个版本的完整权重 + 收益 |
| `archive/pre_v31_cleanup/` | 归档 | V1/V2 验证脚本、Stage 2 测试、3-way 对照脚本 |

## 附录 B: 关键运行命令

```bash
# V3.1 单元测试 (18 个)
python -m pytest tests/test_event_track_v3_1.py -v

# V3 + V3.1 单元测试 (33 个)
python -m pytest tests/test_event_track_v3.py tests/test_event_track_v3_1.py -v

# ML Stage 1 单元测试 (15 个)
python -m pytest tests/test_ml_stage1_fix.py -v

# 4-way 离线对照 (V1/V2/V3/V3.1)
python verify_v3_1_vs_v3_vs_v2_vs_v1.py

# Stage 4 验证 (156 周, no PPO, 模拟 PPO 替代)
python verify_stage4.py

# Stage 5 报告 (5 年 WFO, V3 vs V3.1, 277 周真实 PPO)
python verify_stage5.py

# Stage 6 报告 (调参后 V3 vs V3.1)
python verify_stage6.py

# AE 重训 (6 年数据, 50 epochs)
python scripts/train_ae.py --epochs 50

# PPO 重训 (Stage 6: 200000 timesteps = 2000 iter)
python scripts/train_ppo.py --total-timesteps 200000

# 完整 WFO 重跑 (需 PPO checkpoint, AE weights, LLM cache, ClickHouse)
# config.event_track_version: v3_1 (Stage 6 生产) | v3 (备选)
python scripts/run_backtest_wfo.py --start-date 2021-01-01 --end-date 2026-04-30 --output-dir results/wfo/stage6_v3_1
python scripts/run_backtest_wfo.py --start-date 2023-01-01 --end-date 2025-12-31
```
