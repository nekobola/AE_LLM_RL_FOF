# AE_LLM_RL_FOF

设计文档可见 https://mcnx64hcm9yb.feishu.cn/wiki/OW5NwsH8rinjrQkN99UcOwH3nHd

传统强化学习（RL）在量化配置中常因状态空间维度爆炸和相关性崩塌导致策略过拟合。本方案创新性地提出了一种“架构解耦、动态路由”的 FOF 智能体工作流。系统剥离了传统 RL 的“选基”与“生成下单权重动作”任务，大语言模型（LLM）降维充当特征工程与宏观风险阻断器，PPO（近端策略优化）升维为系统超参数调度的“元控制器（Meta-Controller）”。该架构在确保极低调仓摩擦成本的同时，实现了平稳期的 Alpha 增厚与危机期的无遗憾（No-Regret）极致防守。


