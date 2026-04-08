# ETF基本信息数据字段说明

## etf.etf_info 表字段

| 字段名 | 中文名 | 说明 |
|--------|--------|------|
| `MasterFundCode` | 基金主代码 | |
| `FullName` | 基金全称 | |
| `InceptionDate` | 成立日期 | |
| `FundID` | 基金ID | 10+自增长序列 |
| `FundCompanyID` | 基金管理公司ID | |
| `FundCompanyName` | 基金管理公司名称 | |
| `CustodianID` | 托管人ID | |
| `Custodian` | 托管人名称 | |
| `OverseasconsultorID` | 境外投资顾问ID | QDII专用，有多个名称合在一起入库，用逗号隔开 |
| `Overseasconsultor` | 境外投资顾问 | QDII专用，有多个名称合在一起入库，用逗号隔开 |
| `CustodianNameID` | 境外资产托管人ID | QDII专用，有多个名称合在一起入库，用逗号隔开 |
| `CustodianName` | 境外资产托管人 | QDII专用，有多个名称合在一起入库，用逗号隔开 |
| `AccountingFirmID` | 会计师事务所ID | |
| `AccountingFirmName` | 会计师事务所名称 | |
| `Accountant` | 经办会计师 | |
| `LawFirmID` | 律师事务所ID | |
| `LawFirmName` | 律师事务所名称 | |
| `Lawyer` | 经办律师 | |
| `ManagementFee` | 管理费率(%) | |
| `CustodianFee` | 托管费率(%) | |
| `InvestmentGoal` | 投资目标 | |
| `InvestmentScope` | 投资范围 | |
| `Benchmark` | 业绩比较基准 | |
| `RiskDescription` | 风险收益特征 | |
| `Strategy` | 投资策略 | |
| `GuaranteePeriod` | 保本期 | |
| `InceptionTNA` | 基金成立规模 | |
| `InceptionShares` | 基金成立时份额 | |
| `Fundstatus` | 基金状态 | 发行、正常、封闭期、结束 |
| `FundTypeID` | 基金运作方式ID | S0501=契约型开放式；S0502=契约型封闭式 |
| `FundType` | 基金运作方式 | 契约型开放式；契约型封闭式 |
| `CategoryID` | 基金类别ID | S0601=股票型基金；S0602=债券型基金；S0603=货币型基金；S0604=混合型基金；S0605=FOF；S0606=股指期货型基金；S0607=商品型基金；S0608=REITs；S0699=其他 |
| `Category` | 基金类别 | 股票型基金；债券型基金；货币型基金；混合型基金；FOF；股指期货型基金；商品型基金；REITs；其他 |
| `InvestmentStyle` | 投资风格 | 如：增值型、收益型、分红型、稳健型、成长型、价值型、积极成长型、中小企业成长型、混合收益型、稳健成长型等 |
| `IsETF` | 是否ETF | 1=是；2=否 |
| `IsLOF` | 是否LOF | 1=是；2=否 |
| `IsQDII` | 是否QDII | 1=是；2=否 |
| `IsUmbrella` | 是否伞型基金 | 1=是；2=否 |
| `IsIndexFund` | 是否指数基金 | 1=是；2=否 |
| `Structrued` | 是否分级基金 | 1=是；2=否 |
| `IsInnovative` | 是否创新型基金 | 1=是；2=否 |
| `IsActiveOrPassive` | 主动标识 | 1=主动；2=被动 |
| `ShortFinancing` | 短期理财标识 | 1=是；2=否 |
| `BackGround` | 公司背景 | 1=原老基金规范、合并、扩募而形成；2=1998年后成立的新基金 |
| `TransExplanation` | 基金转型说明 | 转型基金的说明：由什么基金转换而来，或者转换成什么基金的说明 |
| `ETFCategory` | ETF基金类型 | 股票型ETF；债券型ETF；商品型ETF；货币型ETF；QDII-ETF |

---

## 字段含义速查

**基金标识字段：**
- `MasterFundCode` = 基金主代码（用于关联日频量价数据）
- `FundID` = 内部自增ID

**管理人字段：**
- `FundCompanyID/Name` = 管理公司
- `CustodianID/Name` = 托管人
- `Overseasconsultor` / `CustodianName` = QDII境外机构（逗号分隔多值）

**费率字段：**
- `ManagementFee` = 管理费率(%)
- `CustodianFee` = 托管费率(%)

**分类字段（重要）：**
- `Category` = 基金类别（股票型/债券型/FOF/商品等）
- `IsETF` = 是否ETF
- `IsIndexFund` = 是否指数基金
- `ETFCategory` = ETF细分类型
