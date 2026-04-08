# 自有文本数据库字段说明

## 数据源总览

| 数据库 | 表名 | 类型 | 频率 | 备注 |
|--------|------|------|------|------|
| `etf` | `etf_daily` | 量价数据 | 日频 | |
| `etf` | `etf_info` | 基本信息 | 不定频 | |
| `text_db` | `eastmoney` | 财经新闻 | 不定频，日内新闻 | 东方财富 |
| `text_db` | `sina` | 财经新闻 | 不定频，日内新闻 | 新浪财经 |
| `text_db` | `govcn` | 政府部门信息 | 不定频 | 基于二级行业名称检索得到的中国政府网上的信息 |
| `text_db` | `csrc` | 证监会信息 | 不定频 | 证监会 |
| `text_db` | `zgrmyh` | 央行信息 | 季度 | 中国人民银行货币政策季度例会 |
| `text_db` | `cninfo` | 个股年报 | 不定频，个股年报 | 巨潮资讯网 |
| `text_db` | `stkcd_ind` | 附加-行业代码变更 | 不定频 | 个股的行业代码发生变化时的记录 |

---

## eastmoney（东方财富财经新闻）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `from` | Nullable(String) | 文本数据来源，类似于文本的发布人（可能为空） |
| `url` | String | 数据的URL |
| `title` | String | 标题 |
| `content` | String | 内容 |

---

## sina（新浪财经新闻）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `from` | Nullable(String) | 文本数据来源，类似于文本的发布人（可能为空） |
| `url` | String | 数据的URL |
| `title` | String | 标题 |
| `content` | String | 内容 |

---

## govcn（中国政府网政策信息）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `industry_name` | String | 行业名称；仅数据源为"中国政府网"的表有该字段；采用中国上市公司协会上市公司行业分类的二级分类 |
| `industry_code` | String | 行业代码；仅数据源为"中国政府网"的表有该字段；采用中国上市公司协会上市公司行业分类的二级分类 |
| `title` | String | 标题 |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `url` | String | 数据的URL |
| `content` | String | 内容 |
| `institution` | Nullable(String) | 发文机关（政策库） |
| `policy_id` | Nullable(String) | 发文字号（政策库） |
| `from` | Nullable(String) | 文本数据来源，类似于文本的发布人（可能为空） |
| `theme` | Nullable(String) | 文章主题分类（政策库） |
| `policy_type` | Nullable(String) | 公文种类（政策库） |
| `written_date` | Nullable(String) | 成文日期（政策库） |
| `file_content` | Nullable(String) | 文章内部附件内容（PDF转文字版） |
| `passage_type` | String | 文章所属栏目（政策库、公报、新闻、要闻、政策、联播） |

---

## csrc（证监会信息）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `title` | String | 标题 |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `url` | String | 数据的URL |
| `content` | String | 内容 |
| `from` | Nullable(String) | 文本数据来源，类似于文本的发布人（可能为空） |
| `passage_type` | String | 文章所属栏目（政策解读、证监会要闻） |

---

## zgrmyh（中国人民银行货币政策季度例会）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `title` | String | 会议名称 |
| `from` | String | 文本数据来源，类似于文本的发布人（可能为空） |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 发布时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `url` | String | 数据的URL |
| `content` | String | 内容（可能为空） |

---

## cninfo（巨潮资讯个股年报）

**数据库**: `text_db`

| 字段名 | 数据类型 | 字段含义 |
|--------|----------|----------|
| `uuid` | String | 每条文本数据的唯一标识 |
| `stock_name` | String | 企业名称 |
| `stock_code` | String | 上市公司代码 |
| `title` | String | 年报名称 |
| `date` | Date | 日度时间戳 |
| `date_time` | DateTime | 发布时间戳；如果原始时间戳只精确到日度，则自动补全为该日最后一分钟的时间戳 |
| `url` | String | 数据的URL |
| `content` | String | 内容 |
