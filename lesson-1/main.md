## 概念

generative AI / LLMs

### 1 traditional AI models

#### 1.1 训练流程：

> 举例：垃圾邮件分类

1 数据收集

- 爬虫：爬网页内容

- APIs：调用第三方接口拿数据

- Proprietary data：自己公司的业务数据库

- Buying data：买现成的标注数据集

2 Data Labeling 数据标注

- 1 给每条数据加上“正确答案”（label），这是监督学习的核心。

- 2 数据清洗：删掉噪音（比如乱码、缺字段），修正错误标签，处理重复数据、不平衡样本等

3 模型训练 => 计算资源

- 1 把“特征 + 标签”喂给算法（逻辑回归、随机森林、XGBoost、CNN…）

- 2 算法自动调整内部参数(权重)，让预测结果尽量接近真实标签。

- 3 输出的是一个模型文件 email_spam.model。 由前端来调用。

4 测试

- 测试集来评估模型性能：准确率（accuracy），精准率（Precision），召回率（recall）

- 失败，则继续前三步骤

性能指标讲解：

给出二分类场景：
TP（True Positive）：本来是垃圾邮件，模型也判断为垃圾邮件 ✅

TN（True Negative）：本来是正常邮件，模型也判正常 ✅

FP（False Positive）：本来是正常邮件，却被判成垃圾邮件（误杀）❌

FN（False Negative）：本来是垃圾邮件，却被判成正常邮件（漏掉）❌

4.1 准确率： 整个样本判对的比例

> 极端情况的不准确：100 封邮件，只有 1 封是垃圾邮件。=> 99%。 模型罢工

Accuracy=（TP+TN）/ （TP+TN+FP+FNTP+TN​）

4.2 精准率： 判断为垃圾邮件中有多少是正确的。即：模型判成垃圾邮件的那些邮件里，有多少真的是垃圾的？

Precision=TP / TP​ + FP

4.3 召回率：实际中有的垃圾邮件，多少被识别出来。

Recall=TP / （TP​+FN）

```
## 题目
还是 100 封邮件，其中 10 封是垃圾邮件。

1 模型 A: 把 10 封邮件判为垃圾，其中 9 封是真的垃圾，1 封冤枉了正常邮件

TP = 9, FP = 1, FN = 1（还有 1 封垃圾邮件没识别出来）,TN = 89

计算：

Accuracy = (9 + 89) / 100 = 98%

Precision = 9 / (9 + 1) = 90%

Recall = 9 / (9 + 1) = 90%


2 模型 B: 把 3 封邮件判为垃圾，并且 3 封都对了.

TP = 3, FP = 0,FN = 7（剩下 7 封垃圾邮件它装作看不见）,TN = 90

计算：

Accuracy = (3 + 90) / 100 = 93%

Precision = 3 / (3 + 0) = 100% （它说是垃圾的就一定是垃圾）

Recall = 3 / (3 + 7) = 30% （很多垃圾没抓到）
```

5 上线

- 把模型部署成线上服务：做成 HTTP API: 延迟：一次预测要多长时间（前端用户能不能接受）, 扩容, 并发量, 监控日志

#### 1.2 传统 AI 缺点

1 Expensive

- 算力
- 试错： 改特性，重新跑训练
- 数据成本：标 label

2 知识鸿沟

- 参入者必须为跨学科的复合型人才

3 团队割裂

- 训练模型组 和 业务组（前后端开发） 分离。 业务反馈：模型不行误删邮件 => 训练组 => 收集数据重新训练（时间+金钱）

### 2 LLMs （AI future）

> large language Models is a new approach to AI. 不同于传统模型。关注点在于："自然语言"（NLP = Natural Language Processing）。

#### 2.1 各类 AI 模型（OpenAI） => 提供 APIs。拥有了：

1 自然语言的处理： 对话，知识库检索

2 内容生成： 多模态

3 代码生成

4 autonomous agents： 理解任务，使用工具，完成任务

=> tools

- 内置工具：web search、file search、image generation 等

- 传统 function calling（你在 tools 里定义的函数）

- 远程 MCP server (工具托管中心) 的 tools=> Model Context Protocol

=> remote MCP 的参入

```
## 1 识别：连接 MCP server。 通过 response API配置

import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function run() {
  const response = await client.responses.create({
    model: "gpt-5.1",
    input: "帮我在 Shopify 店铺里查一下 50 英镑左右的露营椅，然后推荐一款。",
    tools: [
      {
        type: "mcp",
        server_label: "shopify",
        server_url: "https://example.com/api/mcp",
      }
    ],
    tool_choice: "auto"  // 让模型按需决定是否调用工具
  });

  console.log(response.output[0].content[0].text);
}

## 2 运行：MCP server 收到请求 → 调用真实业务逻辑 / 三方 API → 返回结果 JSON

run();

```

#### 2.2 分类

cloud LLMs：

- OpenAI：通过 OpenAI API 使用

- Anthropic：通过 Claude API 调用

- Google：通过 Gemini API 调用

Private / Self-Hosted LLMs： => 部署框架：Ollama、OpenLLM 将其变为 API calls

- Meta Llama 系列

- DeepSeek 系列

### 3 两者比较

1 模型不再从零训练

通常用现成的大模型（GPT、Claude 等），省去了大量 Data Collection / Training 的重活。

你主要做的是：Prompt 设计、RAG（检索增强）、工具调用等。

2 数据标签不一定是人工的“真值”

很多场景只是提供上下文，不是经典的 label。

但“数据质量”仍然很关键：提示词、向量库内容、知识库都等价于这里的“Data Labeling”。

3 Testing / Production 仍然必不可少

要 A/B 测试不同 prompt、不同上下文策略。

仍然需要监控延迟、成本、错误率——前端这边要有完善的埋点和日志。


## 工具准备

1 nodejs 下载

2 vscode


## 网站资源

参见保存标签。
