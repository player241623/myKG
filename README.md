# 基于知识图谱和大模型的 Java 课程问答系统

> 本科毕业设计 — 张非池 (2022302873)  
> 指导教师：陈建全

## 系统简介

本系统实现了一套完整的 **"知识图谱自动构建 + GraphRAG 智能问答"** 一体化方案，专为 Java 编程教学设计。

### 核心功能

1. **LLM 驱动的知识图谱自动构建**
   - 从 Runoob.com 自动爬取 Java 教程
   - 使用大模型（GPT-4o/DeepSeek等）自动抽取实体和关系三元组
   - 采用「提取-反思」两阶段 Agentic Workflow 确保质量
   - 支持手动导入教学文本

2. **知识融合（实体消歧 + 共指消解）**
   - 向量空间初筛：使用 Embedding 模型计算实体语义相似度
   - 大模型逻辑终审：LLM 对高相似度实体对进行最终判别
   - 自动合并同义实体（如 "JVM" 与 "Java虚拟机"）

3. **GraphRAG 智能问答**
   - 查询解析：LLM 识别问题中的核心实体和查询意图
   - 子图检索：从 Neo4j 执行多跳关联检索，获取完整知识上下文
   - 回复生成：基于结构化子图事实生成带知识路径的回答

4. **知识图谱可视化**
   - 基于 ECharts 的交互式力导向图
   - 支持节点搜索、类型筛选、路径展开
   - 问答时自动高亮相关知识节点

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Python + Flask |
| 图数据库 | Neo4j |
| LLM | OpenAI 兼容接口（GPT-4o / DeepSeek / Moonshot） |
| 前端 | Vue 3 + Element Plus + ECharts |
| 数据采集 | BeautifulSoup + Requests |

## 环境搭建（Windows）

### 1. 安装 Java 17（Neo4j 运行依赖）

通过 winget 一键安装：

```powershell
winget install EclipseAdoptium.Temurin.17.JDK --accept-source-agreements --accept-package-agreements --silent
```

安装完成后验证：

```powershell
java -version
# 输出: openjdk version "17.0.18" ...
```

### 2. 安装 Neo4j Community Edition

下载 Neo4j Community 5.26.0 ZIP 包并解压到项目目录：

```powershell
# 下载
Invoke-WebRequest -Uri "https://neo4j.com/artifact.php?name=neo4j-community-5.26.0-windows.zip" -OutFile neo4j.zip

# 解压到项目目录
Expand-Archive -Path neo4j.zip -DestinationPath . -Force

# 清理 zip
Remove-Item neo4j.zip
```

解压后会得到 `neo4j-community-5.26.0/` 目录。

**关闭认证（开发环境方便调试）**：编辑 `neo4j-community-5.26.0/conf/neo4j.conf`，取消注释以下行：

```
dbms.security.auth_enabled=false
```

### 3. 安装 Python 依赖

```powershell
python -m pip install -r requirements.txt
```

### 4. 配置 LLM API Key

```powershell
# 复制示例配置
copy .env.example .env
```

编辑 `.env` 文件，填入你的 LLM API Key。支持 OpenAI / DeepSeek / Moonshot 等兼容接口：

```env
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

如果使用 DeepSeek：

```env
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## 启动系统

### 方式一：一键启动（推荐）

双击 `start.bat` 即可自动启动 Neo4j 和 Flask。

停止服务：双击 `stop.bat`。

### 方式二：手动启动

**第一步：启动 Neo4j**

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
.\neo4j-community-5.26.0\bin\neo4j.bat console
```

等待输出 `Started.` 表示启动成功。Neo4j 控制台：http://localhost:7474

**第二步：启动 Flask 后端**（新开一个终端）

```powershell
cd myKG
python app.py
```

输出 `Running on http://127.0.0.1:5000` 表示启动成功。

**第三步：打开浏览器**

访问 http://localhost:5000

### 验证服务状态

```powershell
# 验证 Neo4j
python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687'); d.verify_connectivity(); print('Neo4j OK'); d.close()"

# 验证 Flask API
python -c "import requests; r=requests.get('http://localhost:5000/api/graph/stats'); print(r.json())"
```

## 使用流程

### Step 1: 构建知识图谱

1. 打开 **"图谱构建"** 标签页
2. 选择以下方式之一：
   - **自动爬取**：设置页面数，点击"开始爬取并构建"
   - **手动输入**：粘贴 Java 教学文本，点击"提取并入库"
3. 等待系统自动抽取实体和关系

### Step 2: 知识融合（可选）

- 在 "图谱构建" 页面点击 **"执行知识融合"**
- 系统将自动发现并合并同义实体

### Step 3: 智能问答

1. 切换到 **"智能问答"** 标签页
2. 输入 Java 相关问题
3. 系统会从知识图谱中检索相关子图，结合 LLM 生成带知识路径的回答

## 系统架构

```
用户提问
   ↓
[查询解析] LLM 识别实体 + 意图
   ↓
[子图检索] Neo4j 多跳关联检索
   ↓
[回复生成] LLM + 结构化子图 → 带知识路径的回答
   ↓
[前端展示] 文字回答 + 知识路径可视化
```

## 项目结构

```
myKG/
├── app.py              # Flask 主应用（API路由）
├── config.py           # 统一配置
├── neo4j_client.py     # Neo4j 图数据库操作
├── llm_service.py      # LLM 服务封装
├── kg_extractor.py     # 知识抽取模块（提取-反思流程）
├── kg_fusion.py        # 知识融合模块（向量初筛+LLM终审）
├── graphrag_qa.py      # GraphRAG 智能问答模块
├── data_crawler.py     # Java 教程数据爬取
├── extract.py          # 早期原型（保留）
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
├── start.bat           # 一键启动脚本（Windows）
├── stop.bat            # 一键停止脚本（Windows）
├── neo4j-community-5.26.0/  # Neo4j 图数据库
├── frontend/
│   └── index.html      # 前端界面（Vue3 + ECharts）
└── data/
    ├── raw/            # 爬取的原始数据
    └── processed/      # 处理后的数据
```

## Ontology 定义（知识图谱底层结构）

### 实体类型
| 类型 | 说明 | 示例 |
|------|------|------|
| Syntax | 语法元素 | 变量声明、for循环、方法定义 |
| Concept | 抽象概念 | 面向对象、多态、JMM |
| Interface | Java 接口 | Comparable、Runnable |
| Class | Java 类 | String、Thread |
| Keyword | 关键字 | volatile、synchronized |
| Error | 异常/错误 | NullPointerException |
| Pattern | 设计模式 | 单例模式、工厂模式 |
| Library | 标准库 | java.util、Collections |
| DataStructure | 数据结构 | ArrayList、HashMap |
| Algorithm | 算法 | 排序、二分搜索 |

### 关系类型
| 关系 | 说明 |
|------|------|
| DEPENDS_ON | 依赖关系 |
| INCLUDES | 包含关系 |
| IMPLEMENTS | 实现关系 |
| EXTENDS | 继承关系 |
| EXPLAINS | 解释关系 |
| THROWN_BY | 异常抛出 |
| USES | 使用关系 |
| BELONGS_TO | 归属关系 |
| PREREQUISITE | 前置知识 |
| SIMILAR_TO | 相似关系 |
| SOLVES | 解决问题 |
| PART_OF | 组成部分 |

## 参考文献

- [1] Bratanic Tomaz. Constructing Knowledge Graphs from Text Using OpenAI Functions
- [2] 刘峤等. 知识图谱构建技术综述
- [3] EDGE D et al. From Local to Global: A GraphRAG Approach
- [5] 冷子霖. 基于知识图谱的C语言程序设计课程问答系统的设计与实现
- [9] Bratanic Tomaz. Enhancing the Accuracy of RAG Applications With Knowledge Graphs
