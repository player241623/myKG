"""
项目配置文件
基于知识图谱和大模型的Java课程问答系统
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ============ LLM 配置 ============
# 支持 OpenAI 兼容接口（如 DeepSeek、ChatGLM、Moonshot 等）
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Embedding 模型配置
# 默认使用硅基流动（SiliconFlow）的免费 bge-m3 模型，国内直连
# 注册即送额度：https://siliconflow.cn  免费模型无需额外充值
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("SILICONFLOW_API_KEY", LLM_API_KEY))
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024  # bge-m3 维度

# ============ Neo4j 配置 ============
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ============ 知识图谱 Ontology 定义 ============
# 根据开题报告，定义Java课程的实体类型和关系类型
ONTOLOGY = {
    "entity_types": [
        "Syntax",       # 语法实体：变量、函数、数组、循环、条件等
        "Concept",      # 概念实体：内存模型、函数调用栈、数据类型等
        "Interface",    # 接口：Java接口定义
        "Class",        # 类：Java类定义
        "Keyword",      # 关键字：volatile、synchronized、static等
        "Error",        # 异常/错误：NullPointerException等
        "Pattern",      # 设计模式：单例模式、工厂模式等
        "Library",      # 标准库/框架：Collections、Stream等
        "DataStructure",# 数据结构：ArrayList、HashMap等
        "Algorithm",    # 算法相关概念
    ],
    "relation_types": [
        "DEPENDS_ON",   # 依赖关系
        "INCLUDES",     # 包含关系
        "IMPLEMENTS",   # 实现关系
        "EXTENDS",      # 继承/扩展关系
        "EXPLAINS",     # 解释/阐述关系
        "THROWN_BY",    # 异常抛出关系
        "USES",         # 使用关系
        "BELONGS_TO",   # 归属关系
        "PREREQUISITE", # 前置知识
        "SIMILAR_TO",   # 相似关系
        "SOLVES",       # 解决问题
        "PART_OF",      # 组成部分
    ]
}

# ============ Flask 配置 ============
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ============ 知识融合配置 ============
SIMILARITY_THRESHOLD = 0.85  # 向量相似度阈值，高于此值认为可能是同义实体
BATCH_SIZE = 5  # LLM 批量处理的文本块数

# ============ 数据目录 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始爬取数据
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # 处理后数据

# 确保目录存在
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
