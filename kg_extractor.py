"""
知识抽取模块
使用 LLM 从 Java 教学文本中抽取实体和关系三元组
实现了开题报告中的「LLM驱动的自动化知识抽取」
"""
import json
import time
from llm_service import LLMService
from neo4j_client import Neo4jClient
from config import ONTOLOGY, BATCH_SIZE

llm = LLMService()


EXTRACT_SYSTEM_PROMPT = """你是一个专业的 Java 编程教育专家，精通知识图谱构建。
你的任务是从给定的 Java 教学文本中提取知识图谱的实体和关系三元组。

## 严格约束
1. 实体类型必须是以下之一：{entity_types}
2. 关系类型必须是以下之一：{relation_types}

## 实体类型说明
- Syntax: 语法元素（变量声明、循环语句、条件判断、方法定义等）
- Concept: 抽象概念（面向对象、多态、封装、JVM内存模型等）
- Interface: Java 接口（Comparable、Serializable、Runnable等）
- Class: Java 类（String、ArrayList、Thread等）
- Keyword: Java 关键字（volatile、synchronized、static、final等）
- Error: 异常/错误类型（NullPointerException、ClassCastException等）
- Pattern: 设计模式（单例模式、工厂模式、观察者模式等）
- Library: 标准库/框架（java.util、java.io、Collections等）
- DataStructure: 数据结构（数组、链表、队列、栈、HashMap等）
- Algorithm: 算法相关（排序、搜索、递归等）

## 关系类型说明
- DEPENDS_ON: A 依赖于 B（使用B才能运行）
- INCLUDES: A 包含 B
- IMPLEMENTS: A 实现了 B（类实现接口）
- EXTENDS: A 继承/扩展自 B
- EXPLAINS: A 解释/阐述了 B 的原理
- THROWN_BY: 异常A由B抛出
- USES: A 使用了 B
- BELONGS_TO: A 属于 B（包/模块归属）
- PREREQUISITE: 学习A之前需要先学习B
- SIMILAR_TO: A 与 B 相似/可比较
- SOLVES: A 解决了 B 问题
- PART_OF: A 是 B 的组成部分

## 输出格式
请以 JSON 数组输出，每个元素包含：
- head: 头实体名称
- head_type: 头实体类型
- head_desc: 头实体简要描述
- relation: 关系类型
- tail: 尾实体名称
- tail_type: 尾实体类型
- tail_desc: 尾实体简要描述
""".format(
    entity_types=", ".join(ONTOLOGY["entity_types"]),
    relation_types=", ".join(ONTOLOGY["relation_types"])
)

REFLECT_SYSTEM_PROMPT = """你是一个知识图谱质量审计专家，需要审核以下 Java 知识三元组的质量。

## 审核标准
1. 实体名称是否准确、无歧义（如：应该是"volatile关键字"还是简单的"volatile"）
2. 关系类型是否正确反映了两个实体之间的真实关系
3. 是否有遗漏的重要三元组
4. 实体类型分类是否合理

## 要求
- 如果全部正确，回复 "PASS"
- 如果有错误或需要补充，输出修正后的完整 JSON 数组（与输入格式相同）
- 不要删除正确的三元组，只修正错误的或补充遗漏的
"""


def extract_triples_from_text(text: str) -> list:
    """
    从单个文本块中抽取三元组
    实现了开题报告中的「提取-反思」两阶段流程（Agentic Workflow）
    """
    # 阶段1: 初始抽取
    extract_prompt = f"请从以下 Java 教学文本中提取知识图谱三元组：\n\n{text}"
    initial_result = llm.chat(EXTRACT_SYSTEM_PROMPT, extract_prompt)
    initial_triples = llm.extract_json(initial_result)

    if not initial_triples:
        print(f"[WARNING] 初始抽取失败，跳过此文本块")
        return []

    # 阶段2: 反思与修正
    reflect_prompt = f"""请审核以下从 Java 教学文本中提取的知识三元组：

原始文本：
{text}

当前三元组：
{json.dumps(initial_triples, ensure_ascii=False, indent=2)}

请按审核标准进行检查。"""

    reflection = llm.chat(REFLECT_SYSTEM_PROMPT, reflect_prompt)

    if "PASS" in reflection.upper():
        return initial_triples
    else:
        refined = llm.extract_json(reflection)
        return refined if refined else initial_triples


def split_text_into_chunks(text: str, max_length: int = 2000) -> list:
    """将长文本分割成适合 LLM 处理的小块"""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def build_kg_from_texts(texts: list, neo4j: Neo4jClient, progress_callback=None):
    """
    从多个文本构建知识图谱
    texts: [{"title": "章节标题", "content": "文本内容"}, ...]
    """
    all_triples = []
    total = len(texts)

    for idx, item in enumerate(texts):
        title = item.get("title", f"chunk_{idx}")
        content = item.get("content", "")

        if not content.strip():
            continue

        # 分割大文本
        chunks = split_text_into_chunks(content)

        for chunk in chunks:
            try:
                triples = extract_triples_from_text(chunk)
                for triple in triples:
                    triple["source_chapter"] = title
                all_triples.extend(triples)

                # 入库
                _store_triples(neo4j, triples)

            except Exception as e:
                print(f"[ERROR] 处理 '{title}' 时出错: {e}")

            # 避免 API 限频
            time.sleep(0.5)

        if progress_callback:
            progress_callback(idx + 1, total, title)

        print(f"[{idx+1}/{total}] 已处理: {title}, 累计三元组: {len(all_triples)}")

    return all_triples


def _store_triples(neo4j: Neo4jClient, triples: list):
    """将三元组存入 Neo4j"""
    for t in triples:
        # 创建头实体
        neo4j.create_entity(
            name=t.get("head", ""),
            entity_type=t.get("head_type", "Concept"),
            description=t.get("head_desc", "")
        )
        # 创建尾实体
        neo4j.create_entity(
            name=t.get("tail", ""),
            entity_type=t.get("tail_type", "Concept"),
            description=t.get("tail_desc", "")
        )
        # 创建关系
        relation = t.get("relation", "RELATED_TO")
        if relation in ONTOLOGY["relation_types"]:
            neo4j.create_relation(
                head=t["head"],
                relation=relation,
                tail=t["tail"]
            )
