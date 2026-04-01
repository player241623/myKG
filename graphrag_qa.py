"""
GraphRAG 智能问答模块
实现开题报告中的三阶段问答：查询解析 → 子图检索 → 回复生成

核心创新点：
1. 通过 LLM 将自然语言问题解析为图谱查询意图
2. 基于 Neo4j 执行多跳关联检索，获取完整知识子图
3. 将结构化子图事实喂给 LLM 生成带知识路径的回答
"""
import json
from llm_service import LLMService
from neo4j_client import Neo4jClient
from config import ONTOLOGY

llm = LLMService()


# ============ 阶段1: 查询解析 ============

QUERY_PARSE_PROMPT = """你是一个 Java 编程教育问答系统的查询解析器。
请分析用户的自然语言问题，提取以下信息：

1. **核心实体**：问题中涉及的 Java 知识点实体名称（尽量使用规范术语）
2. **查询意图**：
   - "definition": 查询某个知识点的定义
   - "relation": 查询两个知识点之间的关系
   - "reasoning": 需要多跳推理解释原理
   - "comparison": 比较两个概念的异同
   - "usage": 查询使用方法/示例
3. **检索深度**：建议的图谱检索跳数（1-3跳）

## 输出 JSON 格式
{
    "entities": ["实体1", "实体2"],
    "intent": "reasoning",
    "depth": 2,
    "cypher_hint": "可选的 Cypher 查询提示"
}"""


def parse_query(question: str) -> dict:
    """解析用户问题，提取实体和意图"""
    response = llm.chat(QUERY_PARSE_PROMPT, f"用户问题：{question}")
    parsed = llm.extract_json(response)
    if not parsed:
        # 回退：直接把问题当关键词搜索
        return {
            "entities": [question],
            "intent": "definition",
            "depth": 2
        }
    return parsed


# ============ 阶段2: 子图检索 ============

def retrieve_subgraph(neo4j: Neo4jClient, query_info: dict) -> dict:
    """
    基于解析结果从知识图谱中检索子图
    实现多跳关联检索，获取前驱、后继节点形成完整上下文
    """
    entities = query_info.get("entities", [])
    intent = query_info.get("intent", "definition")
    depth = min(query_info.get("depth", 2), 3)  # 最多3跳

    context = {
        "matched_entities": [],
        "subgraph_triples": [],
        "paths": [],
        "entity_details": []
    }

    # 1. 搜索实体
    all_matched = []
    for entity_name in entities:
        results = neo4j.search_entity(entity_name, limit=5)
        if results:
            all_matched.extend(results)
            context["matched_entities"].extend(results)

    if not all_matched:
        # 全局搜索兜底
        for entity_name in entities:
            words = entity_name.split()
            for word in words:
                if len(word) > 1:
                    results = neo4j.search_entity(word, limit=3)
                    all_matched.extend(results)
                    context["matched_entities"].extend(results)

    # 2. 获取子图
    matched_names = list(set(e["name"] for e in all_matched))
    if matched_names:
        subgraph = neo4j.get_subgraph_for_query(matched_names, depth=depth)
        context["subgraph_triples"] = subgraph

    # 3. 获取实体详细信息及邻居
    for name in matched_names[:5]:  # 限制数量避免信息过载
        detail = neo4j.get_entity_with_neighbors(name, depth=depth)
        if detail:
            context["entity_details"].append(detail)

    # 4. 如果是relation类型，尝试获取路径
    if intent == "relation" and len(matched_names) >= 2:
        path = neo4j.get_path_between(matched_names[0], matched_names[1])
        if path:
            context["paths"].append(path)

    # 5. 如果是reasoning类型，需要更深度的检索
    if intent == "reasoning" and matched_names:
        for name in matched_names[:3]:
            detail = neo4j.get_entity_with_neighbors(name, depth=3)
            if detail and detail not in context["entity_details"]:
                context["entity_details"].append(detail)

    return context


# ============ 阶段3: 回复生成 ============

ANSWER_SYSTEM_PROMPT = """你是一个专业的 Java 编程教育助手。你的回答基于知识图谱中检索到的结构化事实。

## 回答要求
1. **知识定位**：首先明确回答涉及的核心知识点及其在 Java 知识体系中的位置
2. **逻辑推演**：按照知识图谱中的关系链条，逐步展开逻辑推演过程
3. **路径展示**：在回答末尾，以 "知识路径" 的形式展示相关知识点的逻辑链条
4. **准确性**：严格基于检索到的知识图谱事实回答，如果图谱中没有相关信息，诚实说明
5. **教育性**：使用清晰易懂的语言，适合编程学习者理解

## 回答格式
1. 先给出核心回答
2. 详细解释推演过程
3. 最后以如下格式展示知识路径：

**知识路径：** `实体A` → [关系] → `实体B` → [关系] → `实体C`
"""


def generate_answer(question: str, context: dict) -> dict:
    """
    基于检索到的子图上下文，使用 LLM 生成回答
    """
    # 构建子图描述
    subgraph_text = _format_context(context)

    if not subgraph_text.strip():
        # 无图谱信息时，使用 LLM 通用知识回答
        prompt = f"""用户问题：{question}

注意：知识图谱中未找到直接相关的信息。请基于你的 Java 编程知识回答，但请标注这是通用回答，建议用户导入更多教学资源来丰富知识图谱。"""
    else:
        prompt = f"""用户问题：{question}

以下是从 Java 知识图谱中检索到的结构化事实：

{subgraph_text}

请基于以上知识图谱事实，生成包含"知识点定位 + 逻辑推演过程"的回复。"""

    answer_text = llm.chat(ANSWER_SYSTEM_PROMPT, prompt, temperature=0.3)

    # 构建知识路径用于前端展示
    knowledge_paths = _extract_paths(context)

    return {
        "answer": answer_text,
        "knowledge_paths": knowledge_paths,
        "matched_entities": [e["name"] for e in context.get("matched_entities", [])],
        "subgraph_size": len(context.get("subgraph_triples", []))
    }


def _format_context(context: dict) -> str:
    """将子图上下文格式化为文本描述"""
    parts = []

    # 匹配到的实体
    if context.get("matched_entities"):
        parts.append("## 匹配的实体")
        for e in context["matched_entities"]:
            parts.append(f"- **{e['name']}** (类型: {e.get('type', '未知')}): {e.get('description', '')}")

    # 子图三元组
    if context.get("subgraph_triples"):
        parts.append("\n## 知识关系")
        seen = set()
        for t in context["subgraph_triples"]:
            key = f"{t['source']}-{t['relation']}-{t['target']}"
            if key not in seen:
                seen.add(key)
                parts.append(f"- {t['source']}({t.get('source_type','')}) --[{t['relation']}]--> {t['target']}({t.get('target_type','')})")

    # 实体详情
    if context.get("entity_details"):
        parts.append("\n## 实体详情及关联")
        for detail in context["entity_details"]:
            parts.append(f"\n### {detail.get('name', '')} ({detail.get('type', '')})")
            parts.append(f"描述: {detail.get('description', '无')}")

            outgoing = detail.get("outgoing", [])
            if outgoing:
                for out in outgoing:
                    if out.get("name"):
                        rels = out.get("relation", [])
                        rel_str = " -> ".join(rels) if rels else "RELATED"
                        parts.append(f"  → [{rel_str}] {out['name']} ({out.get('type', '')})")

            incoming = detail.get("incoming", [])
            if incoming:
                for inc in incoming:
                    if inc.get("name"):
                        rels = inc.get("relation", [])
                        rel_str = " -> ".join(rels) if rels else "RELATED"
                        parts.append(f"  ← [{rel_str}] {inc['name']} ({inc.get('type', '')})")

    # 路径
    if context.get("paths"):
        parts.append("\n## 知识路径")
        for path in context["paths"]:
            nodes = path.get("nodes", [])
            relations = path.get("relations", [])
            path_str = ""
            for i, node in enumerate(nodes):
                path_str += f"{node['name']}"
                if i < len(relations):
                    path_str += f" --[{relations[i]}]--> "
            parts.append(f"路径: {path_str}")

    return "\n".join(parts)


def _extract_paths(context: dict) -> list:
    """提取知识路径用于前端可视化"""
    paths = []

    # 从子图三元组中构建路径
    triples = context.get("subgraph_triples", [])
    for t in triples:
        paths.append({
            "source": t["source"],
            "target": t["target"],
            "relation": t["relation"],
            "source_type": t.get("source_type", ""),
            "target_type": t.get("target_type", "")
        })

    return paths


# ============ 主问答入口 ============

def answer_question(question: str, neo4j: Neo4jClient) -> dict:
    """
    完整的 GraphRAG 问答流程
    1. 查询解析
    2. 子图检索
    3. 回复生成
    """
    # 阶段1: 查询解析
    query_info = parse_query(question)
    print(f"[查询解析] 实体: {query_info.get('entities')}, 意图: {query_info.get('intent')}")

    # 阶段2: 子图检索
    context = retrieve_subgraph(neo4j, query_info)
    print(f"[子图检索] 匹配实体: {len(context['matched_entities'])}, "
          f"三元组: {len(context['subgraph_triples'])}")

    # 阶段3: 回复生成
    result = generate_answer(question, context)

    # 附加查询元信息
    result["query_info"] = query_info

    return result
