"""
知识融合模块
实现开题报告中的「向量空间初筛 + 大模型逻辑终审」混合架构
解决实体消歧和共指消解问题
"""
import json
import numpy as np
from llm_service import LLMService
from neo4j_client import Neo4jClient
from config import SIMILARITY_THRESHOLD

llm = LLMService()


def cosine_similarity(v1: list, v2: list) -> float:
    """计算余弦相似度"""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_similar_entities(neo4j: Neo4jClient, threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    通过向量相似度找出可能的同义实体对
    第一阶段：向量空间初筛
    """
    entities = neo4j.get_all_entities()
    if len(entities) < 2:
        return []

    # 获取所有实体名称的 embedding
    names = [e["name"] for e in entities]
    print(f"[知识融合] 正在计算 {len(names)} 个实体的语义向量...")

    # 分批获取 embedding
    batch_size = 50
    embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_emb = llm.get_embeddings_batch(batch)
        embeddings.extend(batch_emb)

    # 计算两两相似度，找出高相似度对
    similar_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                similar_pairs.append({
                    "entity1": names[i],
                    "entity2": names[j],
                    "type1": entities[i].get("type", ""),
                    "type2": entities[j].get("type", ""),
                    "similarity": round(sim, 4)
                })

    print(f"[知识融合] 发现 {len(similar_pairs)} 对潜在同义实体")
    return similar_pairs


def llm_judge_merge(pairs: list) -> list:
    """
    第二阶段：大模型逻辑终审
    对向量初筛出的实体对，使用 LLM 判断是否应该合并
    """
    if not pairs:
        return []

    system_prompt = """你是一个 Java 编程知识领域的实体消歧专家。
我会给你一些可能是同义词的实体对，请判断它们是否指代同一个概念。

## 判断标准
1. 完全相同概念的不同表达（如 "JVM" 和 "Java虚拟机"）-> 应该合并
2. 缩写与全称（如 "OOP" 和 "面向对象编程"）-> 应该合并
3. 相关但不同的概念（如 "ArrayList" 和 "LinkedList"）-> 不应合并
4. 同一概念的中英文表达 -> 应该合并

## 输出格式
对每一对实体，输出 JSON 数组，每个元素：
{
    "entity1": "实体1",
    "entity2": "实体2",
    "should_merge": true/false,
    "keep": "保留的实体名称（选更规范的那个）",
    "reason": "判断理由"
}"""

    # 分批处理
    results = []
    batch_size = 10
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        prompt = "请判断以下实体对是否应该合并：\n\n"
        for idx, p in enumerate(batch):
            prompt += f"{idx+1}. \"{p['entity1']}\"({p['type1']}) vs \"{p['entity2']}\"({p['type2']}), 相似度={p['similarity']}\n"

        response = llm.chat(system_prompt, prompt)
        parsed = llm.extract_json(response)
        if parsed:
            results.extend(parsed)

    return [r for r in results if r.get("should_merge", False)]


def execute_fusion(neo4j: Neo4jClient, merge_decisions: list):
    """
    执行实体合并
    """
    merged_count = 0
    for decision in merge_decisions:
        keep = decision.get("keep", decision.get("entity1"))
        remove = decision.get("entity2") if keep == decision.get("entity1") else decision.get("entity1")

        if not keep or not remove:
            continue

        try:
            neo4j.simple_merge_entities(keep_name=keep, remove_name=remove)
            merged_count += 1
            print(f"[融合] 合并 '{remove}' -> '{keep}' | 原因: {decision.get('reason', '')}")
        except Exception as e:
            print(f"[融合错误] 合并 '{remove}' -> '{keep}' 失败: {e}")

    print(f"[知识融合] 共合并 {merged_count} 个实体")
    return merged_count


def run_knowledge_fusion(neo4j: Neo4jClient) -> dict:
    """
    完整的知识融合流程：向量初筛 + LLM终审 + 执行合并
    """
    print("=" * 50)
    print("[知识融合] 开始执行...")

    # 第一阶段：向量初筛
    similar_pairs = find_similar_entities(neo4j)

    if not similar_pairs:
        print("[知识融合] 未发现需要融合的实体")
        return {"similar_pairs": 0, "merged": 0}

    # 第二阶段：LLM终审
    merge_decisions = llm_judge_merge(similar_pairs)
    print(f"[知识融合] LLM 确认需要合并的实体对: {len(merge_decisions)}")

    # 第三阶段：执行合并
    merged = execute_fusion(neo4j, merge_decisions)

    return {
        "similar_pairs": len(similar_pairs),
        "llm_approved": len(merge_decisions),
        "merged": merged
    }
