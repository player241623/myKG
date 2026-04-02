"""
知识融合模块
实现开题报告中的「向量空间初筛 + 大模型逻辑终审」混合架构
解决实体消歧和共指消解问题

优化：使用 LSH（局部敏感哈希）代替 O(n²) 全量两两比较
复杂度从 O(n²) 降到接近 O(n)，大规模实体对齐不再卡顿
"""
import json
import numpy as np
from llm_service import LLMService
from neo4j_client import Neo4jClient
from config import SIMILARITY_THRESHOLD

llm = LLMService()


# ============ LSH 近似最近邻 ============

class SimHashLSH:
    """
    基于随机超平面的 LSH（SimHash）
    原理：用 k 个随机超平面把高维空间切成 2^k 个桶，
    余弦相似度高的向量大概率落入同一个桶。
    只比较同桶内的向量对，跳过绝大多数不相似的对。

    num_tables: 哈希表数量（越多召回越高，但慢）
    num_bits:   每个表的哈希位数（越大桶越细，候选越精准）
    """

    def __init__(self, dim: int, num_tables: int = 8, num_bits: int = 12):
        self.num_tables = num_tables
        self.num_bits = num_bits
        # 每个表一组随机超平面
        self.planes = [np.random.randn(num_bits, dim).astype(np.float32)
                       for _ in range(num_tables)]
        self.tables = [dict() for _ in range(num_tables)]  # hash_key -> [indices]

    def _hash(self, vec: np.ndarray, table_idx: int) -> str:
        projections = self.planes[table_idx] @ vec
        bits = (projections > 0).astype(int)
        return bits.tobytes()  # 用 bytes 做 key 比字符串快

    def index(self, vectors: np.ndarray):
        """批量建索引"""
        for i, vec in enumerate(vectors):
            for t in range(self.num_tables):
                key = self._hash(vec, t)
                self.tables[t].setdefault(key, []).append(i)

    def query_candidates(self) -> set:
        """返回所有至少在一个桶中共现的 (i, j) 对（i < j）"""
        candidates = set()
        for t in range(self.num_tables):
            for bucket in self.tables[t].values():
                if len(bucket) < 2:
                    continue
                for a in range(len(bucket)):
                    for b in range(a + 1, len(bucket)):
                        i, j = bucket[a], bucket[b]
                        candidates.add((min(i, j), max(i, j)))
        return candidates


def find_similar_entities_lsh(names: list, entities: list, embeddings: list,
                              threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    LSH 加速的实体对齐
    1. 建 LSH 索引，O(n) 找候选对
    2. 只对候选对精确计算余弦相似度
    3. 过阈值的才保留
    """
    n = len(names)
    if n < 2:
        return []

    emb = np.array(embeddings, dtype=np.float32)
    # 归一化（归一化后内积 = 余弦相似度）
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_normed = emb / norms

    # 建 LSH 索引
    dim = emb.shape[1]
    # 自适应参数：实体少时多表高召回，实体多时少位快速
    num_tables = min(12, max(4, n // 20))
    num_bits = min(16, max(8, int(np.log2(n + 1)) + 4))
    lsh = SimHashLSH(dim, num_tables=num_tables, num_bits=num_bits)
    lsh.index(emb_normed)

    # 获取候选对
    candidates = lsh.query_candidates()
    print(f"[LSH] {n} 个实体, 候选对 {len(candidates)} (全量需 {n*(n-1)//2})")

    # 精确验证
    similar_pairs = []
    for i, j in candidates:
        sim = float(emb_normed[i] @ emb_normed[j])
        if sim >= threshold:
            similar_pairs.append({
                "entity1": names[i],
                "entity2": names[j],
                "type1": entities[i].get("type", ""),
                "type2": entities[j].get("type", ""),
                "similarity": round(sim, 4)
            })

    print(f"[LSH] 过阈值的相似对: {len(similar_pairs)}")
    return similar_pairs


def find_similar_entities(neo4j: Neo4jClient, threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    通过 LSH 加速找出可能的同义实体对
    第一阶段：向量空间初筛
    """
    entities = neo4j.get_all_entities()
    if len(entities) < 2:
        return []

    names = [e["name"] for e in entities]
    print(f"[知识融合] 正在计算 {len(names)} 个实体的语义向量...")

    # 分批获取 embedding
    batch_size = 50
    embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_emb = llm.get_embeddings_batch(batch)
        embeddings.extend(batch_emb)

    return find_similar_entities_lsh(names, entities, embeddings, threshold)


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
    """执行实体合并"""
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
    完整的知识融合流程：LSH向量初筛 + LLM终审 + 执行合并
    """
    print("=" * 50)
    print("[知识融合] 开始执行（LSH 加速模式）...")

    # 第一阶段：LSH 向量初筛
    similar_pairs = find_similar_entities(neo4j)

    if not similar_pairs:
        print("[知识融合] 未发现需要融合的实体")
        return {"similar_pairs": 0, "merged": 0}

    # 第二阶段：LLM 终审
    merge_decisions = llm_judge_merge(similar_pairs)
    print(f"[知识融合] LLM 确认需要合并的实体对: {len(merge_decisions)}")

    # 第三阶段：执行合并
    merged = execute_fusion(neo4j, merge_decisions)

    return {
        "similar_pairs": len(similar_pairs),
        "llm_approved": len(merge_decisions),
        "merged": merged
    }
