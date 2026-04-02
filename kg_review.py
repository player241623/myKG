"""
抽取审核模块
提供 Embedding 降维聚类 + 离群点检测，用于可视化审核
数据先暂存，用户审核确认后再批量写入 Neo4j
"""
import numpy as np
from llm_service import LLMService
from kg_extractor import extract_triples_from_text, split_text_into_chunks
from kg_fusion import find_similar_entities_lsh
from config import ONTOLOGY, SIMILARITY_THRESHOLD

llm = LLMService()

# 内存暂存区：session_id -> { triples, entities, embeddings, clusters }
_review_sessions = {}


def _collect_unique_entities(triples: list) -> list:
    """从三元组中收集去重的实体列表"""
    seen = {}
    for t in triples:
        h = t.get("head", "")
        if h and h not in seen:
            seen[h] = {"name": h, "type": t.get("head_type", "Concept"), "desc": t.get("head_desc", "")}
        tail = t.get("tail", "")
        if tail and tail not in seen:
            seen[tail] = {"name": tail, "type": t.get("tail_type", "Concept"), "desc": t.get("tail_desc", "")}
    return list(seen.values())


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """手写 PCA 降到 2 维，不依赖 sklearn"""
    X = embeddings - embeddings.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 取最大的两个特征向量
    idx = np.argsort(eigenvalues)[::-1][:2]
    W = eigenvectors[:, idx]
    return X @ W


def _dbscan_simple(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    简易 DBSCAN 实现，不依赖 sklearn
    返回每个点的簇标签，-1 表示离群点
    """
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(idx):
        dists = np.linalg.norm(points - points[idx], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1  # 噪声
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1

    return labels


def _find_merge_candidates(entities: list, embeddings: list, threshold: float) -> list:
    """使用 LSH 加速找出高相似度的实体对，作为合并候选"""
    names = [e["name"] for e in entities]
    pairs = find_similar_entities_lsh(names, entities, embeddings, threshold)
    # 补充 idx 信息供前端使用
    name_to_idx = {e["name"]: i for i, e in enumerate(entities)}
    for p in pairs:
        p["idx_a"] = name_to_idx.get(p["entity1"], -1)
        p["idx_b"] = name_to_idx.get(p["entity2"], -1)
        p["entity_a"] = p.pop("entity1")
        p["entity_b"] = p.pop("entity2")
        p["type_a"] = p.pop("type1", "")
        p["type_b"] = p.pop("type2", "")
    return pairs


# ============ 对外接口 ============

def extract_for_review(title: str, content: str) -> dict:
    """
    仅抽取，不入库。返回三元组 + 实体 embedding 散点 + 聚类 + 离群点
    """
    import uuid
    session_id = str(uuid.uuid4())[:8]

    # 1. LLM 抽取三元组
    chunks = split_text_into_chunks(content)
    all_triples = []
    for chunk in chunks:
        triples = extract_triples_from_text(chunk)
        for t in triples:
            t["source_chapter"] = title
        all_triples.extend(triples)

    if not all_triples:
        return {"session_id": session_id, "triples": [], "scatter": [], "clusters": [],
                "outliers": [], "merge_candidates": []}

    # 给每个三元组一个 id
    for i, t in enumerate(all_triples):
        t["_id"] = i

    # 2. 收集去重实体，获取 embedding
    entities = _collect_unique_entities(all_triples)
    names = [e["name"] for e in entities]

    batch_size = 50
    raw_embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        raw_embeddings.extend(llm.get_embeddings_batch(batch))

    emb_array = np.array(raw_embeddings)

    # 3. PCA 降维到 2D
    if len(entities) >= 2:
        coords_2d = _pca_2d(emb_array)
    else:
        coords_2d = np.zeros((len(entities), 2))

    # 4. DBSCAN 聚类 + 离群点检测
    if len(entities) >= 3:
        # 用 PCA 坐标做聚类（已降维，效率高）
        # 自适应 eps：取所有点对距离的中位数 * 0.5
        from itertools import combinations
        dists = [np.linalg.norm(coords_2d[i] - coords_2d[j])
                 for i, j in combinations(range(len(coords_2d)), 2)]
        eps = float(np.median(dists) * 0.5) if dists else 1.0
        labels = _dbscan_simple(coords_2d, eps=eps, min_samples=2)
    else:
        labels = np.zeros(len(entities), dtype=int)

    # 5. 找合并候选
    merge_candidates = _find_merge_candidates(entities, raw_embeddings, SIMILARITY_THRESHOLD)

    # 6. 构建散点数据
    scatter_data = []
    outlier_indices = []
    for i, ent in enumerate(entities):
        is_outlier = int(labels[i]) == -1
        scatter_data.append({
            "name": ent["name"],
            "type": ent["type"],
            "desc": ent["desc"],
            "x": round(float(coords_2d[i][0]), 4),
            "y": round(float(coords_2d[i][1]), 4),
            "cluster": int(labels[i]),
            "is_outlier": is_outlier,
            "entity_index": i
        })
        if is_outlier:
            outlier_indices.append(i)

    # 7. 暂存到内存
    _review_sessions[session_id] = {
        "triples": all_triples,
        "entities": entities,
        "embeddings": raw_embeddings,
        "labels": labels.tolist(),
        "scatter": scatter_data
    }

    return {
        "session_id": session_id,
        "triples": all_triples,
        "scatter": scatter_data,
        "outlier_names": [entities[i]["name"] for i in outlier_indices],
        "merge_candidates": merge_candidates,
        "total_entities": len(entities),
        "total_triples": len(all_triples),
        "num_clusters": int(labels.max() + 1) if len(labels) > 0 and labels.max() >= 0 else 0,
        "num_outliers": len(outlier_indices)
    }


def remove_noise(session_id: str, entity_names: list) -> dict:
    """从暂存区删除指定实体及其关联三元组"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}

    remove_set = set(entity_names)
    before = len(session["triples"])
    session["triples"] = [
        t for t in session["triples"]
        if t.get("head") not in remove_set and t.get("tail") not in remove_set
    ]
    session["entities"] = [e for e in session["entities"] if e["name"] not in remove_set]
    session["scatter"] = [s for s in session["scatter"] if s["name"] not in remove_set]

    removed = before - len(session["triples"])
    return {"removed_triples": removed, "remaining_triples": len(session["triples"])}


def merge_entities_in_session(session_id: str, keep: str, remove: str) -> dict:
    """在暂存区合并两个实体：将 remove 的名字全部替换为 keep"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}

    count = 0
    for t in session["triples"]:
        if t.get("head") == remove:
            t["head"] = keep
            count += 1
        if t.get("tail") == remove:
            t["tail"] = keep
            count += 1

    # 更新实体列表
    session["entities"] = [e for e in session["entities"] if e["name"] != remove]
    session["scatter"] = [s for s in session["scatter"] if s["name"] != remove]

    return {"merged": True, "affected_triples": count}


def delete_triples_in_session(session_id: str, triple_ids: list) -> dict:
    """在暂存区删除指定三元组"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}

    id_set = set(triple_ids)
    before = len(session["triples"])
    session["triples"] = [t for t in session["triples"] if t.get("_id") not in id_set]
    return {"removed": before - len(session["triples"]), "remaining": len(session["triples"])}


def commit_to_neo4j(session_id: str, neo4j) -> dict:
    """用户确认后，将暂存区数据批量写入 Neo4j"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}

    triples = session["triples"]
    for t in triples:
        neo4j.create_entity(
            name=t.get("head", ""),
            entity_type=t.get("head_type", "Concept"),
            description=t.get("head_desc", "")
        )
        neo4j.create_entity(
            name=t.get("tail", ""),
            entity_type=t.get("tail_type", "Concept"),
            description=t.get("tail_desc", "")
        )
        relation = t.get("relation", "RELATED_TO")
        if relation in ONTOLOGY["relation_types"]:
            neo4j.create_relation(head=t["head"], relation=relation, tail=t["tail"])

    # 清理 session
    del _review_sessions[session_id]
    return {"committed_triples": len(triples)}


def get_session(session_id: str) -> dict:
    """获取暂存区当前状态"""
    session = _review_sessions.get(session_id)
    if not session:
        return None
    return {
        "triples": session["triples"],
        "scatter": session["scatter"],
        "total_entities": len(session["entities"]),
        "total_triples": len(session["triples"])
    }
