"""
抽取审核模块（v2）

核心改动：
1. 向量持久化：入库成功的实体，其 embedding 存到本地 vector_store.json
2. 增量相似度：新实体同时与「同批」和「已入库向量库」做 LSH 比较
3. 展示全部三元组，疑似重复的标记 duplicate_of 字段，由用户决定是否入库
"""
import os
import json
import uuid
import numpy as np
from llm_service import LLMService
from kg_extractor import extract_triples_from_text, split_text_into_chunks
from kg_fusion import SimHashLSH
from config import ONTOLOGY, SIMILARITY_THRESHOLD, DATA_DIR

llm = LLMService()

VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.json")

# 内存暂存区
_review_sessions = {}


# ============ 向量持久化 ============

def _load_vector_store() -> dict:
    """加载已持久化的向量库 {name: {type, embedding}}"""
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_vector_store(store: dict):
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


def _persist_vectors(entities: list, embeddings: list):
    """将新入库的实体向量追加到持久化存储"""
    store = _load_vector_store()
    for ent, emb in zip(entities, embeddings):
        store[ent["name"]] = {"type": ent["type"], "embedding": emb}
    _save_vector_store(store)
    print(f"[向量库] 持久化 {len(entities)} 个向量，总计 {len(store)}")


# ============ 工具函数 ============

def _collect_unique_entities(triples: list) -> list:
    seen = {}
    for t in triples:
        for role, prefix in [("head", "head"), ("tail", "tail")]:
            name = t.get(role, "")
            if name and name not in seen:
                seen[name] = {
                    "name": name,
                    "type": t.get(f"{prefix}_type", "Concept"),
                    "desc": t.get(f"{prefix}_desc", "")
                }
    return list(seen.values())


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    X = embeddings - embeddings.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:2]
    return X @ eigenvectors[:, idx]


def _dbscan_simple(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = list(np.where(dists <= eps)[0])
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            q = neighbors[j]
            if not visited[q]:
                visited[q] = True
                q_dists = np.linalg.norm(points - points[q], axis=1)
                q_nb = list(np.where(q_dists <= eps)[0])
                if len(q_nb) >= min_samples:
                    neighbors.extend(q_nb)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1
    return labels


def _find_duplicates_against_store(new_names: list, new_embeddings: list,
                                    threshold: float) -> dict:
    """
    新实体与已入库向量库做增量比较。
    返回 {新实体名: {match: 已有实体名, similarity: float}} 的字典
    使用 LSH 加速，不做全量比较。
    """
    store = _load_vector_store()
    if not store:
        return {}

    stored_names = list(store.keys())
    stored_embs = [store[n]["embedding"] for n in stored_names]

    # 将新旧向量拼在一起建 LSH 索引
    all_embs = np.array(stored_embs + new_embeddings, dtype=np.float32)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
    all_normed = all_embs / norms

    n_stored = len(stored_names)
    n_new = len(new_names)
    n_total = n_stored + n_new

    dim = all_normed.shape[1]
    lsh = SimHashLSH(dim, num_tables=min(12, max(4, n_total // 20)),
                     num_bits=min(16, max(8, int(np.log2(n_total + 1)) + 4)))
    lsh.index(all_normed)
    candidates = lsh.query_candidates()

    duplicates = {}
    for i, j in candidates:
        # 只关心跨新旧的对 (i 在 stored，j 在 new) 或反过来
        if i < n_stored and j >= n_stored:
            stored_idx, new_idx = i, j - n_stored
        elif j < n_stored and i >= n_stored:
            stored_idx, new_idx = j, i - n_stored
        else:
            continue  # 都是旧的或都是新的，跳过

        sim = float(all_normed[i] @ all_normed[j])
        if sim >= threshold:
            new_name = new_names[new_idx]
            if new_name not in duplicates or sim > duplicates[new_name]["similarity"]:
                duplicates[new_name] = {
                    "match": stored_names[stored_idx],
                    "similarity": round(sim, 4)
                }

    return duplicates


def _find_duplicates_within_batch(names: list, embeddings: list,
                                   threshold: float) -> list:
    """同一批内部的高相似度对"""
    n = len(names)
    if n < 2:
        return []
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_normed = emb / norms

    dim = emb.shape[1]
    lsh = SimHashLSH(dim, num_tables=min(12, max(4, n // 20)),
                     num_bits=min(16, max(8, int(np.log2(n + 1)) + 4)))
    lsh.index(emb_normed)
    candidates = lsh.query_candidates()

    pairs = []
    for i, j in candidates:
        sim = float(emb_normed[i] @ emb_normed[j])
        if sim >= threshold:
            pairs.append({
                "entity_a": names[i], "entity_b": names[j],
                "type_a": "", "type_b": "",
                "similarity": round(sim, 4)
            })
    return pairs


# ============ 对外接口 ============

def extract_for_review(title: str, content: str) -> dict:
    """
    抽取三元组，标记与已有库和同批的重复实体，不入库。
    """
    session_id = str(uuid.uuid4())[:8]

    # 1. LLM 抽取
    chunks = split_text_into_chunks(content)
    all_triples = []
    for chunk in chunks:
        triples = extract_triples_from_text(chunk)
        for t in triples:
            t["source_chapter"] = title
        all_triples.extend(triples)

    if not all_triples:
        return {"session_id": session_id, "triples": [], "scatter": [],
                "merge_candidates": [], "total_entities": 0, "total_triples": 0}

    for i, t in enumerate(all_triples):
        t["_id"] = i

    # 2. 收集实体 + 获取 embedding
    entities = _collect_unique_entities(all_triples)
    names = [e["name"] for e in entities]
    raw_embeddings = []
    for i in range(0, len(names), 50):
        raw_embeddings.extend(llm.get_embeddings_batch(names[i:i+50]))

    emb_array = np.array(raw_embeddings)

    # 3. PCA 降维 + DBSCAN 聚类
    if len(entities) >= 2:
        coords_2d = _pca_2d(emb_array)
    else:
        coords_2d = np.zeros((len(entities), 2))

    if len(entities) >= 3:
        from itertools import combinations
        dists = [np.linalg.norm(coords_2d[a] - coords_2d[b])
                 for a, b in combinations(range(len(coords_2d)), 2)]
        eps = float(np.median(dists) * 0.5) if dists else 1.0
        labels = _dbscan_simple(coords_2d, eps=eps, min_samples=2)
    else:
        labels = np.zeros(len(entities), dtype=int)

    # 4. 与已入库向量库做增量比较
    dup_vs_store = _find_duplicates_against_store(names, raw_embeddings, SIMILARITY_THRESHOLD)
    print(f"[审核] 与已有库重复实体: {len(dup_vs_store)}")

    # 5. 同批内部相似度
    batch_pairs = _find_duplicates_within_batch(names, raw_embeddings, SIMILARITY_THRESHOLD)
    # 补充 type 信息
    name2type = {e["name"]: e["type"] for e in entities}
    for p in batch_pairs:
        p["type_a"] = name2type.get(p["entity_a"], "")
        p["type_b"] = name2type.get(p["entity_b"], "")

    # 6. 标记三元组的重复状态
    # duplicate_flag: "store_dup" / "batch_dup" / null
    for t in all_triples:
        t["_dup_head"] = dup_vs_store.get(t.get("head"))  # {match, similarity} or None
        t["_dup_tail"] = dup_vs_store.get(t.get("tail"))
        t["_checked"] = True  # 默认勾选（用户可取消）

    # 如果实体在已有库中存在完全同名，也标记
    store = _load_vector_store()
    for t in all_triples:
        if t["head"] in store and t["_dup_head"] is None:
            t["_dup_head"] = {"match": t["head"], "similarity": 1.0}
        if t["tail"] in store and t["_dup_tail"] is None:
            t["_dup_tail"] = {"match": t["tail"], "similarity": 1.0}

    # 7. 散点数据
    outlier_indices = []
    scatter_data = []
    for i, ent in enumerate(entities):
        is_outlier = int(labels[i]) == -1
        is_dup = ent["name"] in dup_vs_store
        scatter_data.append({
            "name": ent["name"], "type": ent["type"], "desc": ent["desc"],
            "x": round(float(coords_2d[i][0]), 4),
            "y": round(float(coords_2d[i][1]), 4),
            "cluster": int(labels[i]),
            "is_outlier": is_outlier,
            "is_store_dup": is_dup,
            "dup_match": dup_vs_store.get(ent["name"], {}).get("match", ""),
        })
        if is_outlier:
            outlier_indices.append(i)

    # 8. 暂存
    _review_sessions[session_id] = {
        "triples": all_triples,
        "entities": entities,
        "embeddings": [e.tolist() if isinstance(e, np.ndarray) else e for e in raw_embeddings],
        "scatter": scatter_data,
    }

    return {
        "session_id": session_id,
        "triples": all_triples,
        "scatter": scatter_data,
        "outlier_names": [entities[i]["name"] for i in outlier_indices],
        "merge_candidates": batch_pairs,
        "store_duplicates": dup_vs_store,
        "total_entities": len(entities),
        "total_triples": len(all_triples),
        "num_clusters": int(labels.max() + 1) if len(labels) > 0 and labels.max() >= 0 else 0,
        "num_outliers": len(outlier_indices),
        "num_store_dups": len(dup_vs_store),
        "vector_store_size": len(store),
    }


def update_triple_checked(session_id: str, triple_id: int, checked: bool) -> dict:
    """用户切换某条三元组的勾选状态"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}
    for t in session["triples"]:
        if t["_id"] == triple_id:
            t["_checked"] = checked
            return {"ok": True}
    return {"error": "triple not found"}


def batch_set_checked(session_id: str, triple_ids: list, checked: bool) -> dict:
    """批量设置勾选状态"""
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}
    id_set = set(triple_ids)
    count = 0
    for t in session["triples"]:
        if t["_id"] in id_set:
            t["_checked"] = checked
            count += 1
    return {"updated": count}


def merge_entities_in_session(session_id: str, keep: str, remove: str) -> dict:
    """在暂存区合并两个实体"""
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
    session["entities"] = [e for e in session["entities"] if e["name"] != remove]
    session["scatter"] = [s for s in session["scatter"] if s["name"] != remove]
    return {"merged": True, "affected_triples": count}


def commit_to_neo4j(session_id: str, neo4j) -> dict:
    """
    只入库用户勾选 _checked=True 的三元组。
    入库后，将这些实体的向量持久化到 vector_store。
    """
    session = _review_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}

    checked_triples = [t for t in session["triples"] if t.get("_checked", True)]
    skipped = len(session["triples"]) - len(checked_triples)

    # 收集需要入库的实体名
    committed_entities = set()
    for t in checked_triples:
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
        committed_entities.add(t["head"])
        committed_entities.add(t["tail"])

    # 持久化向量
    name2idx = {e["name"]: i for i, e in enumerate(session["entities"])}
    to_persist_ents = []
    to_persist_embs = []
    for name in committed_entities:
        if name in name2idx:
            idx = name2idx[name]
            to_persist_ents.append(session["entities"][idx])
            to_persist_embs.append(session["embeddings"][idx])
    if to_persist_ents:
        _persist_vectors(to_persist_ents, to_persist_embs)

    del _review_sessions[session_id]
    return {
        "committed_triples": len(checked_triples),
        "skipped_triples": skipped,
        "persisted_vectors": len(to_persist_ents)
    }


def get_session(session_id: str) -> dict:
    session = _review_sessions.get(session_id)
    if not session:
        return None
    return {
        "triples": session["triples"],
        "scatter": session["scatter"],
        "total_entities": len(session["entities"]),
        "total_triples": len(session["triples"])
    }
