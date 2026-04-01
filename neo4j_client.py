"""
Neo4j 图数据库操作模块
负责知识图谱的存储、查询、更新
"""
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jClient:
    def __init__(self):
        # auth_enabled=false 时无需认证；如果开启了认证则使用配置的用户名密码
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI)
            self.driver.verify_connectivity()
        except Exception:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._init_constraints()

    def _init_constraints(self):
        """初始化唯一性约束和索引"""
        with self.driver.session() as session:
            # 为每个实体名称创建唯一性约束
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            # 创建全文索引用于搜索
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.name, e.description]"
                )
            except Exception:
                pass  # 索引已存在

    def close(self):
        self.driver.close()

    # ============ 实体操作 ============
    def create_entity(self, name: str, entity_type: str, description: str = "",
                      difficulty: str = "", chapter: str = "", properties: dict = None):
        """创建或更新实体节点"""
        props = {
            "name": name,
            "entity_type": entity_type,
            "description": description,
            "difficulty": difficulty,
            "chapter": chapter,
        }
        if properties:
            props.update(properties)

        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e += $props, e:""" + entity_type + """
                RETURN e.name AS name
                """,
                name=name, props=props
            )
            return result.single()

    def batch_create_entities(self, entities: list):
        """批量创建实体"""
        with self.driver.session() as session:
            for ent in entities:
                entity_type = ent.get("entity_type", "Concept")
                session.run(
                    f"""
                    MERGE (e:Entity {{name: $name}})
                    SET e += $props, e:{entity_type}
                    """,
                    name=ent["name"],
                    props=ent
                )

    # ============ 关系操作 ============
    def create_relation(self, head: str, relation: str, tail: str, properties: dict = None):
        """创建关系"""
        props = properties or {}
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (h:Entity {{name: $head}})
                MATCH (t:Entity {{name: $tail}})
                MERGE (h)-[r:{relation}]->(t)
                SET r += $props
                """,
                head=head, tail=tail, props=props
            )

    def batch_create_relations(self, triples: list):
        """批量创建关系三元组"""
        with self.driver.session() as session:
            for triple in triples:
                rel_type = triple.get("relation", "RELATED_TO")
                session.run(
                    f"""
                    MERGE (h:Entity {{name: $head}})
                    MERGE (t:Entity {{name: $tail}})
                    MERGE (h)-[r:{rel_type}]->(t)
                    SET r.description = $desc
                    """,
                    head=triple["head"],
                    tail=triple["tail"],
                    desc=triple.get("description", "")
                )

    # ============ 查询操作 ============
    def search_entity(self, keyword: str, limit: int = 20):
        """模糊搜索实体"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($keyword)
                   OR toLower(e.description) CONTAINS toLower($keyword)
                RETURN e.name AS name, e.entity_type AS type,
                       e.description AS description
                ORDER BY CASE WHEN toLower(e.name) = toLower($keyword) THEN 0 ELSE 1 END
                LIMIT $limit
                """,
                keyword=keyword, limit=limit
            )
            return [dict(r) for r in result]

    def get_entity_with_neighbors(self, name: str, depth: int = 1):
        """获取实体及其邻居节点（用于子图检索）"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (center:Entity {name: $name})
                OPTIONAL MATCH path = (center)-[r*1..""" + str(depth) + """]->(neighbor:Entity)
                WITH center, collect(DISTINCT {
                    name: neighbor.name,
                    type: neighbor.entity_type,
                    description: neighbor.description,
                    relation: [rel IN r | type(rel)],
                    path_names: [n IN nodes(path) | n.name]
                }) AS outgoing
                OPTIONAL MATCH path2 = (prev:Entity)-[r2*1..""" + str(depth) + """]->(center)
                WITH center, outgoing, collect(DISTINCT {
                    name: prev.name,
                    type: prev.entity_type,
                    description: prev.description,
                    relation: [rel IN r2 | type(rel)],
                    path_names: [n IN nodes(path2) | n.name]
                }) AS incoming
                RETURN center.name AS name, center.entity_type AS type,
                       center.description AS description,
                       outgoing, incoming
                """,
                name=name
            )
            record = result.single()
            if record:
                return dict(record)
            return None

    def get_path_between(self, entity1: str, entity2: str, max_depth: int = 5):
        """获取两个实体之间的路径"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (a:Entity {name: $e1})-[*1..""" + str(max_depth) + """]-(b:Entity {name: $e2})
                )
                RETURN [n IN nodes(path) | {name: n.name, type: n.entity_type}] AS nodes,
                       [r IN relationships(path) | type(r)] AS relations
                """,
                e1=entity1, e2=entity2
            )
            record = result.single()
            if record:
                return dict(record)
            return None

    def get_subgraph_for_query(self, entity_names: list, depth: int = 2):
        """获取多个实体的子图（用于GraphRAG检索）"""
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $names AS ename
                MATCH (e:Entity {name: ename})
                OPTIONAL MATCH path = (e)-[*1..""" + str(depth) + """]-(neighbor:Entity)
                WITH e, neighbor, relationships(path) AS rels
                UNWIND rels AS rel
                WITH DISTINCT startNode(rel) AS src, endNode(rel) AS tgt, type(rel) AS relType
                RETURN src.name AS source, tgt.name AS target,
                       src.entity_type AS source_type, tgt.entity_type AS target_type,
                       src.description AS source_desc, tgt.description AS target_desc,
                       relType AS relation
                """,
                names=entity_names
            )
            return [dict(r) for r in result]

    def get_all_entities(self, entity_type: str = None, limit: int = 500):
        """获取所有实体"""
        with self.driver.session() as session:
            if entity_type:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.entity_type = $type
                    RETURN e.name AS name, e.entity_type AS type,
                           e.description AS description
                    LIMIT $limit
                    """,
                    type=entity_type, limit=limit
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.entity_type AS type,
                           e.description AS description
                    LIMIT $limit
                    """,
                    limit=limit
                )
            return [dict(r) for r in result]

    def get_all_relations(self, limit: int = 1000):
        """获取所有关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                RETURN h.name AS source, t.name AS target, type(r) AS relation,
                       h.entity_type AS source_type, t.entity_type AS target_type
                LIMIT $limit
                """,
                limit=limit
            )
            return [dict(r) for r in result]

    def get_graph_stats(self):
        """获取图谱统计信息"""
        with self.driver.session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            types = session.run(
                "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
            )
            type_stats = {r["type"]: r["count"] for r in types}
            return {"total_entities": nodes, "total_relations": rels, "entity_types": type_stats}

    def delete_entity(self, name: str):
        """删除实体及其相关关系"""
        with self.driver.session() as session:
            session.run(
                "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                name=name
            )

    def clear_all(self):
        """清空整个图谱"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def merge_entities(self, keep_name: str, remove_name: str):
        """合并两个实体（知识融合 - 共指消解）"""
        with self.driver.session() as session:
            # 将remove_name的所有关系转移到keep_name
            session.run(
                """
                MATCH (old:Entity {name: $remove})-[r]->(target:Entity)
                MATCH (keep:Entity {name: $keep})
                WHERE NOT (keep)-[:SAME_AS]->(target) AND keep <> target
                WITH keep, target, type(r) AS relType, properties(r) AS props
                CALL apoc.create.relationship(keep, relType, props, target) YIELD rel
                RETURN count(rel)
                """,
                remove=remove_name, keep=keep_name
            )
            session.run(
                """
                MATCH (source:Entity)-[r]->(old:Entity {name: $remove})
                MATCH (keep:Entity {name: $keep})
                WHERE NOT (source)-[:SAME_AS]->(keep) AND source <> keep
                WITH source, keep, type(r) AS relType, properties(r) AS props
                CALL apoc.create.relationship(source, relType, props, keep) YIELD rel
                RETURN count(rel)
                """,
                remove=remove_name, keep=keep_name
            )
            # 删除旧节点
            session.run(
                "MATCH (e:Entity {name: $remove}) DETACH DELETE e",
                remove=remove_name
            )

    def simple_merge_entities(self, keep_name: str, remove_name: str):
        """简化版合并实体（不依赖APOC插件）"""
        with self.driver.session() as session:
            # 获取旧实体的所有出边
            out_rels = session.run(
                """
                MATCH (old:Entity {name: $remove})-[r]->(target:Entity)
                RETURN type(r) AS relType, target.name AS targetName
                """,
                remove=remove_name
            )
            for rec in out_rels:
                session.run(
                    f"""
                    MATCH (keep:Entity {{name: $keep}})
                    MATCH (target:Entity {{name: $target}})
                    MERGE (keep)-[:{rec['relType']}]->(target)
                    """,
                    keep=keep_name, target=rec["targetName"]
                )

            # 获取旧实体的所有入边
            in_rels = session.run(
                """
                MATCH (source:Entity)-[r]->(old:Entity {name: $remove})
                RETURN type(r) AS relType, source.name AS sourceName
                """,
                remove=remove_name
            )
            for rec in in_rels:
                session.run(
                    f"""
                    MATCH (source:Entity {{name: $source}})
                    MATCH (keep:Entity {{name: $keep}})
                    MERGE (source)-[:{rec['relType']}]->(keep)
                    """,
                    source=rec["sourceName"], keep=keep_name
                )

            # 删除旧节点
            session.run(
                "MATCH (e:Entity {name: $remove}) DETACH DELETE e",
                remove=remove_name
            )
