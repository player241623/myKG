"""
Flask 后端主应用
提供知识图谱和问答系统的 RESTful API
"""
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from neo4j_client import Neo4jClient
from kg_extractor import build_kg_from_texts, extract_triples_from_text
from kg_fusion import run_knowledge_fusion
from graphrag_qa import answer_question
from data_crawler import crawl_java_tutorials, load_local_tutorials, load_custom_text
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, ONTOLOGY

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# 全局 Neo4j 客户端
neo4j = None


def get_neo4j():
    global neo4j
    if neo4j is None:
        neo4j = Neo4jClient()
    return neo4j


# ============ 前端页面 ============
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ============ 知识图谱 API ============

@app.route("/api/graph/stats", methods=["GET"])
def graph_stats():
    """获取图谱统计信息"""
    try:
        stats = get_neo4j().get_graph_stats()
        return jsonify({"code": 0, "data": stats})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/entities", methods=["GET"])
def get_entities():
    """获取所有实体"""
    try:
        entity_type = request.args.get("type")
        limit = int(request.args.get("limit", 500))
        entities = get_neo4j().get_all_entities(entity_type=entity_type, limit=limit)
        return jsonify({"code": 0, "data": entities})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/relations", methods=["GET"])
def get_relations():
    """获取所有关系"""
    try:
        limit = int(request.args.get("limit", 1000))
        relations = get_neo4j().get_all_relations(limit=limit)
        return jsonify({"code": 0, "data": relations})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/search", methods=["GET"])
def search_entity():
    """搜索实体"""
    try:
        keyword = request.args.get("keyword", "")
        if not keyword:
            return jsonify({"code": -1, "msg": "缺少搜索关键词"})
        results = get_neo4j().search_entity(keyword)
        return jsonify({"code": 0, "data": results})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/entity/<name>", methods=["GET"])
def get_entity_detail(name):
    """获取实体详情及邻居"""
    try:
        depth = int(request.args.get("depth", 2))
        detail = get_neo4j().get_entity_with_neighbors(name, depth=depth)
        if detail:
            return jsonify({"code": 0, "data": detail})
        return jsonify({"code": -1, "msg": f"未找到实体: {name}"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/path", methods=["GET"])
def get_path():
    """获取两个实体之间的路径"""
    try:
        e1 = request.args.get("entity1")
        e2 = request.args.get("entity2")
        if not e1 or not e2:
            return jsonify({"code": -1, "msg": "请提供两个实体名称"})
        path = get_neo4j().get_path_between(e1, e2)
        if path:
            return jsonify({"code": 0, "data": path})
        return jsonify({"code": -1, "msg": "未找到路径"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/entity", methods=["POST"])
def add_entity():
    """手动添加实体"""
    try:
        data = request.json
        result = get_neo4j().create_entity(
            name=data["name"],
            entity_type=data.get("entity_type", "Concept"),
            description=data.get("description", "")
        )
        return jsonify({"code": 0, "msg": "实体创建成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/relation", methods=["POST"])
def add_relation():
    """手动添加关系"""
    try:
        data = request.json
        get_neo4j().create_relation(
            head=data["head"],
            relation=data["relation"],
            tail=data["tail"]
        )
        return jsonify({"code": 0, "msg": "关系创建成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/entity/<name>", methods=["DELETE"])
def delete_entity(name):
    """删除实体"""
    try:
        get_neo4j().delete_entity(name)
        return jsonify({"code": 0, "msg": "删除成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/graph/clear", methods=["POST"])
def clear_graph():
    """清空图谱"""
    try:
        get_neo4j().clear_all()
        return jsonify({"code": 0, "msg": "图谱已清空"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


# ============ 知识图谱构建 API ============

@app.route("/api/build/crawl", methods=["POST"])
def crawl_and_build():
    """爬取 Java 教程并构建知识图谱"""
    try:
        max_pages = request.json.get("max_pages", 10) if request.json else 10
        # 爬取教程
        tutorials = crawl_java_tutorials(max_pages=max_pages)
        if not tutorials:
            return jsonify({"code": -1, "msg": "爬取失败，未获取到教程内容"})

        # 构建知识图谱
        triples = build_kg_from_texts(tutorials, get_neo4j())
        return jsonify({
            "code": 0,
            "data": {
                "pages_crawled": len(tutorials),
                "triples_extracted": len(triples)
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/build/local", methods=["POST"])
def build_from_local():
    """从本地已爬取的数据构建知识图谱"""
    try:
        tutorials = load_local_tutorials()
        if not tutorials:
            return jsonify({"code": -1, "msg": "未找到本地教程数据，请先执行爬取"})

        triples = build_kg_from_texts(tutorials, get_neo4j())
        return jsonify({
            "code": 0,
            "data": {"triples_extracted": len(triples)}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/build/text", methods=["POST"])
def build_from_text():
    """从用户提交的文本构建知识图谱"""
    try:
        data = request.json
        title = data.get("title", "用户输入")
        content = data.get("content", "")
        if not content:
            return jsonify({"code": -1, "msg": "文本内容为空"})

        texts = [{"title": title, "content": content}]
        triples = build_kg_from_texts(texts, get_neo4j())
        return jsonify({
            "code": 0,
            "data": {
                "triples_extracted": len(triples),
                "triples": triples
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/api/build/fusion", methods=["POST"])
def run_fusion():
    """执行知识融合"""
    try:
        result = run_knowledge_fusion(get_neo4j())
        return jsonify({"code": 0, "data": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": str(e)})


# ============ 智能问答 API ============

@app.route("/api/qa/ask", methods=["POST"])
def ask_question():
    """GraphRAG 智能问答"""
    try:
        data = request.json
        question = data.get("question", "")
        if not question:
            return jsonify({"code": -1, "msg": "请输入问题"})

        result = answer_question(question, get_neo4j())
        return jsonify({"code": 0, "data": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": str(e)})


# ============ 系统信息 API ============

@app.route("/api/ontology", methods=["GET"])
def get_ontology():
    """获取 Ontology 定义"""
    return jsonify({"code": 0, "data": ONTOLOGY})


if __name__ == "__main__":
    print("=" * 60)
    print("  基于知识图谱和大模型的 Java 课程问答系统")
    print("  启动中...")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
