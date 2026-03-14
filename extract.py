import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 1. 定义底层结构（Ontology）作为 LLM 的核心依据
ONTOLOGY = {
    "entities": ["Syntax", "Concept", "Interface", "Class", "Keyword", "Error"],
    "relations": ["DEPENDS_ON", "INCLUDES", "IMPLEMENTS", "EXTENDS", "EXPLAINS", "THROWN_BY"]
}


class JavaKGBuilder:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

    def extract_and_reflect(self, text_chunk):
        # 步骤 1: 初始抽取 (符合开题报告中的强类型约束)
        extract_prompt = f"""
        你是一个 Java 教育专家。请从文本中提取知识图谱。
        约束：
        - 实体类型必须属于: {ONTOLOGY['entities']}
        - 关系类型必须属于: {ONTOLOGY['relations']}
        输出格式: [{{'s': '头实体', 'p': '关系', 'o': '尾实体', 's_type': '类型'}}]

        文本内容: {text_chunk}
        """
        initial_kg = self.llm.invoke(extract_prompt).content

        # 步骤 2: 反思与修正 (Agentic Workflow 核心)
        reflect_prompt = f"""
        作为审计员，检查以下 Java 知识三元组是否存在逻辑错误（如：volatile 应 EXPLAINS JMM）。
        原始文本: {text_chunk}
        当前三元组: {initial_kg}

        如果准确无误，回复 'PASS'；否则列出错误并给出修正后的 JSON 列表。
        """
        reflection_res = self.llm.invoke(reflect_prompt).content

        if "PASS" in reflection_res.upper():
            return initial_kg
        else:
            # 简单演示：这里可以根据 reflection_res 再次调用 llm 进行修正
            return reflection_res


# 2. 模拟运行
raw_text = "volatile 关键字是 Java 提供的一种轻量级同步机制，它保证了变量在 JMM 中的可见性。"
builder = JavaKGBuilder(api_key="your_key")
final_triples = builder.extract_and_reflect(raw_text)

print("最终提取的 Java 知识三元组：\n", final_triples)