from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import pandas as pd
import re
DB_DIRECTORY = "./chroma_db"               # 向量数据库存储路径
QUERY_FILE = "../docs/queries.txt"         # 查询文件路径（使用###分隔）
SOURCE_FILE = "../docs/软件测试知识库.txt"  # 原始文档路径
OUTPUT_FILE = "../docs/results5.txt"       # 输出检索结果路径
MODEL_NAME = "BAAI/bge-large-zh-v1.5"     # 嵌入模型
def read_and_split(file_path):
    """
    读取文件并将内容分割为文档列表。
    :param file_path: 文件路径
    :return: 文档列表（List[Document]）
    """
    try:
        # 打开文件并读取内容
        with open(file_path, encoding='utf-8') as f:
            state_of_the_union = f.read()  # 读取文件返回字符串
            texts = state_of_the_union.split("###")  # 按符号分割文本
            # for text in texts:
            #     print(text)
            documents = [
                Document(page_content=text, metadata={"source": file_path})
                for i, text in enumerate(texts)
            ]

        return documents
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []


def create_or_load_vector_db(file_path, persist_directory="./chroma_db", model_name="BAAI/bge-large-zh-v1.5"):
    """
    创建或加载向量数据库。

    :param file_path: 文本文件路径
    :param persist_directory: 持久化存储目录
    :param model_name: 嵌入模型名称
    :return: Chroma 向量数据库对象
    """
    # 初始化嵌入模型
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hfe = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # 如果持久化目录已存在，则直接加载向量数据库
    if os.path.exists(persist_directory):
        print(f"加载已有的向量数据库: {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,  # 指定持久化目录
            embedding_function=hfe  # 使用相同的嵌入模型
        )
    else:
        # 如果持久化目录不存在，则创建新的向量数据库
        print(f"创建新的向量数据库: {persist_directory}")
        texts = read_and_split(file_path)
        db = Chroma.from_documents(
            documents=texts,  # 文本数据
            embedding=hfe,  # 嵌入模型
            persist_directory=persist_directory  # 持久化存储目录
        )
    return db
def read_queries(file_path):
    """读取查询文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return [q.strip() for q in content.split('###') if q.strip()]

def batch_retrieval(retriever, queries):
    """批量检索并格式化结果"""
    results = []
    for query in queries:
        docs = retriever.invoke(query)
        answer = "\n\n".join([doc.page_content for doc in docs])
        results.append({"query": query, "answer": answer})
    return results


def clean_excel_text(text):
    """
    清理 Excel 不支持的非法字符
    - 移除 ASCII 控制字符（0-31，除了换行符 \n 和回车符 \r）
    - 替换特殊符号（如不间断空格）
    """
    # 移除 ASCII 控制字符（保留换行符和回车符）
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', '', text)
    # 替换特殊空格
    text = text.replace('\u00A0', ' ')
    return text

def save_to_txt(results, output_path):
    """将结果保存为TXT文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(results, 1):
            f.write(f"=== 查询 {idx} ===\n")
            f.write(f"问题：{result['query']}\n\n")
            f.write("回答：\n")
            f.write(result['answer'])
            f.write("\n" + "="*60 + "\n")


if __name__ == '__main__':
    # 文件路径
    file_path = "../docs/软件测试知识库.txt"
    db = create_or_load_vector_db(file_path)
    # # 创建检索器
    retriever = db.as_retriever(search_kwargs={"k": 5})
    # 查询示例
    queries = read_queries("../docs/querys.txt")
    print(f"共加载 {len(queries)} 个查询")

    # for q in queries:
    #     print("""-------------------------""")
    #     print(q)
    results = batch_retrieval(retriever, queries)
    save_to_txt(results, OUTPUT_FILE)