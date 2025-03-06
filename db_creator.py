from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
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
            texts = state_of_the_union.split("\n")  # 按行分割文本
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


if __name__ == '__main__':
    # 文件路径
    file_path = "docs/ch1软件测试知识库.txt"
    # 创建或加载向量数据库
    db = create_or_load_vector_db(file_path)
    # 创建检索器
    retriever = db.as_retriever(search_kwargs={"k": 2})
    # 查询示例
    query = "软件缺陷"
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs)