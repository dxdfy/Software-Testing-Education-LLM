# -*- coding: utf-8 -*-
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

class RAG:
    def __init__(
        self,
        file_path,
        persist_dir="./chroma_db",
        model_name="BAAI/bge-large-zh-v1.5",
        device="cpu",
        k=3
    ):
        self.file_path = file_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.device = device
        self.embedding = self._get_embedding_model()
        self.db = self._init_vector_db()
        self.retriever = self.db.as_retriever(search_kwargs={"k": k})

    def _get_embedding_model(self):
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _init_vector_db(self):
        """初始化向量数据库"""
        if os.path.exists(self.persist_dir):
            print(f"加载已有数据库: {self.persist_dir}")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )
        else:
            print(f"创建新数据库: {self.persist_dir}")
            documents = self._load_documents()
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_dir
            )

    def _load_documents(self):
        """加载并分割文档"""
        try:
            with open(self.file_path, encoding='utf-8') as f:
                state_of_the_union = f.read()  # 读取文件返回字符串
                texts = state_of_the_union.split("###")  # 按行分割文本
                documents = [
                    Document(page_content=text, metadata={"source": self.file_path})
                    for i, text in enumerate(texts)
                ]
            return documents
        except Exception as e:
            raise ValueError(f"加载文档失败: {e}")

    def retrieve(self, query):
        """执行检索并返回上下文"""
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
# if __name__ == '__main__':
#     # 文件路径
#     file_path = "../docs/ch1软件测试知识库.txt"
#     # 创建或加载向量数据库
#     rag = RAG(file_path)
#     # 创建检索器
#     # retriever = db.as_retriever(search_kwargs={"k": 2})
#     # 查询示例
#     query = "什么是软件缺陷"
#     retrieved_docs = rag.retrieve(query)
#     print(retrieved_docs)