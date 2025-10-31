import hashlib
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class ChromaUtils:
    def __init__(self, embedding_model: HuggingFaceEmbeddings, persist_dir: str):
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir

    def generate_collection_name(self, file_path: str) -> str:
        """
        根据文件路径生成 collection_name（使用文件的 SHA256 哈希值）
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Invalid file path: {file_path}")

        # 计算文件的 SHA256 哈希
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()
        file_name = os.path.basename(file_path).split('.')[0]
        file_type = os.path.basename(file_path).split('.')[-1]
        collection_name = f"{file_hash[:8]}_{file_name}_{file_type}"  # 使用前 8 位哈希值
        return collection_name

    def create_chroma_db(self, file_path: str) -> Chroma:
        """
        根据文件路径生成 collection_name 并创建 Chroma 数据库
        """
        collection_name = self.generate_collection_name(file_path)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir
        )
        return db


# # 示例：如何使用这个工具类
# embedding_model = HuggingFaceEmbeddings(model_name="./local_embedding/bge-small-en")
# persist_dir = "./chroma_db"
#
# chroma_utils = ChromaUtils(embedding_model, persist_dir)
#
# file_path = "./data/sample_file.txt"  # 输入文件路径
# db = chroma_utils.create_chroma_db(file_path)
#
# # 向数据库添加文本
# texts = ["Chroma is awesome!", "LangChain integrates well."]
# db.add_texts(texts)
# db.persist()
#
# print(f"Collection name: {db.collection_name}")
