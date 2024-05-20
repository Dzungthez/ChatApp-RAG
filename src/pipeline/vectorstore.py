from typing import Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class MyDatabase:
    def __init__(self, documents=None, vectorstore: Union[FAISS, Chroma] = None,
                 embeddings=HuggingFaceEmbeddings()):
        self.documents = documents
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.database = self.build_db(documents)

    def add_documents(self, documents):
        """
        Add new documents to database
        :param documents:
        :return:
        """
        self.documents.extend(documents)
        self.database.add_documents(documents)

    def build_db(self, documents):
        """
        Build database from documents by vectorizing them
        :param documents:
        :return:
        """
        db = self.vectorstore.from_documents(self.documents, self.embeddings)
        return db

    def get_retriever(self, kwargs=None):
        if kwargs is None:
            kwargs = {'k': 10}
        return self.database.as_retriever(search_type="similarity",**kwargs)
