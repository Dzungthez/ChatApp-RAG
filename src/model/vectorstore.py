from typing import Union, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class MyDatabase:
    def __init__(self, documents=None, vectorstore: Union[FAISS, Chroma] = Chroma,
                 embeddings=HuggingFaceEmbeddings()):
        self.documents = documents
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.database = self.build_db(documents)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to database
        :param documents: List of documents to add
        :return: None
        """
        self.documents.extend(documents)
        self.database.add_documents(documents)

    def build_db(self, documents: List[Document]) -> Union[FAISS, Chroma]:
        """
        Build database from documents by vectorizing them
        :param documents: List of documents to build the database from
        :return: A vectorstore object
        """
        db = self.vectorstore.from_documents(documents, self.embeddings)
        return db

    def get_retriever(self, kwargs=None):
        if kwargs is None:
            kwargs = {'k': 8}
        return self.database.as_retriever(search_type="similarity",**kwargs)
