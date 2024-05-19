import multiprocessing
from typing import Union, List, Literal
import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader as pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_noised_character(text: str) -> str:
    """
    Remove special characters in the document to reduce context memory length.
    """
    return ' '.join(''.join(c for c in word if ord(c) < 128) for word in text.split(' '))


def load_pdf(file):
    """
    load content from pdf file using PyPDF loader from langchain
    :param file: file path
    :return: list of page content in data type: Document (langchain)
    """
    docs = pypdf(file).load()
    for doc in docs:
        doc.page_content = remove_noised_character(doc.page_content)
    return docs


def get_num_processing() -> int:
    """
    Get number of processing cores
    :return: number of processing cores
    """
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self):
        self.num_process = get_num_processing()

    def __call__(self, *args, **kwargs):
        pass


class PDFLoader(BaseLoader):
    """
    PDF loader class to load pdf files and return list of documents
    """
    def __init__(self):
        super().__init__()

    def __call__(self, files: List[str], **kwargs) -> List:
        num_processing = min(self.num_process, kwargs['workers'])
        with multiprocessing.Pool(num_processing) as pool:
            docs = []
            for doc in tqdm(pool.imap_unordered(load_pdf, files), total=len(files)):
                docs.extend(doc)
        return docs


class TextSplitter:
    """
    Text splitter class to split text into chunks, using RecursiveCharacterTextSplitter from langchain
    """
    def __init__(self, chunk_size=300, chunk_overlap=0) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(self, documents):
        """
        Split document into chunks
        :param documents: List[Document]
        :return:
        """
        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        :param text: text content
        :return: list of chunks
        """
        return self.splitter.split_text(text)

    def __call__(self, documents):
        with multiprocessing.Pool() as pool:
            results = pool.map(self.split_document, documents)
        split_docs = [doc for sublist in results for doc in sublist]
        return split_docs


class Loader:
    def __init__(self, file_type: str = Literal["pdf"],
                 split_kwargs: dict = {"chunk_size": 300, "chunk_overlap": 0}):
        assert file_type in ["pdf"], "wrong file type, only support pdf file type"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("wrong file type, only support pdf file type..")

        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=workers)

