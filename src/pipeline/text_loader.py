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
    :param file:
    :return:
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
    def __init__(self):
        super().__init__()

    def __call__(self, files: List[str], **kwargs) -> List:
        num_processing = min(self.num_process, kwargs['workers'])
        with multiprocessing.Pool(self.num_process) as pool:
            docs = []
            for doc in tqdm(pool.imap(load_pdf, files), total=len(files)):
                docs.extend(doc)
        return docs
