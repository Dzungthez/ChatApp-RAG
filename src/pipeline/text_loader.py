import multiprocessing
from typing import Union, List
import glob
import pdfplumber
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_noised_character(text: str) -> str:
    return ''.join([c if ord(c) < 128 else ' ' for c in text])


def load_pdf(file):
    docs = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text = remove_noised_character(text)
                docs.append(Document(page_content=text))
    return docs


def get_num_processing() -> int:
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self):
        self.num_process = get_num_processing()

    def __call__(self, *args, **kwargs):
        pass


class PDFLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def __call__(self, files: List[str], **kwargs) -> List[Document]:
        num_processing = min(self.num_process, kwargs.get('workers', 4))
        with multiprocessing.Pool(num_processing) as pool:
            docs = []
            for doc in tqdm(pool.imap_unordered(load_pdf, files), total=len(files)):
                docs.extend(doc)
        return docs


class TextSplitter:
    def __init__(self, chunk_size=300, chunk_overlap=0) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(self, document: Document) -> List[Document]:
        return self.splitter.split_documents([document])

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def __call__(self, documents: List[Document]) -> List[Document]:
        with multiprocessing.Pool() as pool:
            results = pool.map(self.split_document, documents)
        split_docs = [doc for sublist in results for doc in sublist]
        return split_docs


class Loader:
    def __init__(self, split_kwargs=None):
        if split_kwargs is None:
            split_kwargs = {"chunk_size": 300, "chunk_overlap": 0}
        self.doc_loader = PDFLoader()
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers=4) -> List[Document]:
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split

    def load_text(self, text: Union[str, List[str]]) -> List[Document]:
        text_data = text
        if isinstance(text, str):
            text_data = [text]
        docs = [Document(page_content=remove_noised_character(text_)) for text_ in text_data]
        doc_split = self.doc_splitter(docs)
        return doc_split

    def load_dir(self, dir_path: str, workers=4) -> List[Document]:
        files = glob.glob(f"{dir_path}/*.pdf")
        assert len(files) > 0, f"No pdf files found in {dir_path}"
        return self.load(files, workers=workers)
