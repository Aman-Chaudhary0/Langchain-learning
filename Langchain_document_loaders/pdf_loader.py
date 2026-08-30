from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "dl-curriculum.pdf"

loader = PyPDFLoader(str(pdf_path))

docs = loader.load()

print(docs)
print(len(docs))