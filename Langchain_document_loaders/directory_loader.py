from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

books_path = Path(__file__).parent / "books"

loader = DirectoryLoader(
    path=str(books_path),
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

print(f"Total documents loaded: {len(docs)}")

for doc in docs:
    print(doc.page_content[:200])