from langchain_community.document_loaders import CSVLoader
from pathlib import Path

csv_path = Path(__file__).parent / "Social_Network_Ads.csv"

loader = CSVLoader(file_path=str(csv_path))

docs = loader.load()

print(len(docs))
print(docs[1])