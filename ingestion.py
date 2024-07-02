from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from llama_index.readers.wikipedia import WikipediaReader

load_dotenv()

reader = WikipediaReader()
docs = reader.load_data(pages=["aran island"])

print(type(docs[0]))

# text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
# texts = text_splitter.split_documents(docs)
# print(f"created {len(texts)} chunks")