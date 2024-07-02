from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()
print(os.environ["PINECONE_API_KEY"])

docs = WikipediaLoader(query="Aran Island", load_max_docs=3).load()

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError(f"Environment variable OPENAI_API_KEY is not set")

if "INDEX_NAME" not in os.environ:
    raise EnvironmentError(f"Environment variable INDEX_NAME is not set")

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
PineconeVectorStore.from_documents(
    docs, embeddings,
    index_name=os.environ.get("INDEX_NAME")
    )