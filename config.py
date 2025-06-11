# config.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# Load Azure and DB credentials from environment
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")

AZURE_OPENAI_ENDPOINT_EMBEDDING = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
AZURE_OPENAI_API_KEY_EMBEDDING = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
API_VERSION_EMBEDDING = os.getenv("API_VERSION_EMBEDDING", "2024-02-01")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Initialize Azure OpenAI Chat Model
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    openai_api_version=API_VERSION,
    
)

# Initialize Azure OpenAI Embeddings Model
embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT_EMBEDDING,
    api_key=AZURE_OPENAI_API_KEY_EMBEDDING,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    openai_api_version=API_VERSION_EMBEDDING
)
print("Configuration loaded and clients initialized.")