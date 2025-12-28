import os
import zipfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found in .env file")

ZIP_PATH = "class6_english_syllabus.zip"
EXTRACT_DIR = "extracted_pdfs"
VECTOR_DB_DIR = "vectordb"

# 1. Extract ZIP
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# 2. Load PDFs
documents = []

for root, _, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

if not documents:
    raise Exception("No PDFs found inside ZIP")

# 3. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# 4. Create embeddings + vector DB
embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)

db.persist()
print("✅ Ingestion complete. Vector DB ready.")
