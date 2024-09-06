
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import uuid
import os
import openai
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import pickle


# Load environment variables from .env file
load_dotenv(".env")

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the Qdrant collection name
qdrant_collection = 'Ashara'

def process_pdf(pdf_path, max_chunk_size=4096):
   
    try:
        from PyPDF2 import PdfReader  # Import PyPDF2 for PDF processing
    except ImportError:
        print("Error: PyPDF2 library not installed. Please install using 'pip install PyPDF2'.")
        return []

    for i in os.listdir(r"D:\ashara_lamda_function\documents\\"):
        if i.endswith(".pdf"):
            reader = PdfReader(r"D:\ashara_lamda_function\documents\\"+str(i))
            docs = []
            cnt = 0
            for page in reader.pages:
                docs.append(page.extract_text())
                cnt+=1
                
    return docs

def generate_embeddings(text):
    embed = OpenAIEmbeddings()
    embeddings = embed.embed_query(text)
    return embeddings

def save_embeddings_to_qdrant(documents, client):

    #   embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAIEmbeddings instance

    points = []
    for doc in documents:
        try:
            embeddings_vector = generate_embeddings(doc)
        except Exception as e:
            print(f"Error generating embeddings for document '{doc[:100]}': {e}")
            continue  # Skip document on error

        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embeddings_vector,
            payload={"text":doc}
        ))
    client.upsert(
        collection_name=qdrant_collection,
        points=points
    )
    return client


def create_qdrant_collection(force_recreate=False):
    client = QdrantClient(":memory:")
    # qdrant_client = Qdrant(client, qdrant_collection)
    client.recreate_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    return client

def main():
    """
    Main function to process PDF, generate embeddings, and save to Qdrant (local storage).
    """
    client = create_qdrant_collection()

    pdf_path = 'ZonalDepartment.pdf'
    documents = process_pdf(pdf_path)

    if documents:
        db = save_embeddings_to_qdrant(documents, client)
        print('Embeddings generated and saved to Qdrant (local storage)')
        
        with open(r"D:\ashara_lamda_function\ashara_chat.pkl", "wb") as f:
            pickle.dump(db, f)
    else:
        print("No documents created from PDF. Please check the file or adjust processing logic.")

if __name__ == "__main__":
  main()