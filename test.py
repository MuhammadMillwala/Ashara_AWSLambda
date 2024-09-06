
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
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pickle


with open(r"D:\ashara_lamda_function\ashara_chat.pkl", "rb") as f:
    db = pickle.load(f)
    
print(db)    
embed = OpenAIEmbeddings()
embeddings = embed.embed_query("what are the two famous hotels in karachi?")
qdrant = Qdrant(
        client=db,
        collection_name="Ashara",
        embeddings=embed
    )

question = "what are the two famous hotels in karachi?"
context = ""

template = """
    You are a support agent for the Ashara an event in Dawoodi Bohra Community in Karachi,Pakistan for the year 1446, 
    do not play any other role.You will help our community members understand details about the event in an easy 
    to understand way. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you can only answer Ashara related questions,
    if you are asked about details in Karachi you can answer them using your knowledge.  
    don't try to make up an answer. Use five sentences maximum. Keep the answer as concise as possible with a 
    word limit of about 300 tokens or 50-100 words.
    Context: {context}

    Question: {question}"
    Helpful Answer:
    """

prompt_with_results = template + question
prompt_template = PromptTemplate(input_variables=["context","question"], template=prompt_with_results)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=300)

mem = ConversationBufferWindowMemory(memory_key="chat_history",k=2, return_messages=True,input_key="question",output_key='answer')
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=qdrant.as_retriever(search_type="similarity", search_kwargs={"k":3}), memory=mem, verbose=True, combine_docs_chain_kwargs={"prompt": prompt_template})

result = qa_chain({"question": prompt_with_results})['answer']

# ret = db.search(collection_name="Ashara", query_vector=embeddings, limit=3)
print(result)