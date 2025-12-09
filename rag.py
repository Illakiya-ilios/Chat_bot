import os
import boto3
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock

from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List


# =========================
#   AWS + Environment Setup
# =========================
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

session = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print(" AWS Authentication Successful")


# =========================
#    Data + Vector Store
# =========================
PDF_NAME = "elcot-poc.pdf"
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("Loaded existing Vector DB.")
else:
    loader = PyPDFLoader(PDF_NAME)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    documents = splitter.split_documents(docs)

    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=PERSIST_DIR)
    db.persist()
    print("Vector DB created & saved.")


# ==========================================================
#           HYBRID SEARCH (BM25 + VECTOR SEMANTIC)
# ==========================================================

class HybridRetriever(BaseRetriever):
    """
    Combines BM25 (lexical) and Vector (semantic) search with deduplication.
    """

    bm25_retriever: BaseRetriever
    vector_retriever: BaseRetriever
    k: int = 10

    def _get_relevant_documents(self, query: str) -> List[Document]:

        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        
        combined = bm25_docs + vector_docs

        seen = set()
        unique_docs = []

        for doc in combined:
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
                if len(unique_docs) >= self.k:
                    break

        return unique_docs[:self.k]


def setup_hybrid_retriever():
    print("\nSetting up BM25 retriever...")
    all_docs = db.get(include=["documents", "metadatas"])  # load all docs
    documents = [
        Document(page_content=d, metadata=m)
        for d, m in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 10

    print("Setting up Vector retriever...")
    vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    print("Creating Hybrid retriever...")
    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector_retriever,
        k=10
    )


hybrid_retriever = setup_hybrid_retriever()


# =========================
#         TOOL
# =========================

@tool
def search_pdf(query: str):
    """Searches the stored document and returns best matching text using hybrid search."""
    
    results = hybrid_retriever.invoke(query)

    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"

    return output


# =========================
#    LLM + Prompt + Chain
# =========================

llm = ChatBedrock(
    model="meta.llama3-70b-instruct-v1:0",
    temperature=0,
    client=session
)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Answer like you are the first person in this conversation. You are a government policy advisor. Your task is to explain details, in a professional but simple manner, as if assisting a citizen.
Always base your answers strictly on the provided context.
Do not include information from outside the PDF.

When responding:
1)Be polite, helpful, and factual
2)Summarize only what is relevant
3)If context lacks details, say: “I don’t have that information available right now..”
4)Read the full context before answering.
5)Keep answers concise (2–4 sentences)
6)Quote specific details when available.
Format:

Context: {context}

Question: {question}

"""
)


chain = RunnableSequence(
    {
        "question": RunnablePassthrough(),
        "context": search_pdf
    }
    | prompt
    | llm
)


# =========================
#        Main Handler
# =========================

def ask_chatbot(question: str):
    """Use this tool for all government scheme information."""
    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else response


# =========================
#        CLI Mode
# =========================

if __name__ == "__main__":
    print("\n RAG PDF Chatbot Ready!")
    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_chatbot(query)
        print("\n Answer:", answer)
