import os
import boto3
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.runnables import RunnableSequence, RunnablePassthrough


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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=PERSIST_DIR)
    db.persist()
    print("Vector DB created & saved.")


# =========================
#         TOOL
# =========================

@tool
def search_pdf(query: str):
    """Searches the stored document and returns best matching text."""
    results = db.max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5)

    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"

    return output


# =========================
#    LLM + Prompt + Chain
# =========================

llm = ChatBedrock(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    client=session
)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an efficient assistant answering questions.

Use the provided context to respond clearly and concisely in **3 sentences max**.
If the answer is unknown, respond: "I could not find this in the document."

Question: {question}

Context:
{context}

Answer:
"""
)

#  RAG PIPELINE
chain = RunnableSequence(
    {
        "question": RunnablePassthrough(),
        "context": search_pdf  # Using TOOL inside chain
    }
    | prompt
    | llm
)


# =========================
#        Main Handler
# =========================

def ask_chatbot(question: str):
    """Use this tool for all government scheme information, including scheme details,
    eligibility, benefits, required documents, and general questions.
    Do NOT use this tool for student laptop allocation or delivery status."""
    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else response


# =========================
#        CLI Mode
# =========================

if __name__ == "__main__":
    print("\n RAG PDF Chatbot Ready!")
    while True:
        query = input("\nAsk something about the document (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_chatbot(query)
        print("\n Answer:", answer)