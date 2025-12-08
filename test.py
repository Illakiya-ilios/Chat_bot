import os
import boto3
from dotenv import load_dotenv
import json
from typing import Annotated, Literal

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


# ========================
# Load Environment
# ========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "elcotlaptopscheme")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
PDF_NAME = "elcot-poc.pdf"

# ========================
# AWS Bedrock Client
# ========================
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

print("✓ AWS Authentication Successful")


# ========================
# Database Connection (SQL Tool Setup)
# ========================
uri = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(uri, include_tables=["students"])
sql_tool = QuerySQLDatabaseTool(db=db)


# ========================
# Vector Store Setup (PDF Tool)
# ========================
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("✓ Loaded existing Vector DB.")
else:
    loader = PyPDFLoader(PDF_NAME)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)
    vector_db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=PERSIST_DIR)
    vector_db.persist()
    print("✓ Vector DB created & saved.")


# ========================
# LLM Model
# ========================
llm = ChatBedrock(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock
)


# ========================
# SQL Generator Prompt
# ========================
sql_prompt = PromptTemplate.from_template(
"""
You are an expert SQL assistant who converts user questions into SQL queries
to retrieve data from a database table called `students`.

Database schema:
{schema}

Columns available:
id, student_id, email, indent_id, password, is_password_reset, reason, salutation,
student_name_certificate, student_name_aadhaar, dob, gender, blood_group,
nationality, religion, community, caste, aadhaar_number, passport_number,
is_first_graduate, first_graduate_cert_number, is_differently_abled,
mobile_number, district, taluk, village, location_type, block, pincode,
postal_address, is_orphan, father_name, father_occupation, mother_name,
mother_occupation, guardian_name, guardian_mobile, annual_family_income,
ifsc_code, bank_name, branch, bank_city, account_type, account_number,
academic_year_joined, stream_type, course_type, course, academic_branch,
medium_instruction, mode_of_study, admission_date, admission_type,
admission_number, registration_number, year_of_study, laptop_allotted,
delivery_status, otp, otp_created_at, otp_expires_at, financial_year_id,
status, is_deleted, created, modified, created_by, modified_by,
deleted_at, created_at, updated_at, action

You must first check whether the user has provided enough identifying information
(such as student ID, registered name, email, or mobile number).

- If NOT enough details are given, respond with:
  "Please provide more identifying details such as your student ID, registered mobile number, or full name so that I can retrieve your laptop allocation status."

- If enough information is provided, generate a syntactically correct SQL query that retrieves:
  id, student_id, student_name_certificate AS name, email, mobile_number, laptop_allotted, delivery_status, created_at
  from the students table.

Use the following examples as reference:

Example 1:
Question: What is the laptop allocation status of student ID STUVEL007?
SQL Query:
SELECT id, student_id, student_name_certificate AS name, email, mobile_number, laptop_allotted, delivery_status, created_at
FROM students
WHERE student_id = 'STUVEL007'
LIMIT 1;

Example 2:
Question: Find details for Vel Geetha
SQL Query:
SELECT id, student_id, student_name_certificate AS name, email, mobile_number, laptop_allotted, delivery_status, created_at
FROM students
WHERE student_name_certificate ILIKE '%Vel Geetha%'
LIMIT 1;

Only return one of the following:
- A direct polite message asking for more details (if insufficient info)
- OR the SQL query (if sufficient info)

Question: {question}
"""
)

generate_query = sql_prompt | llm | StrOutputParser()


# ========================
# PDF RAG Prompt
# ========================
pdf_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are an efficient assistant answering questions about government schemes.

Use the provided context to respond clearly and concisely in 3 sentences max.
If the answer is unknown, respond: "I could not find this in the document."

Question: {question}

Context:
{context}

Answer:"""
)


# ========================
# Internal Functions
# ========================

def search_pdf_internal(query: str) -> str:
    """Search vector store for relevant PDF context"""
    results = vector_db.max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5)

    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"

    return output


def execute_sql_query(question: str) -> str:
    """Generate and execute SQL query for laptop status"""
    schema = db.get_table_info()
    query = generate_query.invoke({"question": question, "schema": schema})

    # Check if we need more information
    if "Please provide" in query:
        return query

    # Execute the SQL query
    result = sql_tool.invoke(query)
    return result


def ask_pdf_rag(question: str) -> str:
    """RAG pipeline for PDF queries"""
    chain = RunnableSequence(
        {
            "question": RunnablePassthrough(),
            "context": search_pdf_internal
        }
        | pdf_prompt
        | llm
    )

    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else response


# ========================
# MERGED TOOLS
# ========================

@tool
def get_laptop_status(question: str) -> str:
    """Retrieve student laptop allocation and delivery status from the database.

    Use this tool for:
    - Laptop allocation status for a specific student
    - Laptop delivery status
    - Student laptop details and timeline
    - Any query related to individual student laptop information

    Args:
        question: Query about laptop status (e.g., "What's the status for student ID STUVEL007?"
                 or "Find Vel Geetha's laptop details")

    Returns:
        Student laptop information with allocation and delivery status
    """
    return execute_sql_query(question)


@tool
def chat_with_pdf(question: str) -> str:
    """Chat with PDF to answer questions about government schemes, eligibility, and benefits.

    Use this tool for:
    - Scheme details and overview
    - Eligibility criteria
    - Benefits and specifications
    - Required documents
    - Distribution timeline
    - Policy information
    - General scheme questions

    Do NOT use for student laptop allocation or delivery status tracking.

    Args:
        question: Question about government schemes or general information from the document

    Returns:
        Answer based on the uploaded PDF document
    """
    return ask_pdf_rag(question)


# ========================
# List all tools
# ========================
tools = [get_laptop_status, chat_with_pdf]


# ========================
# STEP 1: BIND TOOLS WITH LLM
# ========================
llm_with_tools = llm.bind_tools(tools)

print("✓ Tools bound to LLM")


# ========================
# STEP 2: INITIALIZE STATE
# ========================
class State(TypedDict):
    """Define the agent state with message history"""
    messages: Annotated[list, add_messages]


# ========================
# STEP 3: CREATE TOOL EXECUTION NODE
# ========================
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=tools)


# ========================
# STEP 4: CREATE CHATBOT NODE
# ========================
def chatbot(state: State):
    """LLM node that processes messages and decides on tools"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ========================
# STEP 5: CREATE ROUTING FUNCTION
# ========================
def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Route to the tool node if the last message has tool calls.
    Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


# ========================
# STEP 6: BUILD THE GRAPH
# ========================
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "chatbot")

# Conditional routing
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "__end__": "__end__"},
)

# Tool node loops back to chatbot
graph_builder.add_edge("tools", "chatbot")


# ========================
# STEP 7: COMPILE GRAPH WITH MEMORY
# ========================
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

print("✓ Agent graph compiled successfully")


# ========================
# STEP 8: OPTIONAL - VISUALIZE GRAPH
# ========================
try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
    print("✓ Graph visualization generated")
except Exception as e:
    print(f"⚠ Graph visualization skipped: {e}")


# ========================
# STEP 9: RUN THE AGENT
# ========================
def run_agent(user_input: str, thread_id: str = "default"):
    """
    Run the agent with user input

    Args:
        user_input: User's question
        thread_id: Conversation thread ID for persistence
    """
    print(f"\n{'='*60}")
    print(f"User: {user_input}")
    print(f"{'='*60}")

    config = {"configurable": {"thread_id": thread_id}}

    # Stream the agent execution
    for step in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    ):
        if "messages" in step:
            last_message = step["messages"][-1]

            if isinstance(last_message, ToolMessage):
                print(f"\n[Tool Result] {last_message.name}:")
                print(f"  {last_message.content[:200]}...")  # Print first 200 chars

            elif isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    print(f"\n[LLM Decision] Calling tools:")
                    for tool_call in last_message.tool_calls:
                        print(f"  - {tool_call['name']}: {tool_call['args']}")
                else:
                    print(f"\n[Final Answer]:\n{last_message.content}")


# ========================
# MAIN - INTERACTIVE MODE
# ========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 ELCOT STUDENT LAPTOP AGENT")
    print("="*60)
    print("\nAvailable Tools:")
    print("  1. get_laptop_status - Query student laptop status from database")
    print("  2. chat_with_pdf - Get information from government scheme documents")
    print("\nType 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        if not user_input:
            print("Please enter a valid question.\n")
            continue

        # Run the agent
        run_agent(user_input)