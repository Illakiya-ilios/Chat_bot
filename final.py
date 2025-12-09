import os
import boto3
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain.tools import tool


# ========================
# Load environment
# ========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "elcotlaptopscheme")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# ========================
# AWS Bedrock Client
# ========================
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

# ========================
# Database Connection
# ========================
uri = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(uri, include_tables=["students"])

# Create the SQL execution tool
sql_tool = QuerySQLDatabaseTool(db=db)


# ========================
# TOOL for executing queries
# ========================
@tool
def run_sql(query: str):
    """Use this tool ONLY to retrieve student laptop status from the SQL database.
    This includes checking whether a laptop is allocated, delivered, or received.
    Do NOT use this tool for government scheme details, policies, or general questions."""
    return sql_tool.invoke(query)


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
# Answer Generation Prompt
# ========================
answer_prompt = PromptTemplate.from_template(
"""You are a student services support assistant. Your job is to read the SQL result that has already been fetched and convert it into a clear, natural answer for the user.

Always base your answer strictly on the information inside the SQL result.  
Do not mention SQL, queries, databases, or any technical process.

When responding:
1) Be polite, helpful, and factual.
2) Keep the answer short and clear (2–4 sentences).
3) Use the student's name and details exactly as found in the result.
4) If the SQL result is empty or no matching record is found, say: “No record found for the given details.”
5) If some fields are missing, respond naturally with: “I don’t have that information available right now.”
6) Your answer should sound like you are assisting a student or parent.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
"""
)

format_answer = answer_prompt | llm | StrOutputParser()


# ========================
# MAIN CHAIN
# ========================
chain = (
    RunnablePassthrough()
    .assign(
        schema=lambda x: db.get_table_info()
    )
    .assign(
        query=lambda x: generate_query.invoke(
            {"question": x["question"], "schema": x["schema"]}
        )
    )
    .assign(
        result=lambda x: (
            "Please provide more identifying details such as your student ID, registered mobile number, or full name so that I can retrieve your laptop allocation status."
            if "Please provide" in x["query"]
            else run_sql.invoke(x["query"])
        )
    )
    | format_answer
)



# ========================
# Run the Bot
# ========================
if __name__ == "__main__":
    print("\n SQL DB Chatbot Ready!\n")
    while True:
        q = input("Ask: ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", chain.invoke({"question": q}), "\n")