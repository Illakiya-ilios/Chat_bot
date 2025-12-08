import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ========================
# Load environment variables
# ========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "elcotlaptopscheme")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

# PostgreSQL connection URI
uri = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize DB connection
db = SQLDatabase.from_uri(uri, include_tables=["students"])

# Initialize LLM
llm = ChatBedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")

# ========================
# SQL Query Generation Prompt
# ========================
sql_prompt = PromptTemplate.from_template(
    """You are an expert SQL assistant who converts user questions into SQL queries
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
# SQL Execution Tool
# ========================
execute_query = QuerySQLDatabaseTool(db=db)

# ========================
# Answer Generation Prompt (Modified)
# ========================
answer_prompt = PromptTemplate.from_template(
    """You are a helpful assistant.

Your task is to take the SQL result and answer the user's question clearly.
Never mention SQL or database terms.

Rules:
- If the SQL result is empty or no matching record is found, respond politely with:
  "Sorry, I couldn't find any record with the provided details. Could you please share your registered phone number or student ID so I can check again?"

- If the model earlier asked for more details, repeat that same message politely.

- If valid data is found, respond in a clear and natural tone, for example:
  "Vel Geetha has been allotted a laptop and the delivery status is 'Delivered'."

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

# ========================
# LangChain Pipeline
# ========================
chain = (
    RunnablePassthrough.assign(
        schema=lambda _: db.get_table_info(),
        query=lambda x: generate_query.invoke(
            {"question": x["question"], "schema": db.get_table_info()}
        )
    ).assign(
        result=lambda x: (
            "Please provide more identifying details such as your student ID, registered mobile number, or full name so that I can retrieve your laptop allocation status."
            if "Please provide" in x["query"]
            else execute_query.invoke(x["query"])
        )
    )
    | rephrase_answer
)

# ========================
# Flask App Setup
# ========================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Missing 'question' field"}), 400

        response = chain.invoke({"question": question})
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# Run Flask App
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)