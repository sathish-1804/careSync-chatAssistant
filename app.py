from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_sql_query_chain
import dotenv

app = Flask(__name__)
CORS(app)
dotenv.load_dotenv()

# Set API Key
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Initialize LLM and Database connection
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
db = SQLDatabase.from_uri(f"mysql+mysqlconnector://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}@{os.environ.get('HOST_NAME')}/{os.environ.get('DB_NAME')}")

# Create SQL query chain
chain = create_sql_query_chain(llm, db)

@app.route('/process_context', methods=['POST'])
def process_context():
    data = request.get_json()
    question = data.get("question")
    userId = data.get("userId")

    if not question:
        return jsonify({"error": "No context provided"}), 400

    db_schema = db.get_table_info()
    
    # Build the prompt
    prompt = f"""
        You are an expert in converting English questions to SQL query!
        The SQL database has tables, and these are the schemas: {db_schema}. 
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        The sql code should not have ``` in beginning or end and sql word in output.
        You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        If the question does not seem related to the database, just return "null" as the answer.
        
        Now I want you to generate the structured query (in single line ending with semi-colon) for below question: {question} for the specified user id: {userId}.
    """

    # Get the response from the chain
    response = chain.invoke({"question": prompt})

    # Extract the SQL query from the response using regex
    sql_query_match = re.search(r'SELECT.*?;', response, re.DOTALL)
    if sql_query_match:
        sql_query = sql_query_match.group(0)
        sql_response = f"Generated SQL Query: {sql_query}"

        # Run the extracted SQL query on the database
        result = db.run(sql_query)
        sql_response += f"\nQuery Result: {result}"
        
        # Properly invoke the chain for the answer generation step
        answer = chain.invoke({
            "question": f"""
            Based on the sql response, write an intuitive answer for the user question, it should be short and crisp. :
                User Question: {question},
                sql_response: {sql_response}
                if could not find the answer, return a helpful and relevant answer to the user's question.
            """
        })
        match = re.search(r'Answer:\s*(.*)', answer, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()  # Extracts the text after "Answer:"
            return jsonify({"answer": final_answer})
        else:
            return jsonify({"error": "Try again later"}), 500
    else:
        # If the SQL query is not found, answer the user's question directly using the LLM
        fallback_answer = chain.invoke({
            "question": f"""
            Since a SQL query could not be generated, provide a helpful and relevant answer to the user's question, it should be super short and crisp. :
                User Question: {question}
            """
        })
        
        # Extract the final answer
        fallback_match = re.search(r'Answer:\s*(.*)', fallback_answer, re.DOTALL)
        if fallback_match:
            final_answer = fallback_match.group(1).strip()  # Extracts the text after "Answer:"
            return jsonify({"answer": final_answer})
        else:
            return jsonify({"error": "Try again later"}), 500

if __name__ == '__main__':
    app.run(debug=True)
