
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio
import docx
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import certifi
import ssl
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Retrieve environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if environment variables are loaded correctly
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set correctly")

# Define the directory to save uploaded files
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

async def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

async def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    return df.to_string(index=False)

def extract_text_from_url(url):
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=context) as response:
        html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def extract_text_from_document(doc):
    if isinstance(doc, str):
        return [doc]  # If the entire document is text, return it as a list
    elif isinstance(doc, dict):
        text_data = []
        for key, value in doc.items():
            if isinstance(value, str):
                text_data.append(value)  # If the value is text, append it to the list
            elif isinstance(value, (dict, list)):
                text_data.extend(extract_text_from_document(value))  # Recursively search nested dictionaries or lists
        return text_data
    elif isinstance(doc, list):
        text_data = []
        for item in doc:
            text_data.extend(extract_text_from_document(item))  # Recursively search each item in the list
        return text_data
    else:
        return []

def get_data_from_mongodb(connection_string, database_name, collection_name):
    # Create a custom SSL context and disable certificate verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    client = MongoClient(connection_string, tlsInsecure=True)
    db = client[database_name]
    collection = db[collection_name]

    # Retrieve documents from the collection
    documents = collection.find()

    # Extract text data from documents dynamically
    data = []
    for doc in documents:
        text_data = extract_text_from_document(doc)
        data.extend(text_data)

    client.close()
    return data



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_question", methods=["POST"])
def process_question():
    if request.method == 'POST':
        data = request.json
        user_question = data.get('question')
        if user_question:
            response = asyncio.run(user_input(user_question))
            return jsonify({"response": response})
        else:
            return jsonify({"error": "No question provided"}), 400
    else:
        return jsonify({"error": "Only POST requests are supported"}), 405

@app.route("/upload", methods=["POST"])
def upload():
    print(request.files)  # Debug: print request.files to see the uploaded files
    if len(request.files) == 0:
        print("No file part in the request")
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = None
    for key in request.files:
        uploaded_file = request.files[key]
        break  # Get the first file found

    if uploaded_file is None or uploaded_file.filename == "":
        print("No selected file")
        return jsonify({"error": "No file selected"}), 400

    print(f"File received: {uploaded_file.filename}")

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    uploaded_file.save(file_path)

    file_extension = uploaded_file.filename.split('.')[-1].lower()
    text = ""

    if file_extension == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension == "docx":
        text = extract_text_from_docx(file_path)
    elif file_extension == "csv":
        text = extract_text_from_csv(file_path)
    elif file_extension == "txt":
        text = extract_text_from_txt(file_path)
    elif file_extension in ["xls", "xlsx"]:
        text = extract_text_from_excel(file_path)
    else:
        print("Unsupported file type")
        return jsonify({"error": "Unsupported file type"}), 400

    text_chunks = get_text_chunks(text)
    asyncio.run(get_vector_store(text_chunks))

    return jsonify({"message": "File uploaded and processed successfully"}), 200

@app.route("/upload_url", methods=["POST"])
def upload_url():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        text = extract_text_from_url(url)
    except Exception as e:
        return jsonify({"error": f"Failed to extract text from URL: {str(e)}"}), 400

    text_chunks = get_text_chunks(text)
    asyncio.run(get_vector_store(text_chunks))

    return jsonify({"message": "URL content processed successfully"}), 200


@app.route("/process_mongodb_data", methods=["POST"])
def process_mongodb_data():
    data = request.json
    connection_string = data.get('connection_string')
    database_name = data.get('database_name')
    collection_name = data.get('collection_name')

    if not connection_string or not database_name or not collection_name:
        return jsonify({"error": "Incomplete information provided"}), 400

    try:
        mongodb_data = get_data_from_mongodb(connection_string, database_name, collection_name)
        if mongodb_data:
            text = '\n'.join(mongodb_data)
            text_chunks = get_text_chunks(text)
            asyncio.run(get_vector_store(text_chunks))
            return jsonify({"message": "MongoDB data processed successfully"}), 200
        else:
            return jsonify({"error": "No data found in MongoDB collection"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve and process MongoDB data: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(port=5001,debug=True)



