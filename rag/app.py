from flask import Flask, request, jsonify, render_template
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from opencc import OpenCC
import openai

# 設定 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"})

    # 使用 OpenAI 進行問答
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./rag/db/temp/", embedding_function=embeddings)
    docs = db.similarity_search(question)
    
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.5
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    
    with get_openai_callback() as cb:
        response = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)
    
    cc = OpenCC('s2t')
    answer = cc.convert(response['output_text'])

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
