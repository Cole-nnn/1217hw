from flask import Flask, render_template, request, jsonify
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI  # 更新的導入語句
from opencc import OpenCC
import openai

# 使用環境變數來存儲 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# 假設這是你的聊天歷史記錄
chat_history = []

# 設置首頁路由，返回一個 HTML 頁面
@app.route('/')
def home():
    return render_template('index.html')  # 確保你的模板文件夾下有 index.html

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.get_json()["question"]
    if not user_input:
        return jsonify({'error': 'No user input provided'})

    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./db/temp/", embedding_function=embeddings)
    docs = db.similarity_search(user_input)
    
    llm = ChatOpenAI(
        model_name="gpt-4",  # 確保使用的是正確的模型名稱
        temperature=0.5
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    
    with get_openai_callback() as cb:
        response = chain.invoke({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    
    cc = OpenCC('s2t')
    answer = cc.convert(response['output_text'])

    # 這裡可以將聊天歷史記錄儲存到 session 或其他地方
    chat_history.append({'user': user_input, 'assistant': answer})

    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(debug=True)
