from flask import Flask, render_template, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from opencc import OpenCC
import os

app = Flask(__name__)

# 初始化聊天記錄
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')
    
# 問答處理函式
@app.route('/get_response', methods=['POST'])
def get_response():
    # 取得用戶輸入
    data = request.json  # 解析 JSON 格式數據
    user_input = data.get('question')  # 獲取 'question' 欄位
    if not user_input:
        return jsonify({'error': 'No user input provided'})

    try:
        # 初始化嵌入向量與 Chroma 資料庫
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="./db/temp/", embedding_function=embeddings)

        # 相似性檢索
        docs = db.similarity_search(user_input)

        # 初始化 LLM 與問答鏈
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        chain = load_qa_chain(llm, chain_type="stuff")

        # 問答處理
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_input)

        # 繁簡轉換
        cc = OpenCC('s2t')
        answer = cc.convert(response)

        # 儲存聊天記錄
        chat_history.append({'user': user_input, 'assistant': answer})

        return jsonify({'response': answer})

    except Exception as e:
        return jsonify({'error': str(e)})

# 啟動 Flask 應用
if __name__ == '__main__':
    app.run(debug=True)
