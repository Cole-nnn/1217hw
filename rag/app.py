from flask import Flask, render_template, request, jsonify
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from opencc import OpenCC
import openai

# 設定 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# 聊天歷史記錄
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        # 接收用戶問題
        user_input = request.json.get('question')
        if not user_input:
            return jsonify({'error': 'No user input provided'})

        # 日誌記錄
        print("User Input:", user_input)

        # 加載嵌入和資料庫
        embeddings = OpenAIEmbeddings()
        db_path = "./rag/db/temp/"
        if not os.path.exists(db_path):
            return jsonify({'error': 'Database path not found.'})
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # 搜索相似文檔
        docs = db.similarity_search(user_input)
        print("Docs Retrieved:", len(docs))
        if not docs:
            return jsonify({'error': 'No relevant documents found for the question.'})

        # 使用 LLM 生成答案
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.invoke({"input_documents": docs, "question": user_input})
        if 'output_text' not in response:
            return jsonify({'error': 'Failed to generate a response.'})

        # 繁簡轉換
        cc = OpenCC('s2t')
        answer = cc.convert(response['output_text'])

        # 更新聊天記錄
        chat_history.append({'user': user_input, 'assistant': answer})
        print("Generated Answer:", answer)

        return jsonify({'response': answer})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == "__main__":
    app.run(debug=True)
