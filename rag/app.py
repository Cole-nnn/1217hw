from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data.get("question", "")
    # 假設簡單回應邏輯
    response = f"這是對於問題「{question}」的回答。"
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
