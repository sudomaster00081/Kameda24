from flask import Flask, render_template, request, jsonify

department = ''
date =''
doctor =''
time=''


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if True:
        input = msg
        return get_Chat_response(input)


# Updated get_Chat_response function
def get_Chat_response(text):
    # Sample response with chat and options
    response = {
        'chat': 'Hey, how can I help you?',
        'options': ['Option 1', 'Option 2', 'Option 3']
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run()
