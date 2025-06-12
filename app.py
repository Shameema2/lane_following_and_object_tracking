from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    # Add your code to start object tracking here
    return 'Tracking started'

if __name__ == '__main__':
    app.run(host='0.0.0.0')
