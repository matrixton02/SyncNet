from flask import Flask, jsonify, render_template
import server

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/progress")
def progress():
    return jsonify(server.client_updates)

if __name__ == "__main__":
    app.run(port=8000)
