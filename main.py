import net
from flask import Flask, redirect, request, json, render_template

#flask basic code 
app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def dashboard():
    print(net.called)
    return "hsllo"
if __name__ == "__main__":
    app.run()
