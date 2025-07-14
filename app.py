# # app.py

# from flask import Flask, request, render_template
# import joblib

# app = Flask(__name__)

# # Load entire pipeline
# pipeline = joblib.load("model/spam_pipeline.pkl")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     message = request.form["message"]
#     prediction = pipeline.predict([message])[0]
#     result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
#     return render_template("index.html", prediction=result)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template
from train_model import trained_vectorizer, trained_model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    message_vec = trained_vectorizer.transform([message])
    prediction = trained_model.predict(message_vec)[0]
    result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)