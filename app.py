
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    normal_count = attack_count = normal_pct = attack_pct = None

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            error = "Please upload a valid CSV file."
            return render_template("index.html", error=error)
        try:
            df = pd.read_csv(file)
            df = df.select_dtypes(include=["number"])
            predictions = model.predict(df)
            normal_count = (predictions == "Normal").sum()
            attack_count = (predictions == "Attack").sum()
            total = normal_count + attack_count
            normal_pct = round((normal_count / total) * 100, 1)
            attack_pct = round((attack_count / total) * 100, 1)
            plt.figure(figsize=(4, 4))
            plt.pie([normal_count, attack_count], labels=["Normal", "Attack"], autopct='%1.1f%%')
            plt.title("Traffic Classification")
            plt.savefig("static/chart.png")
            plt.close()
        except Exception as e:
            error = f"Error processing file: {str(e)}"
    return render_template("index.html", error=error, normal_count=normal_count, attack_count=attack_count, normal_pct=normal_pct, attack_pct=attack_pct)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
