import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template

load_dotenv()

app = Flask(__name__)

API_URL = os.getenv("BIOX_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("BIOX_API_KEY")


@app.route("/health")
def health():
    return {"status": "ok", "app": "flask-front"}


@app.route("/", methods=["GET", "POST"])
def index():
    recos = None
    error = None
    form_data = {
        "references": "",
        "top_k": "1"
    }

    if request.method == "POST":
        form_data["references"] = request.form.get("references", "").strip()
        form_data["top_k"] = request.form.get("top_k", "1").strip()

        references = [x.strip() for x in form_data["references"].split("\n") if x.strip()]

        payload = {
            "references": references,
            "top_k": int(form_data["top_k"]),
        }

        try:
            r = requests.post(
                f"{API_URL}/recommendations/by-reference",
                json=payload,
                headers={"X-API-Key": API_KEY},
                timeout=60,
            )

            if r.status_code == 200:
                recos = r.json()
            else:
                error = f"Erreur API {r.status_code} : {r.text}"

        except Exception as e:
            error = f"Erreur de connexion à l'API : {e}"

    return render_template("index.html", recos=recos, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)