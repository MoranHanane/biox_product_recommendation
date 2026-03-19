import os
import requests
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, session, flash

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
BIOX_API_KEY = os.getenv("BIOX_API_KEY")

API_URL = os.getenv("BIOX_API_URL", "http://127.0.0.1:8000")

if not app.secret_key:
    raise ValueError("FLASK_SECRET_KEY manquante dans le .env")

if not BIOX_API_KEY:
    raise ValueError("BIOX_API_KEY manquante dans le .env")

APP_USERS = {
    os.getenv("BIOX_ADMIN_USER", "admin"): {
        "password": os.getenv("BIOX_ADMIN_PASSWORD", "admin123"),
        "role": "admin",
    },
    os.getenv("BIOX_COMMERCIAL_USER", "commercial"): {
        "password": os.getenv("BIOX_COMMERCIAL_PASSWORD", "commercial123"),
        "role": "commercial",
    },
}

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

def role_required(*allowed_roles):
    """
    Décorateur de contrôle d’accès basé sur le rôle utilisateur.

    Ce décorateur vérifie que l’utilisateur authentifié possède
    le rôle requis pour accéder à une route donnée.

    Paramètre :
    - required_role (str) : rôle attendu (ex: "admin", "commercial")

    Comportement :
    - Si l’utilisateur n’est pas connecté → redirection login
    - Si le rôle ne correspond pas → accès refusé (403 ou redirection)

    Utilisation :
    @app.route("/")
    @role_required("commercial")

    Retour :
    - Fonction décorée si accès autorisé
    - Redirection ou erreur sinon
    """
    def decorator(view_func):
        """
        Enveloppe une fonction Flask avec une logique de contrôle d’accès.

        Ce décorateur interne est utilisé par role_required afin de centraliser
        la vérification des droits utilisateur avant exécution de la route.

        Paramètre :
        - func : fonction Flask décorée

        Retour :
        - wrapper : fonction enrichie avec contrôle d’accès
        """
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            """
            Fonction intermédiaire exécutée avant la route protégée.

            Elle vérifie la présence d’une session utilisateur valide et
            la conformité du rôle avant d’autoriser l’exécution de la route.

            Paramètres :
            - *args, **kwargs : arguments de la route Flask

            Retour :
            - Exécution de la route si autorisée
            - Redirection vers login ou erreur sinon
            """
            if "role" not in session:
                return redirect(url_for("login"))
            if session["role"] not in allowed_roles:
                return "Accès refusé", 403
            return view_func(*args, **kwargs)
        return wrapper
    return decorator

@app.after_request
def add_security_headers(response):
    """
    Ajoute des en-têtes HTTP de sécurité à chaque réponse Flask.

    Cette fonction est appelée automatiquement après chaque requête
    afin de renforcer la sécurité de l’application (OWASP Top 10).

    Headers ajoutés :
    - X-Content-Type-Options : empêche le MIME sniffing (force le navigateur à respecter strictement le Content-Type déclaré, sans lui laisser la possibilité de "deviner" le Content-Type en lisant les premiers octets du fichier)
    - X-Frame-Options : protège contre le clickjacking ( interdit totalement l'intégration de la page dans une iframe, quelle que soit l'origine)
    - X-XSS-Protection : active les protections XSS du navigateur (bloque l'injection de JavaScript malveillant)
    - Strict-Transport-Security : impose HTTPS (si activé)

    Paramètre :
    - response (Flask Response) : réponse HTTP à enrichir

    Retour :
    - response modifiée avec headers de sécurité
    """    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Gère l’authentification d’un utilisateur via formulaire HTML.

    Cette route permet de créer une session Flask contenant les informations
    minimales d’identification (username, rôle). Elle est utilisée comme point
    d’entrée obligatoire avant l’accès aux routes protégées.

    Comportement :
    - GET : affiche le formulaire de connexion (template login.html)
    - POST : vérifie les identifiants fournis et initialise la session utilisateur

    Variables de session définies :
    - session["username"] : identifiant utilisateur
    - session["role"] : rôle associé (ex: "commercial", "admin")

    Retour :
    - Redirection vers "/" si succès
    - Réaffichage du formulaire sinon
    """
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = APP_USERS.get(username)
        if user and password == user["password"]:
            session["username"] = username
            session["role"] = user["role"]
            return redirect(url_for("index"))

        error = "Identifiants invalides."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    """
    Termine la session utilisateur active.

    Cette route supprime les informations de session afin de révoquer
    l’accès aux routes protégées de l’application Flask.

    Effets :
    - Suppression des clés "username" et "role" de la session

    Retour :
    - Redirection vers la page de login
    """    
    session.clear()
    return redirect(url_for("login"))

@app.route("/health") 
@login_required
def health():
    """
    Endpoint de vérification de disponibilité du frontend Flask.

    Retour :
    - statut simple ("ok") pour monitoring ou tests.
    """   
    return {"status": "ok", "app": "flask-front", "user": session.get("username"), "role": session.get("role")}

@app.route("/", methods=["GET", "POST"])
@login_required
@role_required("admin", "commercial")
def index():
    """
    Point d’entrée principal du frontend Flask.

    - Récupère les références produits saisies par l’utilisateur (textarea).
    - Construit une requête POST vers l’API FastAPI (/recommendations/by-reference).
    - Gère les erreurs :
        * erreur API (code HTTP ≠ 200)
        * erreur de connexion (timeout, API indisponible)
    - Retourne un rendu HTML avec :
        * les recommandations (recos)
        * un message d’erreur éventuel
        * les données du formulaire (pour persistance UI)

    Sert d’interface simple entre utilisateur métier et moteur de recommandation.
    """
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

        if not references:
            error = "Veuillez saisir au moins une référence produit."
            return render_template("index.html", recos=recos, error=error, form_data=form_data)

        try:
            top_k = int(form_data["top_k"])
        except ValueError:
            error = "Le champ top_k doit être un entier."
            return render_template("index.html", recos=recos, error=error, form_data=form_data)

        if top_k < 1 or top_k > 5:
            error = "Le champ top_k doit être compris entre 1 et 5."
            return render_template("index.html", recos=recos, error=error, form_data=form_data)

        payload = {
            "references": references,
            "top_k": top_k,
        }

        try:
            r = requests.post(
                f"{API_URL}/recommendations/by-reference",
                json=payload,
                headers={"X-API-Key": BIOX_API_KEY},
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