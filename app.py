import nltk
import methods

from flask import Flask, app, request, jsonify
from flask_cors import CORS


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


app = Flask(__name__)

cors = CORS(app)


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add(
        "Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept"
    )
    response.headers.add("Access-Control-Allow-Methods",
                         "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


@app.route("/recommend-movies", methods=["POST"])
def recommend_movies():
    if request.method == "POST":
        try:
            form_values = request.form.to_dict()
            print(form_values)
            movie_name = str(form_values["movie_name"]).strip().lower()
            n_recommendation = int(str(form_values["number_of_recommendations"]))
            return methods.movie_recommender_engine(movie_name, n_recommendation)
        except:
            data = {
                "error": "Invalid Data"
            }
            return jsonify(data)


@app.route("/movie-reviews-sentiment", methods=["POST"])
def movie_reviews_sentiment():
    if request.method == "POST":
        try:
            form_values = request.form.to_dict()
            movie_imdb_id = str(form_values["movie_imdb_id"])
            reviews_data = methods.movie_reviews_with_sentiment(movie_imdb_id)
            return jsonify(reviews_data)
        except:
            data = {
                "error": "Invalid IMDB Id"
            }
            return jsonify(data)


@app.route("/", methods=["GET"])
def home():
    if request.method == "GET":
        data = {
            "result": "This is Movie Recommender API"
        }
        return jsonify(data)