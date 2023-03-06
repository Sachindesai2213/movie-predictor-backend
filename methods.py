import difflib
import json
import pickle
import re
import urllib

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmdbv3api import Movie, TMDb

# MOVIE RECOMMENDATION MODEL
data = pd.read_csv("./pickle-files/final_data.csv")
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(data["movie_feature"])
similarity_matrix = cosine_similarity(count_matrix)
tmdb = TMDb()
TMDB_API_KEY = "a288cadf42347603c0549b9838a981df"
tmdb.api_key = TMDB_API_KEY

CLASSIFIER_PATH = "./pickle-files/sentiment_classifier.pkl"
TF_IDF_PATH = "./pickle-files/tf_idf.pkl"

sentiment_classifier = pickle.load(open(CLASSIFIER_PATH, "rb"))
tf_idf = pickle.load(open(TF_IDF_PATH, "rb"))


lemmatizer = WordNetLemmatizer()


def convert(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError


def preprocess_review(review):
    soup = BeautifulSoup(review, "html.parser")
    review = soup.get_text()
    review = re.sub("\[[^]]*\]", " ", review)
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    review = [
        word.strip() for word in review if word not in set(stopwords.words("english"))
    ]
    review = [lemmatizer.lemmatize(word) for word in review]
    return " ".join(review)


def movie_search_engine(movie_name):
    tmdb_movie = Movie()
    search_result = tmdb_movie.search(movie_name.strip())
    movie_id = search_result[0]["id"]
    movie = tmdb_movie.details(movie_id)
    movie_data = {
        'movie_id': movie.id,
        'title': movie.title,
        'poster_path': movie.poster_path
    }
    return movie_data


def movie_reviews_with_sentiment(movie_imdb_id):
    sauce = urllib.request.urlopen(
        "https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt".format(
            movie_imdb_id
        )
    ).read()
    soup = BeautifulSoup(sauce, "lxml")
    soup_result = soup.find_all(
        "div", {"class": "text show-more__control"})
    movie_reviews = []
    processed_reviews = []
    count = 0

    for reviews in soup_result:
        review = reviews.text
        if review is not None:
            movie_reviews.append(review)
            processed_reviews.append(preprocess_review(review))
            if count == 10:
                break
            count += 1

    processed_reviews = tf_idf.transform(processed_reviews)
    predictions = sentiment_classifier.predict(processed_reviews)
    labels = ["BAD", "GOOD"]

    all_reviews_data = []
    index = 1
    for review, sentiment in zip(movie_reviews, predictions):
        all_reviews_data.append(
            {
                "id": index,
                "content": review,
                "sentiment": labels[sentiment]
            }
        )
        index += 1
    return all_reviews_data


def movie_recommender_engine(movie_name, n_top_recommendations=10):
    movie_name = movie_name.strip().lower()
    if movie_name not in data["movie_title"].unique():
        temp = difflib.get_close_matches(
            movie_name, list(data.movie_title.unique()))
        if temp != []:
            word_similarity = get_word_similarity(movie_name, temp[0])
            if word_similarity > 75:
                movie_name = temp[0]
            else:
                return json.dumps(
                    {
                        "error": "Sorry! Movie is not in our database. Please check the spelling or try with another movie name"
                    },
                    default=convert,
                )
        else:
            return json.dumps(
                {
                    "error": "Sorry! Movie is not in our database. Please check the spelling or try with another movie name"
                },
                default=convert,
            )
    index = data.loc[data["movie_title"] == movie_name].index[0]
    matrix = list(enumerate(similarity_matrix[index]))
    matrix = sorted(matrix, key=lambda x: x[1], reverse=True)
    recommended_indexes = [
        index for (index, similarity) in matrix[0: n_top_recommendations + 5]
    ]
    recommended_movies = {"recommendations": []}
    rank = 1
    r_count = 1
    i = 0
    while i < len(recommended_indexes) and r_count <= n_top_recommendations:
        index = recommended_indexes[i]
        r_movie_name = data["movie_title"][index]
        try:
            movie_data = movie_search_engine(r_movie_name)
            if movie_name == r_movie_name:
                recommended_movies["input_movie"] = movie_data
            else:
                movie_data["rank"] = rank
                recommended_movies["recommendations"].append(movie_data)
                rank += 1
                r_count += 1
        except:
            pass
        i += 1
    recommended_movies = jsonify(recommended_movies)
    return recommended_movies