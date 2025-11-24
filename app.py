import os
import pickle
import requests
import streamlit as st
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import gdown


load_dotenv()

TMDB_API_KEY = None
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found. Add it to .env or to Streamlit Secrets and restart.")
    st.stop()

TMDB_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Image"

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"])
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)


def fetch_movie_details(movie_id):
    try:
        url = f"{TMDB_BASE}/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        resp = session.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        poster = IMAGE_BASE + poster_path if poster_path else PLACEHOLDER_IMAGE
        overview = data.get("overview", "")
        release_date = data.get("release_date", "")
        return poster, overview, release_date
    except requests.exceptions.RequestException as e:
        print(f"[fetch_movie_details] movie_id={movie_id} failed: {e}")
        return PLACEHOLDER_IMAGE, "", ""

expected_files = {"movies_dict.pkl", "similarity.pkl"}
missing = [f for f in expected_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Place them in project root.")
    st.stop()

with open("movies_dict.pkl", "rb") as f:
    movies_dict = pickle.load(f)
movies = pd.DataFrame(movies_dict)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown(
    """
    <style>
    .movie-title { font-size:16px; font-weight:700; text-align:center; margin-bottom:6px; }
    .release { font-size:12px; color: #9aa0a6; text-align:center; margin-bottom:8px; }
    .container { padding: 6px 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Movie Recommendation System")
st.caption("Select a movie and get 5 recommendations with posters and summaries.")

selected_movie_name = st.selectbox("Select movie to recommend", movies["title"].values)

def recommend(movie, k=5):
    matched = movies[movies["title"] == movie]
    if matched.empty:
        return [], [], [], []
    movie_index = matched.index[0]
    distances = similarity[movie_index]
    topk = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : k + 1]

    rec_titles = []
    rec_posters = []
    rec_overviews = []
    rec_dates = []
    for idx, _score in topk:
        movie_id = movies.iloc[idx].movie_id
        rec_titles.append(movies.iloc[idx].title)
        poster, overview, release_date = fetch_movie_details(movie_id)
        rec_posters.append(poster)
        rec_overviews.append(overview)
        rec_dates.append(release_date)
    return rec_titles, rec_posters, rec_overviews, rec_dates


if st.button("Recommend"):
    names, posters, overviews, dates = recommend(selected_movie_name)
    if not names:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(len(names))
        for i, col in enumerate(cols):
            with col:

                st.markdown(
                    f"<div class='movie-title'>{names[i]}</div><div class='release'>{dates[i]}</div>",
                    unsafe_allow_html=True,
                )

                st.image(posters[i], use_container_width=True)
                with st.expander("Summary"):
                    if overviews[i]:
                        st.write(overviews[i])
                    else:
                        st.write("No summary available.")
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

with st.expander("About This App"):
    st.write("""
    This movie recommendation system was designed and developed by **Varsh Vishwakarma**.
    Powered by TMDB API & Machine Learning similarity models.
    """)

st.markdown("---")
st.caption("Tip: to keep secrets safe, add TMDB_API_KEY to Streamlit Secrets or your local .env (do not commit it).")
