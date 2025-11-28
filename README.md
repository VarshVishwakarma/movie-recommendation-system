🎬 Movie Recommendation System

A smart, dynamic, ML-powered movie recommender built with Streamlit

This project delivers a seamless recommendation experience by blending machine learning similarity models with The Movie Database (TMDB) API for real-time posters, summaries, and release details. Lightweight, fast, and designed for production-grade deployment.

🚀 Key Features
🔍 Intelligent Movie Recommendations

Select any movie from the trained dataset.

System returns top 5 similar movies using cosine similarity.

Uses TMDB API to fetch:

Posters

Summaries

Release dates

📦 Local + Cloud Hybrid Model Storage

movies_dict.pkl stored locally in repo.

similarity.pkl auto-downloaded from Google Drive using gdown.

Built-in retry logic and graceful error handling.

🎨 Modern UI with Streamlit

Clean, responsive layout.

Hover-friendly card design.

Expandable movie summaries.

Error-proof experience with fallback images and missing-data handling.

🔒 Secure API Handling

TMDB API key loaded via:

Streamlit Secrets

OR local .env

🧠 How It Works (Under the Hood)

Similarity Model
The ML model computes movie similarity using vectorized feature representations.

Runtime Download (Google Drive)
If similarity.pkl is missing, the app fetches it via Google Drive’s direct-download ID.

Recommendation Pipeline

Identify selected movie

Compute similarity distances

Select top-K matches (excluding the movie itself)

Fetch metadata from TMDB

Render results with posters + summaries

Resilient HTTP Requests
Adaptive retry mechanism:

Handles rate limits

429/500/502/503/504 auto-retry

6-second timeout protection

📁 Project Structure
📂 Movie-Recommender
│── app.py               # Main Streamlit application
│── movies_dict.pkl      # Local dataset of movies
│── similarity.pkl       # Downloaded ML similarity matrix
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── .env (optional)       # Local TMDB API key

🔧 Installation & Setup
1. Clone Repository
git clone <your-repo-url>
cd Movie-Recommender

2. Install Dependencies
pip install -r requirements.txt

3. Add TMDB API Key

Option A — Streamlit Secrets

[general]
TMDB_API_KEY = "your_api_key_here"


Option B — .env file

TMDB_API_KEY=your_api_key_here

4. Run App
streamlit run app.py


App launches instantly in your browser.

🌐 External Services Used
TMDB API

Used for metadata and media content.
Visit: https://www.themoviedb.org/

Google Drive (gdown)

Used to host and download similarity.pkl.

🏗 Dependencies
streamlit
pandas
numpy
requests
gdown
python-dotenv
urllib3

🖼 UI Experience

The interface provides:

Smooth recommendation flow

Column-based film display

Expandable summaries

Automatic placeholder images

Graceful error recovery

🧑‍💻 Author

Varsh Vishwakarma
AI • ML • DL • Data Science • Cloud • Full-Stack ML Developer
