# Music Recommendation System

A content-based music recommendation system that suggests similar songs based on **audio features, vibe/mood, and genre**.  
The project supports both **global** and **regional (Indian)** music datasets and is deployed as an interactive **Streamlit** web application.

---

Live App: https://music-recommender-system-dxgyhmu6pkfa4kyjtmycox.streamlit.app

---

## Features
- Music / Beats similarity using audio features  
- Vibe / Mood-based recommendations  
- Genre-based recommendations  
- Global + Regional dataset support  
- Fast inference using precomputed embeddings  

---

## Approach
- Cleaned and standardized song and artist names
- Created canonical keys to deduplicate tracks
- Built feature matrices for music and vibe similarity
- Used **cosine similarity** for content-based recommendations
- Performed heavy computation offline and inference online

---

## Project Structure
- app.py                          # Streamlit application
- df_global.pkl                   # Global songs metadata
- df_regional.pkl                 # Regional songs metadata
- music_matrix_global.pkl         # Audio feature embeddings (global)
- music_matrix_regional.pkl       # Audio feature embeddings (regional)
- vibe_matrix_global.pkl          # Vibe embeddings (global)
- vibe_matrix_regional.pkl        # Vibe embeddings (regional)
- music_recommendation_combined.ipynb  # Data preprocessing & feature engineering
- requirements.txt                # Python dependencies
- README.md

---

## Evaluation
Traditional accuracy metrics are not applicable.  
The system is validated using:
- Feature-space similarity analysis  
- Intra-list similarity of recommendations  
- Qualitative inspection across modes  

---

## Tech Stack
Python, Pandas, NumPy, scikit-learn, Streamlit

---

## Run Locally
pip install -r requirements.txt  
streamlit run app.py


