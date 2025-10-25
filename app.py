import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
import requests 
from datetime import datetime, date

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- API KEY (from your previous message) ---
TMDB_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NTU4YjM3NDI4OTE1OTRhOTY0ZTY1YmQ5MmU1MmIzYiIsIm5iZiI6MTc2MTM1MjUxOS44NTQsInN1YiI6IjY4ZmMxYjQ3MDA4NTRlNzc2MzZhNzJhZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.YBiAUXpERoj_hFtL1PiY3MOIL0hdcrv2CgVMKVR8CgI" 

# --- Get Absolute Path to Files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
IMAGE_PATH = os.path.join(BASE_DIR, "movie.jpg") # Changed to movie.jpg
MODEL_PATH = os.path.join(BASE_DIR, "simple_rf_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "simple_scaler.joblib")

# --- Function to load and encode image ---
@st.cache_data
def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file not found at: {file_path}. Please ensure '{os.path.basename(file_path)}' is in the same folder as 'app.py'.") 
        return None

# --- Get encoded files ---
img_base64 = get_image_as_base64(IMAGE_PATH)

# --- CSS Styling ---
if img_base64:
    background_css = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #FAFAFA;
        position: relative;
        padding-top: 5vh; 
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(10, 5, 30, 0.7); 
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: -1; 
    }}
    """
else:
    background_css = """
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #FAFAFA;
        padding-top: 5vh;
    }}
    """

st.markdown(f"""
    <style>
    /* --- Apply background style --- */
    {background_css}
    
    /* --- Main content area (transparent) --- */
    section[data-testid="stVerticalBlock"] {{
        z-index: 1;
    }}

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {{ 
        background: rgba(0, 0, 0, 0.4); 
        backdrop-filter: blur(15px); 
        -webkit-backdrop-filter: blur(15px); 
        border-right: 1px solid rgba(255, 255, 255, 0.1); 
        padding: 1.5rem; 
        transition: all 0.3s ease; /* Added transition */
    }}
    /* Sidebar hover (optional, if you want it to react) */
    /* [data-testid="stSidebar"]:hover {{ background: rgba(0, 0, 0, 0.5); }} */

    /* --- Main Page Titles --- */
    h1 {{
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(243, 156, 18, 0.7), 0 0 20px rgba(243, 156, 18, 0.5);
        font-size: 3rem; 
        font-weight: 700;
        transition: color 0.3s ease; /* Added transition */
    }}
    h3 {{ 
        font-weight: 400; 
        color: #E0E0E0; 
        text-shadow: 0 0 10px rgba(243, 156, 18, 0.5);
        transition: color 0.3s ease; /* Added transition */
    }}

    /* --- Sidebar Titles --- */
    [data-testid="stSidebar"] h3 {{ 
        color: #FFFFFF; 
        text-shadow: 0 0 5px rgba(243, 156, 18, 0.5); 
        text-align: left; 
        font-size: 1.25rem; 
        font-weight: 600; 
        margin-top: 0;
        margin-bottom: 1rem;
    }}
    
    /* --- Sidebar Widgets (Selectbox) --- */
    [data-testid="stSidebar"] .stSelectbox label {{ 
        color: #E0E0E0 !important; 
        font-weight: 500; 
        text-shadow: none; 
        transition: color 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{ 
        background-color: rgba(255, 255, 255, 0.15); 
        color: #FAFAFA; 
        border-radius: 10px; 
        border: 1px solid rgba(255, 255, 255, 0.3); 
        transition: all 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {{ 
        background-color: rgba(255, 255, 255, 0.25); 
        border-color: rgba(243, 156, 18, 0.5);
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div:focus-within {{ /* For when it's actively open/focused */
        border-color: #F39C12;
        box-shadow: 0 0 0 2px rgba(243, 156, 18, 0.5);
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div > div {{ color: #FAFAFA !important; }}

    /* --- Sidebar Widgets (Text Input) --- */
    [data-testid="stSidebar"] .stTextInput label {{ 
        color: #E0E0E0 !important; 
        font-weight: 500; 
        text-shadow: none; 
        transition: color 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.15); 
        color: #FAFAFA; 
        border-radius: 10px; 
        border: 1px solid rgba(255, 255, 255, 0.3); 
        transition: all 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] .stTextInput > div > div > input:hover {{ 
        background-color: rgba(255, 255, 255, 0.25); 
        border-color: rgba(243, 156, 18, 0.5);
    }}
    [data-testid="stSidebar"] .stTextInput > div > div > input:focus {{ 
        border-color: #F39C12;
        box-shadow: 0 0 0 2px rgba(243, 156, 18, 0.5);
        outline: none; /* Remove default outline */
    }}

    /* --- Sidebar Tabs --- */
    [data-testid="stSidebar"] [data-testid="stTabs"] button {{
        color: #E0E0E0;
        border-radius: 8px;
        transition: all 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] [data-testid="stTabs"] button:hover {{
        background-color: rgba(243, 156, 18, 0.15);
        color: #F1C40F;
    }}
    [data-testid="stSidebar"] [data-testid="stTabs"] button[aria-selected="true"] {{
        color: #FFFFFF;
        background-color: rgba(243, 156, 18, 0.3);
        font-weight: 600;
        border-bottom: 2px solid #F39C12;
    }}

    /* --- Sidebar Button --- */
    [data-testid="stSidebar"] .stButton > button {{ 
        width: 100%; margin: 2rem auto 0 auto; display: block; padding: 12px 30px; 
        font-size: 1.1rem; font-weight: 700; color: #111111; 
        background: linear-gradient(90deg, #F39C12 0%, #F1C40F 100%); 
        border: none; border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.4); 
        transition: all 0.3s ease; /* Added transition */
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{ 
        transform: translateY(-5px) scale(1.02); /* More pronounced hover */
        box-shadow: 0 8px 25px rgba(243, 156, 18, 0.7); 
        background: linear-gradient(90deg, #F1C40F 0%, #F39C12 100%); /* Subtle gradient shift */
    }}

    /* --- Movie Poster on Main Page --- */
    .movie-poster {{
        width: 100%;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease; /* Added transition */
    }}
    .movie-poster:hover {{
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.9);
    }}

    /* --- Movie Tagline on Main Page --- */
    .movie-tagline {{
        font-size: 1.1rem;
        font-style: italic;
        color: #E0E0E0;
        text-shadow: 0 0 5px rgba(0,0,0,0.5);
        transition: color 0.3s ease; /* Added transition */
    }}

    /* --- Movie Overview on Main Page --- */
    .movie-overview {{
        font-size: 1rem;
        color: #FAFAFA;
        background-color: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 10px;
        transition: background-color 0.3s ease; /* Added transition */
    }}
    .movie-overview:hover {{
        background-color: rgba(0, 0, 0, 0.3);
    }}

    /* --- Prediction Box (Gold) --- */
    .prediction-box, .actual-box {{ /* Combined styles for common properties */
        border-radius: 10px;
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease; /* Added transition */
    }}
    .prediction-box {{
        background-color: rgba(243, 196, 15, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid #F39C12;
    }}
    .prediction-box:hover {{
        background-color: rgba(243, 196, 15, 0.3);
        box-shadow: 0 5px 20px rgba(243, 156, 18, 0.5);
    }}
    .prediction-box span {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(243, 156, 18, 0.8);
    }}
    
    /* --- Actual Revenue Box (Blue) --- */
    .actual-box {{
        background-color: rgba(31, 97, 141, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid #1F618D;
    }}
    .actual-box:hover {{
        background-color: rgba(31, 97, 141, 0.3);
        box-shadow: 0 5px 20px rgba(31, 97, 141, 0.5);
    }}
    .actual-box span {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(84, 153, 199, 0.8);
    }}

    /* --- Cast & Crew Styles --- */
    .cast-member-container {{
        transition: all 0.3s ease; /* For overall column effect */
    }}
    .cast-member-container:hover {{
        transform: translateY(-5px);
    }}
    .cast-photo {{
        width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease; /* Added transition */
    }}
    .cast-photo:hover {{
        box-shadow: 0 6px 15px rgba(243, 156, 18, 0.4);
        border-color: rgba(243, 156, 18, 0.3);
    }}
    .cast-photo-placeholder {{
        width: 100%;
        aspect-ratio: 2/3; /* Maintain poster aspect ratio */
        height: auto;
        border-radius: 10px;
        background-color: #333;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #888;
        font-weight: 500;
        margin-bottom: 0.5rem;
        border: 1px solid #555;
        transition: all 0.3s ease; /* Added transition */
    }}
    .cast-photo-placeholder:hover {{
        background-color: #444;
        border-color: #777;
    }}
    .cast-name, .cast-char {{
        text-align: center;
        transition: color 0.3s ease; /* Added transition */
    }}
    .cast-name {{
        font-weight: 600;
        color: #FFFFFF;
        font-size: 0.9rem;
        height: 2.5em; /* Allow for two lines of text */
        overflow: hidden;
    }}
    .cast-char {{
        font-style: italic;
        color: #E0E0E0;
        font-size: 0.8rem;
    }}
    .cast-member-container:hover .cast-name {{
        color: #F39C12; /* Highlight name on hover */
    }}
    .cast-member-container:hover .cast-char {{
        color: #F1C40F; /* Highlight character on hover */
    }}

    /* --- Styled Dataframe --- */
    [data-testid="stDataFrame"] {{
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        transition: all 0.3s ease; /* Added transition */
    }}
    [data-testid="stDataFrame"]:hover {{
        background-color: rgba(0, 0, 0, 0.3);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
    }}
    [data-testid="stDataFrame"] th {{
        background-color: rgba(255, 255, 255, 0.1);
        color: #F39C12;
        text-shadow: none;
    }}
    [data-testid="stDataFrame"] td {{
        color: #FAFAFA;
    }}
    
    </style>
""", unsafe_allow_html=True)


# --- Load Artifacts ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model/Scaler not found. Please run 'python train_simple_model.py' first.")
        st.error(f"Looking for: {MODEL_PATH} and {SCALER_PATH}")
        return None, None

model, scaler = load_model_and_scaler()
MODEL_FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_year']

# --- API Call Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_upcoming_movies():
    """Fetches upcoming movies from TMDB."""
    url = f"https://api.themoviedb.org/3/movie/upcoming?language=en-US&page=1"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        data = response.json()
        return {movie['title']: movie['id'] for movie in data.get('results', []) if movie.get('id')}
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching upcoming movies: {e}")
        return {}

@st.cache_data(ttl=3600) # Cache for 1 hour
def search_movies(query):
    """Fetches movies from TMDB based on a search query."""
    url = f"https://api.themoviedb.org/3/search/movie?language=en-US&page=1&include_adult=false"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}
    params = {'query': query}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() 
        data = response.json()
        results = {}
        for movie in data.get('results', []):
            if movie.get('id'):
                year = movie.get('release_date', '----').split('-')[0]
                title = movie.get('title', 'Unknown Title')
                results[f"{title} ({year})"] = movie['id']
        return results
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error searching movies: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_movie_details(movie_id):
    """Fetches detailed info for a single movie ID."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie details: {e}")
        return None

@st.cache_data(ttl=3600)
def get_movie_credits(movie_id):
    """Fetches cast (top 5) and director for a single movie ID."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        cast = data.get('cast', [])
        crew = data.get('crew', [])
        
        director = "Not Found"
        for member in crew:
            if member.get('job') == 'Director':
                director = member.get('name', 'Not Found')
                break
        
        top_cast = []
        for actor in cast[:5]:
            top_cast.append({
                "name": actor.get('name', 'N/A'),
                "character": actor.get('character', 'N/A'),
                "profile_path": actor.get('profile_path')
            })
        return director, top_cast
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie credits: {e}")
        return "Error", []

# --- App Layout ---
st.title("🎬 Movie Revenue Predictor")
st.markdown("### How much will your blockbuster make? 💰")

col1, col2 = st.columns([2, 3]) # 2 parts for poster, 3 for info

if model and scaler: 
    
    # --- Sidebar for Inputs ---
    with st.sidebar:
        
        movie_id_to_predict = None
        title_to_predict = None
        
        tab_upcoming, tab_search = st.tabs(["🌟 Upcoming", "🔍 Search"])

        with tab_upcoming:
            st.markdown("### Select Upcoming Movie")
            movie_dict = get_upcoming_movies()
            if movie_dict:
                selected_title_upcoming = st.selectbox("Choose a movie:", 
                                                      options=list(movie_dict.keys()),
                                                      index=0,
                                                      key="upcoming_select")
                if selected_title_upcoming:
                    movie_id_to_predict = movie_dict[selected_title_upcoming]
                    title_to_predict = selected_title_upcoming
            else:
                st.warning("Could not fetch upcoming movies.")

        with tab_search:
            st.markdown("### Search for Any Movie")
            search_query = st.text_input("Type a movie title:")
            
            if search_query:
                search_results = search_movies(search_query)
                if search_results:
                    selected_title_search = st.selectbox("Select a result:", 
                                                         options=list(search_results.keys()),
                                                         key="search_select")
                    if selected_title_search:
                        movie_id_to_predict = search_results[selected_title_search]
                        title_to_predict = selected_title_search
                else:
                    st.info("No results found for your search.")
        
        # --- Predict Button (Common to both tabs) ---
        if st.button("Get Movie Info"):
            if movie_id_to_predict and title_to_predict:
                with st.spinner('Fetching details and predicting...'):
                    
                    # 1. Get Movie Details (Budget, Revenue, Overview...)
                    details = get_movie_details(movie_id_to_predict)
                    
                    # 2. Get Movie Credits (Cast, Director)
                    director, top_cast = get_movie_credits(movie_id_to_predict)
                    
                    if details:
                        # 3. Extract Features for Model
                        budget = details.get('budget', 0)
                        popularity = details.get('popularity', 0)
                        runtime = details.get('runtime', 0)
                        vote_average = details.get('vote_average', 0)
                        vote_count = details.get('vote_count', 0)
                        release_date_str = details.get('release_date', '')
                        
                        # Set defaults for prediction if data is missing
                        if budget < 1000: budget = 1000000 
                        if runtime == 0: runtime = 120 
                        if release_date_str:
                            release_year = datetime.strptime(release_date_str, '%Y-%m-%d').year
                        else:
                            release_year = datetime.now().year 
                        
                        # 4. Create DataFrame for model
                        input_data = pd.DataFrame([[
                            budget, popularity, runtime, vote_average, vote_count, release_year
                        ]], columns=MODEL_FEATURES)

                        # 5. --- (ALWAYS PREDICT) ---
                        input_data_copy = input_data.copy()
                        input_data_copy['budget'] = np.log1p(input_data_copy['budget'])
                        input_scaled = scaler.transform(input_data_copy)
                        pred_log = model.predict(input_scaled)[0]
                        predicted_revenue = np.expm1(pred_log)
                        
                        # 6. --- (CHECK FOR ACTUAL REVENUE) ---
                        status = details.get('status')
                        actual_revenue = details.get('revenue', 0)
                        release_date_obj = None
                        if release_date_str:
                            release_date_obj = datetime.strptime(release_date_str, '%Y-%m-%d').date()
                        
                        today = date.today()
                        
                        st.session_state.actual_revenue = None # Reset
                        if status == 'Released' and release_date_obj and release_date_obj < today and actual_revenue > 0:
                            st.session_state.actual_revenue = actual_revenue

                        # 7. Get Extra Info for Display
                        poster_path = details.get('poster_path')
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
                        tagline = details.get('tagline', '')
                        overview = details.get('overview', 'No overview available.')

                        # 8. Save results to Session State
                        st.session_state.show_results = True
                        st.session_state.title = title_to_predict
                        st.session_state.poster_url = poster_url
                        st.session_state.tagline = tagline
                        st.session_state.overview = overview
                        st.session_state.director = director
                        st.session_state.cast = top_cast
                        st.session_state.predicted_revenue = predicted_revenue # Always save prediction
                        st.session_state.features = {
                            "Budget": f"${budget:,.0f}",
                            "Runtime": f"{runtime} minutes",
                            "Popularity": f"{popularity:.2f}",
                            "Vote Average": f"{vote_average}/10",
                            "Vote Count": f"{vote_count:,}",
                            "Release Year": release_year
                        }
                        
            else:
                st.error("Please select a movie from 'Upcoming' or 'Search' first.")

    # --- Display Results on Main Page (using Session State) ---
    if st.session_state.get('show_results', False):
        
        with col1:
            if st.session_state.poster_url:
                st.markdown(f'<img src="{st.session_state.poster_url}" class="movie-poster">', unsafe_allow_html=True)
            else:
                st.markdown('<div class="movie-poster" style="background-color: #222; display: flex; align-items: center; justify-content: center; height: 500px; border: 2px solid #555;">No Poster Available</div>', unsafe_allow_html=True)

        with col2:
            st.title(st.session_state.title)
            
            if st.session_state.tagline:
                st.markdown(f'<p class="movie-tagline">"{st.session_state.tagline}"</p>', unsafe_allow_html=True)
            
            # --- Always show prediction ---
            st.subheader("Estimated Revenue")
            st.markdown(f'<div class="prediction-box"><span>${st.session_state.predicted_revenue:,.2f}</span></div>', unsafe_allow_html=True)

            # --- Show actual revenue ONLY if it exists ---
            if st.session_state.actual_revenue:
                st.subheader("Actual Box Office")
                st.markdown(f'<div class="actual-box"><span>${st.session_state.actual_revenue:,.2f}</span></div>', unsafe_allow_html=True)

            st.subheader("Overview")
            st.markdown(f'<p class="movie-overview">{st.session_state.overview}</p>', unsafe_allow_html=True)
            
            # --- Display Cast & Crew ---
            st.subheader("Cast & Crew")
            st.markdown(f"**Director:** {st.session_state.director}")
            
            st.markdown("**Top Cast:**")
            cast_cols = st.columns(5)
            for i, actor in enumerate(st.session_state.cast):
                with cast_cols[i]:
                    # Added a container for hover effect on entire cast member
                    st.markdown(f'<div class="cast-member-container">', unsafe_allow_html=True) 
                    if actor['profile_path']:
                        photo_url = f"https://image.tmdb.org/t/p/w185{actor['profile_path']}"
                        st.markdown(f'<img src="{photo_url}" class="cast-photo">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="cast-photo-placeholder">No Photo</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='cast-name'>{actor['name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='cast-char'>as {actor['character']}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True) # Close container
            
            st.subheader("Model Inputs (Auto-Filled)")
            st.dataframe(pd.Series(st.session_state.features), use_container_width=True)

else:
    st.error("Model artifacts not loaded. Please run 'train_simple_model.py' first.")

