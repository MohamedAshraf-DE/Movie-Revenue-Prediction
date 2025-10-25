# 🎬 Movie Revenue Predictor 💰

![Movie App Screenshot](https://github.com/MohamedAshraf-DE/Movie-Revenue-Prediction/blob/main/movie.jpg?raw=true)

A **live-data web app** that predicts movie revenue using machine learning and the **TMDB API**.  
This project bridges the gap between **cinema creativity** and **business intelligence**, allowing both filmmakers and movie enthusiasts to explore how different factors influence a movie’s financial success.

---

## 🌟 Why This Project Matters

### 🎥 For Movie Directors & Studios
- Run “what-if” scenarios to **estimate financial impact** of different production choices.  
- Predict how a higher budget, star cast, or release timing might affect revenue.  
- Reduce financial risks and make **data-backed decisions**.  
- Improve marketing and distribution strategy with clearer revenue insights.

### 👥 For Movie Fans & Users
- Understand why some films succeed while others flop.  
- Compare your “box office gut feeling” with a **real ML prediction**.  
- Explore the influence of cast, runtime, popularity, and release year on a film’s earnings.  

### 💼 Business Value
- Helps studios maximize profits by highlighting **key revenue drivers**.  
- Encourages **strategic planning** in production, marketing, and release.  
- Bridges creative vision with **financial intelligence**, transforming the decision-making process in the entertainment industry.  
- Makes movie analytics **accessible** to everyone—from professionals to enthusiasts.

---

## ✨ Features & Highlights

| Feature | Description |
|---------|-------------|
| 🔍 Live Movie Data | Search any movie or browse upcoming releases via TMDB API. |
| 💵 Dual Revenue Display | Compare **Estimated Revenue** (model) with **Actual Revenue** (historical). |
| 🎨 Full Info Hub | Poster, tagline, plot, director, top 5 cast members with images. |
| 🖱️ Interactive Search | Dynamic “as-you-type” results from TMDB database. |
| 🖌️ Custom UI/UX | Frosted glass sidebar, poster-themed colors, smooth transitions. |
| 🤖 ML-Powered Predictions | Uses Random Forest model trained on 6 key features for accurate estimates. |

---

## 🚀 How to Use This App

1. **Prerequisites**  
   - Python 3.9+  
   - TMDB API Read Access Token

2. **Clone & Install Dependencies**
```bash
git clone https://github.com/MohamedAshraf-DE/Movie-Revenue-Prediction.git
cd Movie-Revenue-Prediction
pip install -r requirements.txt
Add Your API Key
Open app.py and set:

python
Copy code
TMDB_ACCESS_TOKEN = "YOUR_API_KEY"
Train the Model (One-Time)

bash
Copy code
python train_simple_model.py
This will generate simple_rf_model.joblib and simple_scaler.joblib.

Run the App

bash
Copy code
streamlit run app.py
Open your browser & explore the interactive app! 🎉

🛠️ Technical Details (Optional)
Model: RandomForestRegressor

Features Used: budget, popularity, runtime, vote_average, vote_count, release_year

Target: Revenue (log-transformed during training)

Performance:

R² ≈ 0.605

MAE ≈ $58.9M

Scaling: StandardScaler applied to input features

Libraries: pandas, numpy, scikit-learn, xgboost, joblib, streamlit, requests

📞 Contact & Portfolio
Connect or check out my work:

🌐 Portfolio: https://mohamed-ashraf-github-io.vercel.app/

🔗 LinkedIn: https://www.linkedin.com/in/mohamed--ashraff

🐙 GitHub: https://github.com/MohamedAshraf-DE/MohamedAshraf.github.io

Freelance Accounts
💼 Upwork: https://www.upwork.com/freelancers/~0190a07e5b17474f9f?mp_source=share

💼 Mostql: https://mostaql.com/u/MohamedA_Data

💼 Khamsat: https://khamsat.com/user/mohamed_ashraf124

💼 Freelancer: https://www.freelancer.com/dashboard

💼 Outlier: https://app.outlier.ai/profile

vbnet
Copy code
