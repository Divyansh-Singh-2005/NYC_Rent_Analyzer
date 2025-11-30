# NYC_Rent_Analyzer
End-to-end machine learning application for predicting fair NYC rental prices using real listing data, trained with a Random Forest model and deployed via a Streamlit web app.


# 🏙️ NYC Rent Analyzer — Machine Learning Price Prediction

NYC Rent Analyzer is an end-to-end **Machine Learning application built in Python** that estimates fair rental prices for New York City apartments based on real listing data and property features.  
The goal is to provide transparent, data-driven rent evaluation and highlight how applied ML can support housing market analysis.

The system includes:
- Data cleaning and feature engineering pipeline
- Trained regression model for price prediction
- Interactive **Streamlit** dashboard for real-time inference

---

## 📊 Dataset

- **Raw dataset:** 3,539 listings × 18 features  
- **After cleaning:** 3,522 listings × 17 features  
- **Train/Test split:**  
  - Training: 2,817 samples  
  - Testing: 705 samples  

### Core features used (16 inputs)

- Bedrooms, Bathrooms, Size (sqft)
- Borough & Neighborhood
- Floor number
- Minutes to subway
- Building age (years)
- Amenities:
  - No-fee
  - Doorman
  - Elevator
  - Gym
  - Dishwasher
  - Washer/Dryer in unit
  - Patio/Balcony
  - Roof deck

Target:
- **Monthly rent (USD)**

---

## 🤖 Model

- **Algorithm:** Random Forest Regressor (scikit-learn)  
- **Preprocessing pipeline:**
  - Numeric → StandardScaler
  - Categorical → One-Hot Encoding
  - Combined via ColumnTransformer

---

## ✅ Model Performance (Test Set)

| Metric | Result |
|--------|---------|
| **MAE** | **\$703.7** |
| **RMSE** | **\$1,261.8** |
| **R²** | **0.839** |

> The model explains ~84% of the variance in NYC rental prices with an average prediction error under **\$750/month** on unseen listings.

---

## 🖥️ Application

The trained model is deployed using a **Streamlit web app** which provides:

- Live rent prediction for user-entered listings
- Overpriced / Fair / Underpriced classification
- Neighborhood-based comparable property display
- Feature-level input including amenities & transit access

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** – Data processing
- **Scikit-learn** – ML pipelines & modeling
- **Joblib** – Model serialization
- **Streamlit** – Application deployment

---

## 📁 Project Structure

```
NYC_Rent_Analyzer/
│
├── data/
│   ├── streeteasy_rentals.csv      # Raw dataset
│   └── clean_listings.csv          # Cleaned training data
│
├── models/
│   └── rent_model.pkl              # Trained ML pipeline
│
├── training_notebook.ipynb         # Data cleaning & model training
│
├── rent_app.py                     # Streamlit app for live predictions
│
└── README.md
```

---

## ⚙️ Setup & Run

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit
```

---

### 2️⃣ Run training notebook

Open `training_notebook.ipynb` in Jupyter and run all cells.

This will:
- Clean the dataset → `data/clean_listings.csv`
- Train the model
- Save the trained pipeline → `models/rent_model.pkl`

---

### 3️⃣ Launch the application

From the project root:

```bash
streamlit run rent_app.py
```

Open the provided browser URL (usually `http://localhost:8501`) to interact with the app.

---

## 🛠️ What I Learned

- Designing a full **ML pipeline** from raw data preprocessing to deployed inference  
- Performing **feature engineering** across heterogeneous housing attributes  
- Training & evaluating regression models using real-world metrics (MAE, RMSE, R²)  
- Working with **ColumnTransformers and pipelines** to ensure consistent training & prediction flows  
- Packaging and deploying models with **joblib + Streamlit**  
- Managing practical issues in production ML such as feature alignment and inference validation

---

## 🚀 Future Improvements

- Add SHAP explainability dashboards for prediction transparency
- Implement borough-level **time series forecasting** (Prophet / SARIMA)
- Build a **FastAPI REST API** endpoint for scalable ML inference
- Integrate interactive NYC mapping (Folium / Plotly)

---

## 📌 Acknowledgments

Data sourced from publicly available NYC rental and housing listings datasets for educational and analytical purposes.

---

## 📬 Contact

If you'd like to discuss the modeling approach, app architecture, or potential extensions, feel free to connect with me on LinkedIn.

---

**NYC Rent Analyzer – Turning urban data into real-world ML insight.**
