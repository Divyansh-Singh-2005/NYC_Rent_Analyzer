import streamlit as st
import pandas as pd
import joblib
import os

# ---------- Paths ----------
MODEL_PATH = "models/rent_model.pkl"
CLEAN_DATA_PATH = "data/processed/clean_listings.csv"

# ---------- Loaders ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Run the training notebook first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_clean_data():
    if not os.path.exists(CLEAN_DATA_PATH):
        st.error(f"Clean data file not found at {CLEAN_DATA_PATH}. Run the training notebook first.")
        st.stop()
    return pd.read_csv(CLEAN_DATA_PATH)

# ---------- Main ----------
def main():
    st.set_page_config(page_title="NYC Rent Analyzer", page_icon="🏙️", layout="wide")

    st.title("🏙️ NYC Rent Analyzer")
    st.write("Estimate fair rent prices using a Machine Learning model trained on NYC rentals.")

    model = load_model()
    df_clean = load_clean_data()

    st.header("💰 Price Estimator")

    # Dropdowns from data
    boroughs = sorted(df_clean["borough"].dropna().unique().tolist())
    borough = st.selectbox("Borough", boroughs)

    neighborhoods = sorted(
        df_clean.loc[df_clean["borough"] == borough, "neighborhood"].dropna().unique().tolist()
    )
    neighborhood = st.selectbox("Neighborhood", neighborhoods)

    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1, step=1)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
    with col3:
        size_sqft = st.number_input("Size (sqft)", min_value=100, max_value=5000, value=600, step=50)

    st.markdown("---")
    st.subheader("🏢 Building & Amenities (used in your trained model)")

    colA, colB, colC = st.columns(3)

    with colA:
        floor = st.number_input("Floor", min_value=0, max_value=80, value=3, step=1)
        building_age_yrs = st.number_input("Building age (years)", min_value=0, max_value=200, value=20, step=1)
        min_to_subway = st.number_input("Minutes to subway", min_value=0, max_value=60, value=5, step=1)

    with colB:
        no_fee = st.checkbox("No broker fee")
        has_doorman = st.checkbox("Doorman")
        has_elevator = st.checkbox("Elevator")
        has_gym = st.checkbox("Gym")

    with colC:
        has_dishwasher = st.checkbox("Dishwasher")
        has_patio = st.checkbox("Patio/Balcony")
        has_roofdeck = st.checkbox("Roof deck")
        has_washer_dryer = st.checkbox("Washer/Dryer in unit")

    listed_rent = st.number_input(
        "Optional: Listed Rent (USD)",
        min_value=0, max_value=20000, value=0, step=50,
        help="If you enter the asking rent, I will tell you if it's over/underpriced."
    )

    if st.button("Predict Rent"):
        # Build input with ALL columns your model expects
        input_dict = {
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "size_sqft": [size_sqft],
            "borough": [borough],
            "neighborhood": [neighborhood],
            "no_fee": [int(no_fee)],
            "has_dishwasher": [int(has_dishwasher)],
            "min_to_subway": [min_to_subway],
            "floor": [floor],
            "has_doorman": [int(has_doorman)],
            "building_age_yrs": [building_age_yrs],
            "has_elevator": [int(has_elevator)],
            "has_patio": [int(has_patio)],
            "has_roofdeck": [int(has_roofdeck)],
            "has_washer_dryer": [int(has_washer_dryer)],
            "has_gym": [int(has_gym)],
        }

        input_df = pd.DataFrame(input_dict)

        # DEBUG (optional): show the row being sent to the model
        # st.write("Input sent to model:", input_df)

        predicted = model.predict(input_df)[0]

        st.subheader("Prediction")
        st.write(f"**Estimated fair rent:** ${predicted:,.0f} / month")

        if listed_rent > 0:
            diff = listed_rent - predicted
            diff_pct = diff / predicted * 100

            if diff_pct > 15:
                label = "🚨 Overpriced"
            elif diff_pct < -15:
                label = "✅ Underpriced (Good deal)"
            else:
                label = "⚖️ Fairly priced"

            st.markdown(
                f"{label} – listed rent is ${listed_rent:,.0f}, which is {diff_pct:+.1f}% "
                f"relative to the model estimate."
            )

            progress_value = int(50 + diff_pct)
            progress_value = max(0, min(100, progress_value))
            st.progress(progress_value)

if __name__ == "__main__":
    main()
