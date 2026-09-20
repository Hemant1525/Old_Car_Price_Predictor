import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Old Car Price Predictor", page_icon="🚗", layout="centered")

# ---------------------------------------------------------
# Load model bundle (cached so it loads only once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()
rf = bundle["model"]
ss = bundle["scaler"]
car_name_le = bundle["car_name_le"]
fuel_type_le = bundle["fuel_type_le"]
transmission_le = bundle["transmission_le"]
feature_columns = bundle["feature_columns"]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🚗 Old Car Price Predictor")
st.write("Apni purani car ki details daalo, model uska estimated resale price bata dega.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox(
        "Car Name",
        options=bundle["unique_car_names"],
        help="Type karke search bhi kar sakte ho",
    )
    fuel_type = st.selectbox("Fuel Type", options=bundle["unique_fuel_types"])
    transmission = st.selectbox("Transmission", options=bundle["unique_transmissions"])
    ownership = st.selectbox(
        "Ownership (kitne owners pehle rakh chuke)",
        options=[0, 1, 2, 3, 4, 5],
        index=1,
    )

with col2:
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=1000000, value=30000, step=1000)
    manufacture = st.number_input(
        "Manufacture Year",
        min_value=bundle["manufacture_min"],
        max_value=bundle["manufacture_max"],
        value=2018,
        step=1,
    )
    engine_cc = st.number_input("Engine (cc)", min_value=0, max_value=6000, value=1200, step=50)
    seats = st.selectbox("Seats", options=[2, 4, 5, 6, 7, 8], index=2)

st.divider()

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    try:
        # 1. Raw input ko notebook jaise hi ek DataFrame me daalo
        input_df = pd.DataFrame(
            [[car_name, kms_driven, fuel_type, transmission, ownership, manufacture, engine_cc, seats]],
            columns=feature_columns,
        )

        # 2. Label Encoding (training ke waqt wale hi encoders use karo)
        input_df["car_name"] = car_name_le.transform(input_df["car_name"])
        input_df["fuel_type"] = fuel_type_le.transform(input_df["fuel_type"])
        input_df["Transmission"] = transmission_le.transform(input_df["Transmission"])

        # 3. Scaling
        input_scaled = pd.DataFrame(ss.transform(input_df), columns=input_df.columns)

        # 4. Prediction
        predicted_price = rf.predict(input_scaled)[0]

        st.success(f"### Estimated Price: ₹ {predicted_price:,.0f}")
        st.caption(f"(≈ {predicted_price/100000:.2f} Lakh)")

    except Exception as e:
        st.error(f"Prediction me error aaya: {e}")

st.divider()
with st.expander("📊 Dataset Preview"):
    df_preview = pd.read_csv("car_price.csv")
    st.dataframe(df_preview, use_container_width=True)
