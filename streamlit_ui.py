import streamlit as st
import requests
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
#API_URL = "http://127.0.0.1:8000/product_sales_forecasting/v1/forecast/recursive_order_sales_forecast"    #use for local purpost
API_URL = "http://localhost:8000/product_sales_forecasting/v1/forecast/recursive_order_sales_forecast"     # Docker

st.set_page_config(page_title="Sales Forecast App", layout="wide")

st.title("📈 Product Sales Forecasting")

# -------------------------------
# INPUT FORM
# -------------------------------
with st.form("forecast_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        store_id = st.number_input("Store ID", min_value=1, value=1)
        store_type = st.selectbox("Store Type", ["S1", "S2", "S3", "S4"])

    with col2:
        location_type = st.selectbox("Location Type", ["L1", "L2", "L3", "L4"])
        region_code = st.selectbox("Region Code", ["R1", "R2", "R3", "R4"])

    with col3:
        start_date = st.date_input("Prediction Start Date")
        period = st.number_input("Forecast Days", min_value=1, value=7)

    submit = st.form_submit_button("Generate Forecast")

# -------------------------------
# API CALL
# -------------------------------
if submit:

    payload = {
        "Store_id": store_id,
        "Store_Type": store_type,
        "Location_Type": location_type,
        "Region_Code": region_code,
        "Prediction_Start_Date": str(start_date),
        "period": period
    }

    with st.spinner("Calling API..."):

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()

                df = pd.DataFrame(data)

                st.success("Forecast generated successfully!")

                # -------------------------------
                # DISPLAY TABLE
                # -------------------------------
                st.subheader("📊 Forecast Data")
                st.dataframe(df, width="stretch")

                # -------------------------------
                # DISPLAY CHART
                # -------------------------------
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📉 Sales Forecast Trend")
                    st.line_chart(df.set_index("Date")["Pre_Sales"])

                with col2:
                    st.subheader("📦 Orders Forecast Trend")
                    st.line_chart(df.set_index("Date")["Pre_Order"])

            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

####################################################################
######  Run Command:  streamlit run streamlit_ui.py   ##############
####################################################################