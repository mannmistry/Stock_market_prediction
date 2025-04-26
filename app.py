import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime
import time

# Load trained model
model = load_model("model/cnn_lstm_stock_model.h5")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# --- Sidebar UI ---
st.sidebar.title("⚙️ Options")

# Popular stock dropdown
popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
selected_stock = st.sidebar.selectbox("Choose a popular stock:", options=popular_stocks, index=0)

# Custom symbol override
manual_symbol = st.sidebar.text_input("Or enter custom stock symbol:", value=selected_stock)
symbol = manual_symbol if manual_symbol else selected_stock

# Chart type
chart_type = st.sidebar.radio("Chart Type", ["Close Price", "Scaled Price"])

# Future prediction slider (for future expansion)
prediction_horizon = st.sidebar.slider("Days to Predict Ahead", min_value=1, max_value=5, value=1)

# Model structure display
with st.sidebar.expander("📐 Model Architecture (CNN-LSTM)"):
    st.code("""
[Input] → Conv1D(64 filters, kernel_size=3)
       → MaxPooling1D(pool_size=2)
       → LSTM(units=50)
       → Dropout(0.2)
       → Dense(1)
""", language='text')

# --- Main Title ---
st.title("📈 Stock Price Prediction using CNN-LSTM")

# Date input
start_date = st.date_input("Select Start Date", value=datetime.date(2020, 1, 1))
end_date = st.date_input("Select End Date", value=datetime.date.today())

predict = st.button("🔮 Predict Closing Price")

# --- Data Fetch Function ---
def fetch_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = pd.DataFrame()

# --- Prediction Logic ---
if predict:
    try:
        df = fetch_stock_data(symbol, start_date, end_date)

        if df.empty:
            st.error("❌ No stock data found. Please check the symbol or date range.")
        else:
            st.write(f"✅ Loaded **{len(df)}** rows from **{start_date}** to **{end_date}**.")

            if len(df) < 60:
                st.warning(f"⚠️ Only {len(df)} data points found. Please select at least 60 trading days.")
            else:
                # Show train/test split info
                train_size = int(len(df) * 0.8)
                test_size = len(df) - train_size
                st.markdown(f"🧪 **Data Used:** {train_size} training rows, {test_size} testing rows")

                # Preprocess
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df)

                last_60 = scaled[-60:]
                x_input = np.reshape(last_60, (1, 60, 1))

                # Predict with timing
                start_time = time.time()
                pred = model.predict(x_input)
                predicted_price = scaler.inverse_transform(pred)[0][0]
                execution_time = time.time() - start_time

                st.success(f"📊 Predicted Next Closing Price for **{symbol.upper()}**: **${round(predicted_price, 2)}**")
                st.info(f"⏰ Prediction Time: {execution_time:.2f} seconds")

                # Chart display
                st.subheader("📉 Stock Trend")
                if chart_type == "Close Price":
                    st.line_chart(df['Close'])
                else:
                    st.line_chart(pd.Series(scaled.flatten(), index=df.index))

                # Actual vs Predicted visualization
                st.subheader("🧠 Prediction Visual")
                plot_df = df.copy()
                plot_df['Predicted'] = np.nan
                plot_df.iloc[-1, plot_df.columns.get_loc('Predicted')] = predicted_price
                st.line_chart(plot_df.tail(100))

                # Store prediction history
                st.session_state.predictions.append({
                    "Symbol": symbol.upper(),
                    "Start Date": str(start_date),
                    "End Date": str(end_date),
                    "Prediction Date": datetime.datetime.today().strftime('%Y-%m-%d'),
                    "Predicted Price": round(predicted_price, 2)
                })

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Prediction History Display ---
if st.session_state.predictions:
    st.subheader("📋 Prediction History (This Session)")
    history_df = pd.DataFrame(st.session_state.predictions)
    st.dataframe(history_df)

    # Export to CSV
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# --- Sidebar Stock Summary ---
with st.sidebar.expander("📊 Stock Summary"):
    try:
        if not df.empty:
            st.markdown(f"**Last Close:** ${df['Close'].iloc[-1]:.2f}")
            st.markdown(f"**Highest (Range):** ${df['Close'].max():.2f}")
            st.markdown(f"**Lowest (Range):** ${df['Close'].min():.2f}")
            st.markdown(f"**Volatility (STD):** ±{df['Close'].std():.2f}")
    except:
        st.info("Summary will appear after prediction.")
