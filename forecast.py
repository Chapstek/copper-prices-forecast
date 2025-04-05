try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.arima.model import ARIMA
    import yfinance as yf
    from pmdarima import auto_arima
    import streamlit as st
    import time
except ImportError as e:
    print(f"Missing module: {e}. Please install it with 'pip install {e.name}'.")
    exit(1)

# Streamlit app setup
st.title("Zambia Copper Price Forecasting Tool")
st.write("Forecast copper prices in USD and ZMW using historical data and advanced models.")

# Fetch real data
@st.cache_data(ttl=3600)
def fetch_data():
    try:
        # Attempt HG=F with retries
        for attempt in range(3):
            copper_data = yf.download('HG=F', start='2015-01-01', end='2025-03-31', progress=False)
            if not copper_data.empty and 'Close' in copper_data.columns:
                st.write("Using HG=F (Copper Futures) data from yfinance.")
                break
            st.warning(f"Attempt {attempt + 1}: No data for HG=F. Retrying in 2 seconds...")
            time.sleep(2)
        else:
            st.error("No valid data for HG=F (Copper Futures) after retries.")
            st.write("Debug: copper_data from yfinance:", copper_data if copper_data.empty else copper_data.tail())
            st.warning("Switching to COPX (Copper Miners ETF) as fallback...")
            copper_data = yf.download('COPX', start='2015-01-01', end='2025-03-31', progress=False)
            if copper_data.empty or 'Close' not in copper_data.columns:
                st.warning("COPX failed. Trying SCCO (Southern Copper Corp)...")
                copper_data = yf.download('SCCO', start='2015-01-01', end='2025-03-31', progress=False)
                if copper_data.empty or 'Close' not in copper_data.columns:
                    st.warning("SCCO failed. Loading static HG=F data as last resort...")
                    try:
                        copper_data = pd.read_csv('copper_fallback.csv', index_col='Date', parse_dates=True, encoding='utf-8')
                        if copper_data.empty or 'Close' not in copper_data.columns:
                            st.error("Static fallback data in copper_fallback.csv is invalid (empty or missing 'Close' column).")
                            return None
                        st.write("Using static HG=F data from copper_fallback.csv.")
                    except FileNotFoundError:
                        st.error("copper_fallback.csv not found in the app directory.")
                        return None
                else:
                    st.write("Using SCCO data (note: prices reflect stock shares, not futures tons).")
            else:
                st.write("Using COPX data (note: prices reflect ETF shares, not futures tons).")

        copper = copper_data['Close'].resample('ME').mean()
        df_copper = pd.DataFrame({'Price_USD': copper}).dropna()

        # Attempt ZMW=X with retries
        for attempt in range(3):
            zmw_data = yf.download('ZMW=X', start='2015-01-01', end='2025-03-31', progress=False)
            if not zmw_data.empty and 'Close' in zmw_data.columns:
                st.write("Using ZMW=X (Exchange Rate) data from yfinance.")
                break
            st.warning(f"Attempt {attempt + 1}: No data for ZMW=X. Retrying in 2 seconds...")
            time.sleep(2)
        else:
            st.error("No valid data for ZMW=X (Exchange Rate) after retries.")
            st.write("Debug: zmw_data from yfinance:", zmw_data if zmw_data.empty else zmw_data.tail())
            st.warning("Loading static ZMW=X data from zmw_fallback.csv...")
            try:
                zmw_data = pd.read_csv('zmw_fallback.csv', index_col='Date', parse_dates=True, encoding='utf-8')
                if zmw_data.empty or 'Close' not in zmw_data.columns:
                    st.error("Static fallback data in zmw_fallback.csv is invalid (empty or missing 'Close' column).")
                    return None
                st.write("Using static ZMW=X data from zmw_fallback.csv.")
            except FileNotFoundError:
                st.error("zmw_fallback.csv not found in the app directory.")
                return None

        zmw = zmw_data['Close'].resample('ME').mean()
        df_zmw = pd.DataFrame({'Exchange_Rate': zmw}).dropna()

        df = df_copper.join(df_zmw, how='inner')
        if df.empty:
            st.error("No overlapping data between copper prices and exchange rates.")
            return None
        if 'HG=F' in copper_data.columns or copper_data.index.name == 'HG=F':
            df['Price_USD'] *= 2204.6  # Convert to USD/ton (HG=F is in USD/lb)
        df['Price_ZMW'] = df['Price_USD'] * df['Exchange_Rate']

        df['Log_Price_USD'] = np.log(df['Price_USD'])
        df['Log_Exchange_Rate'] = np.log(df['Exchange_Rate'])
        df['USD_Volatility'] = df['Price_USD'].pct_change().rolling(window=3).std().bfill()
        df['ZMW_Volatility'] = df['Exchange_Rate'].pct_change().rolling(window=3).std().bfill()
        st.write("Debug: Data fetched successfully. df shape:", df.shape)
        st.write("Debug: df head:", df.head())
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return None

df = fetch_data()
if df is None:
    st.write("Cannot proceed without data. Please try again later or contact support.")
    st.stop()

# Train-test split
train_size = int(len(df) * 0.8)
train_df, test_df = df[:train_size], df[train_size:]

# Sidebar for forecast horizon
st.sidebar.header("Settings")
forecast_steps = st.sidebar.slider("Forecast Horizon (Months)", min_value=6, max_value=24, value=12, step=6)

# SARIMA models
@st.cache_resource
def train_sarima():
    # Limit seasonal differencing to avoid over-differencing
    model_auto_usd = auto_arima(train_df['Log_Price_USD'], seasonal=True, m=12, max_D=1, max_d=2, stepwise=True, maxiter=50)
    model_usd = ARIMA(train_df['Log_Price_USD'], order=model_auto_usd.order, seasonal_order=model_auto_usd.seasonal_order)
    model_fit_usd = model_usd.fit()

    model_auto_zmw = auto_arima(train_df['Log_Exchange_Rate'], seasonal=True, m=12, max_D=1, max_d=2, stepwise=True, maxiter=50)
    model_zmw = ARIMA(train_df['Log_Exchange_Rate'], order=model_auto_zmw.order, seasonal_order=model_auto_zmw.seasonal_order)
    model_fit_zmw = model_zmw.fit()
    return model_fit_usd, model_fit_zmw

model_fit_usd, model_fit_zmw = train_sarima()

# VAR model
var_data = train_df[['Log_Price_USD', 'Log_Exchange_Rate', 'USD_Volatility', 'ZMW_Volatility']]
model_var = VAR(var_data)
max_lags = min(10, len(train_df) // (var_data.shape[1] + 1) - 1)
try:
    lag_order = model_var.select_order(maxlags=max_lags)
    optimal_lag = lag_order.selected_orders['aic']
except ValueError:
    optimal_lag = 1
var_fit = model_var.fit(optimal_lag)

# Evaluate models
sarima_usd_test_log = model_fit_usd.forecast(steps=len(test_df))
sarima_zmw_test_log = model_fit_zmw.forecast(steps=len(test_df))
sarima_usd_test = np.exp(sarima_usd_test_log)
sarima_zmw_test = np.exp(sarima_zmw_test_log)
sarima_mae_usd = np.mean(np.abs(test_df['Price_USD'] - sarima_usd_test))
sarima_mae_zmw = np.mean(np.abs(test_df['Exchange_Rate'] - sarima_zmw_test))

var_forecast_test = var_fit.forecast(var_data.values[-optimal_lag:], steps=len(test_df))
var_usd_test = np.exp(var_forecast_test[:, 0])
var_zmw_test = np.exp(var_forecast_test[:, 1])
var_mae_usd = np.mean(np.abs(test_df['Price_USD'] - var_usd_test))
var_mae_zmw = np.mean(np.abs(test_df['Exchange_Rate'] - var_zmw_test))

# Ensemble forecast
var_full = VAR(df[['Log_Price_USD', 'Log_Exchange_Rate', 'USD_Volatility', 'ZMW_Volatility']])
var_fit_full = var_full.fit(optimal_lag)
var_forecast_full = var_fit_full.forecast(df[['Log_Price_USD', 'Log_Exchange_Rate', 'USD_Volatility', 'ZMW_Volatility']].values[-optimal_lag:], steps=forecast_steps)
var_usd_full = np.exp(var_forecast_full[:, 0])
var_zmw_full = np.exp(var_forecast_full[:, 1])

sarima_usd_full = np.exp(model_fit_usd.forecast(steps=forecast_steps))
sarima_zmw_full = np.exp(model_fit_zmw.forecast(steps=forecast_steps))

forecast_usd = 0.5 * var_usd_full + 0.5 * sarima_usd_full
forecast_zmw = 0.5 * var_zmw_full + 0.5 * sarima_zmw_full
forecast_zmw_price = forecast_usd * forecast_zmw

# Forecast index starting from April 2025
forecast_start_date = pd.Timestamp('2025-04-30')
forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_steps, freq='ME')

# Tabs
tab1, tab2, tab3 = st.tabs(["Historical Data", "Model Evaluation", "Forecast"])

with tab1:
    st.subheader("Historical Copper Prices")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df.index, df['Price_USD'], label='Price (USD/ton)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD/ton)')
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df.index, df['Price_ZMW'], label='Price (ZMW/ton)', color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (ZMW/ton)')
    ax2.legend()
    st.pyplot(fig2)

with tab2:
    st.subheader("Model Evaluation (Test Set)")
    st.write(f"SARIMA MAE - USD: {sarima_mae_usd:.2f}, ZMW Exchange Rate: {sarima_mae_zmw:.2f}")
    st.write(f"VAR MAE - USD: {var_mae_usd:.2f}, ZMW Exchange Rate: {var_mae_zmw:.2f}")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(df.index, df['Price_USD'], label='Historical (USD)')
    ax3.plot(test_df.index, var_usd_test, label='VAR Test (USD)', color='purple', linestyle='--')
    ax3.plot(test_df.index, sarima_usd_test, label='SARIMA Test (USD)', color='blue', linestyle='--')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price (USD/ton)')
    ax3.legend()
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(df.index, df['Price_ZMW'], label='Historical (ZMW)', color='green')
    ax4.plot(test_df.index, var_usd_test * test_df['Exchange_Rate'], label='VAR Test (ZMW)', color='purple', linestyle='--')
    ax4.plot(test_df.index, sarima_usd_test * test_df['Exchange_Rate'], label='SARIMA Test (ZMW)', color='blue', linestyle='--')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price (ZMW/ton)')
    ax4.legend()
    st.pyplot(fig4)

with tab3:
    st.subheader(f"Forecast for Next {forecast_steps} Months (Ensemble: 50% VAR, 50% SARIMA)")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.plot(df.index, df['Price_USD'], label='Historical (USD)')
    ax5.plot(forecast_index, forecast_usd, label='Forecast (USD)', color='red')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Price (USD/ton)')
    ax5.legend()
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(df.index, df['Price_ZMW'], label='Historical (ZMW)', color='green')
    ax6.plot(forecast_index, forecast_zmw_price, label='Forecast (ZMW)', color='orange')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Price (ZMW/ton)')
    ax6.legend()
    st.pyplot(fig6)

    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price (USD)': forecast_usd,
        'Forecasted Exchange Rate (ZMW/USD)': forecast_zmw,
        'Forecasted Price (ZMW)': forecast_zmw_price
    })
    st.write("Forecast Table:")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name='copper_price_forecast.csv',
        mime='text/csv'
    )