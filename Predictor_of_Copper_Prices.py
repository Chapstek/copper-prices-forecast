# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit app title
st.title("Copper Price Forecasting Tool")

# File upload section
st.header("Upload Excel Files")

# Instructions for the user
st.markdown("""
### Instructions for Uploading Excel Files

To ensure accurate forecasting, please prepare your Excel files as follows before uploading:

1. **File Names**:
   - **Copper Prices**: Name the file `copper_fallback.xlsx` (for copper prices in USD/lb).
   - **Exchange Rates**: Name the file `zm_fallback.xlsx` (for ZMW/USD exchange rates).

2. **Required Columns**:
   - Both files must have at least two columns: `Date` and `Close`.
   - **Date**: The date of the data point (e.g., `2025-03-28`).
   - **Close**: The closing price for copper (in USD/lb) or the exchange rate (ZMW per USD).
   - Other columns (e.g., `Open`, `High`, `Low`) are ignored by the app.

3. **Date Format**:
   - Dates must be in `YYYY-MM-DD` format (e.g., `2025-03-28`).
   - If your dates are in a different format (e.g., `DD/MM/YYYY`), please convert them in Excel:
     - Select the `Date` column, go to `Format > Number > Custom`, and enter `yyyy-mm-dd`.

4. **Align Dates Between Files**:
   - The app matches copper prices and exchange rates by date. Ensure both files have data for the same dates.
   - For example, if `copper_fallback.xlsx` has data from `2025-03-01` to `2025-03-28`, `zm_fallback.xlsx` should cover the same date range.
   - Missing dates in either file will result in an error ("No matching dates found between USD and ZMW data").

5. **Data Values**:
   - **Copper Prices**: Ensure the `Close` column in `copper_fallback.xlsx` contains prices in USD per pound (e.g., `5.15` for $5.15/lb).
   - **Exchange Rates**: Ensure the `Close` column in `zm_fallback.xlsx` contains ZMW/USD exchange rates (e.g., `28.6817` for 28.6817 ZMW per USD).

6. **Where to Acquire Data**:
   - **Copper Prices**: Download historical copper prices from sources like:
     - [MacroTrends](https://www.macrotrends.net/1476/copper-prices-historical-chart-data) (free CSV download of COMEX copper prices).
     - [Investing.com](https://www.investing.com/commodities/copper-historical-data) (free CSV download of copper futures prices).
     - [Trading Economics](https://tradingeconomics.com/commodity/copper) (Excel Add-in for commodity prices).
   - **ZMW/USD Exchange Rates**: Download historical exchange rates from:
     - [Yahoo Finance](https://finance.yahoo.com/quote/ZMWUSD=X/history) (free CSV download of ZMW/USD rates).
     - [Trading Economics](https://tradingeconomics.com/zambia/currency) (Excel Add-in for exchange rates).
     - [OANDA](https://www.oanda.com/currency-converter/en/?from=ZMW&to=USD&amount=1) (historical currency converter).
   - After downloading, format the data as described above and save as `copper_fallback.xlsx` and `zm_fallback.xlsx`.

**Note**: The app expects the last date in your data to be `2025-03-28` for accurate forecasting. Ensure your data includes this date with the correct values (e.g., exchange rate of 28.6817 ZMW/USD on 28/03/2025).
""")

# File upload widgets
usd_file = st.file_uploader("Upload Copper Prices (USD/lb) Excel File", type=["xlsx"])
zmw_file = st.file_uploader("Upload ZMW/USD Exchange Rates Excel File", type=["xlsx"])

# Initialize data storage
usd_data = None
zmw_data = None
exchange_rates = None

# Function to parse dates
def parse_date(date_value):
    try:
        if isinstance(date_value, (int, float)) and 40000 < date_value < 50000:
            excel_epoch = datetime(1899, 12, 30)
            return (excel_epoch + timedelta(days=date_value)).strftime("%Y-%m-%d")
        return pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except:
        return None

# Function to format date as DD/MM/YYYY
def format_date_ddmmyyyy(date_str):
    date = pd.to_datetime(date_str)
    return date.strftime("%d/%m/%Y")

# Process uploaded files
if usd_file and zmw_file:
    # Read Excel files
    usd_df = pd.read_excel(usd_file)
    zmw_df = pd.read_excel(zmw_file)

    # Process USD data
    usd_data = []
    for _, row in usd_df.iterrows():
        date = parse_date(row["Date"])
        close = float(row["Close"]) if pd.notnull(row["Close"]) else None
        if date and close is not None:
            usd_data.append({"date": date, "close": close})

    # Process ZMW data
    zmw_data = []
    for _, row in zmw_df.iterrows():
        date = parse_date(row["Date"])
        close = float(row["Close"]) if pd.notnull(row["Close"]) else None
        if date and close is not None:
            zmw_data.append({"date": date, "close": close})

    # Align exchange rates with USD data dates
    exchange_rates = []
    for usd_row in usd_data:
        zmw_row = next((zmw for zmw in zmw_data if zmw["date"] == usd_row["date"]), None)
        if zmw_row:
            exchange_rates.append({"date": usd_row["date"], "rate": zmw_row["close"]})

    # Display the latest exchange rate
    if exchange_rates:
        latest_rate = exchange_rates[-1]["rate"]
        latest_date = exchange_rates[-1]["date"]
        formatted_date = format_date_ddmmyyyy(latest_date)
        st.header("Latest ZMW/USD Exchange Rate")
        st.write(f"as of {formatted_date}: {latest_rate:.4f}")
    else:
        st.error("No matching dates found between USD and ZMW data.")

    # Plot historical data
    if usd_data and exchange_rates:
        dates = [row["date"] for row in usd_data]
        usd_prices = [row["close"] for row in usd_data]
        zmw_prices = [usd_data[i]["close"] * rate["rate"] for i, rate in enumerate(exchange_rates)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=usd_prices, mode="lines", name="Copper Price (USD/lb)", line=dict(color="#007bff")))
        fig.add_trace(go.Scatter(x=dates, y=zmw_prices, mode="lines", name="Copper Price (ZMW/lb)", line=dict(color="#ff5733"), yaxis="y2"))
        fig.update_layout(
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price (USD/lb)", side="left"),
            yaxis2=dict(title="Price (ZMW/lb)", side="right", overlaying="y"),
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig)

# Forecasting section
st.header("Forecast Copper Prices")
forecast_days = st.selectbox("Select number of days to forecast:", list(range(1, 31)))  # Supports up to 30 days
if st.button("Forecast"):
    if not usd_data or not zmw_data or not exchange_rates:
        st.error("Please upload both USD and ZMW files before forecasting.")
    else:
        # Forecast copper prices in USD
        prices = [d["close"] for d in usd_data]
        rates = [r["rate"] for r in exchange_rates]
        window_size = 3

        if len(prices) < window_size or len(rates) < window_size:
            st.error("Not enough data points for forecasting. Need at least 3 data points.")
        else:
            # Calculate trends
            last_three_prices = prices[-window_size:]
            last_price_average = sum(last_three_prices) / window_size

            last_three_rates = rates[-window_size:]
            last_rate_average = sum(last_three_rates) / window_size

            rate_trend = 0
            if len(rates) >= 2:
                last_rate = rates[-1]
                second_last_rate = rates[-2]
                rate_trend = (last_rate - second_last_rate) / second_last_rate if second_last_rate != 0 else 0

            # Forecast
            forecasts_usd = []
            forecasts_rates = []
            for i in range(forecast_days):
                price_adjustment = 1 + (rate_trend * (i + 1))
                forecasted_price = last_price_average * price_adjustment
                forecasts_usd.append(forecasted_price)

                recent_prices = prices[-(window_size-1):] + [forecasted_price]
                last_price_average = sum(recent_prices) / window_size

                rate_adjustment = 1 + (rate_trend * (i + 1))
                forecasted_rate = last_rate_average * rate_adjustment
                forecasts_rates.append(forecasted_rate)

                recent_rates = rates[-(window_size-1):] + [forecasted_rate]
                last_rate_average = sum(recent_rates) / window_size

            # Convert to ZMW using forecasted rates
            forecasts_zmw = [forecasts_usd[i] * forecasts_rates[i] for i in range(forecast_days)]

            # Generate forecast dates
            last_date = datetime.strptime("2025-03-28", "%Y-%m-%d")
            forecast_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            forecast_dates_formatted = [format_date_ddmmyyyy(date) for date in forecast_dates]

            # Display forecasts
            st.header("Forecasted Prices")
            for i in range(forecast_days):
                st.write(
                    f"Day {i+1}: {forecast_dates_formatted[i]} - ${forecasts_usd[i]:.2f} USD/lb | "
                    f"ZMW {forecasts_zmw[i]:.2f}/lb | Exchange Rate: {forecasts_rates[i]:.4f} (ZMW/USD)"
                )

            # Update chart with forecasts
            dates = [row["date"] for row in usd_data] + forecast_dates
            usd_prices = [row["close"] for row in usd_data] + forecasts_usd
            zmw_prices = [usd_data[i]["close"] * rate["rate"] for i, rate in enumerate(exchange_rates)] + forecasts_zmw

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=usd_prices, mode="lines", name="Copper Price (USD/lb)", line=dict(color="#007bff")))
            fig.add_trace(go.Scatter(x=dates, y=zmw_prices, mode="lines", name="Copper Price (ZMW/lb)", line=dict(color="#ff5733"), yaxis="y2"))
            fig.update_layout(
                xaxis=dict(title="Date"),
                yaxis=dict(title="Price (USD/lb)", side="left"),
                yaxis2=dict(title="Price (ZMW/lb)", side="right", overlaying="y"),
                legend=dict(x=0, y=1.1, orientation="h")
            )
            st.plotly_chart(fig)

# Signature
st.markdown("---")
st.markdown("[Developed by Darwin Chapuswike] (https://darwinchapuswike.com)", unsafe_allow_html=True)