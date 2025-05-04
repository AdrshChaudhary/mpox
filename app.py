import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Streamlit config
st.set_page_config(page_title="🦮 M-pox Forecast Dashboard", layout="wide")
st.title("🦮 M-pox Case Forecasting Dashboard")
st.markdown("""
This interactive dashboard allows you to explore historical M-pox case trends and forecast future cases.
- Select a **country**
- Choose a **forecasting model**
- View dynamic **summary statistics**
- Explore **interactive forecast plots and global insights**
""")

# Load datasets
df_cases = pd.read_csv("data/Daily_Country_Wise_Confirmed_Cases.csv")
df_summary = pd.read_csv("data/Monkey_Pox_Cases_Worldwide.csv")
df_worldwide = pd.read_csv("data/Worldwide_Case_Detection_Timeline.csv")

# Clean column names for df_worldwide
df_worldwide.columns = df_worldwide.columns.str.strip().str.lower().str.replace(' ', '_')

# Reshape daily cases to long format
df_long = df_cases.melt(id_vars='Country', var_name='Date', value_name='Cases')
df_long['Date'] = pd.to_datetime(df_long['Date'])
df_long['Cases'] = pd.to_numeric(df_long['Cases'], errors='coerce').fillna(0)

# Sidebar selections
st.sidebar.header("⚙️ Settings")
countries = sorted(df_long['Country'].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)
model_choice = st.sidebar.radio("Forecasting Model", ["ARIMA", "Prophet"])
forecast_days = st.sidebar.slider("Days to Forecast", 7, 180, 30)

# Filter data by country
country_df = df_long[df_long['Country'] == selected_country]
country_df = country_df.groupby('Date')['Cases'].sum().reset_index()

# Summary stats
summary = df_summary[df_summary['Country'] == selected_country]

st.markdown("### 📊 Summary Statistics for: **{}**".format(selected_country))
if not summary.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🦠 Confirmed Cases", f"{int(summary['Confirmed_Cases'].values[0])}")
    col2.metric("🪒 Suspected Cases", f"{int(summary['Suspected_Cases'].values[0])}")
    col3.metric("🏥 Hospitalized", f"{int(summary['Hospitalized'].values[0])}")
    col4.metric("✈️ Travel History (Yes)", f"{int(summary['Travel_History_Yes'].values[0])}")
else:
    st.warning("No summary data available for this country.")

# Forecasting functions
def run_arima(df, forecast_days):
    ts = df.set_index('Date')['Cases']
    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    last_date = ts.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    return forecast_df

def run_prophet(df, forecast_days):
    prophet_df = df.rename(columns={'Date': 'ds', 'Cases': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})

# Forecasting
if model_choice == "ARIMA":
    forecast_df = run_arima(country_df, forecast_days)
else:
    forecast_df = run_prophet(country_df, forecast_days)

# Merge actual and forecasted data
plot_df = country_df.copy()
plot_df['Type'] = 'Actual'
forecast_df['Type'] = 'Forecast'
forecast_df = forecast_df.rename(columns={'Forecast': 'Cases'})
combined_df = pd.concat([plot_df, forecast_df])

# Plotting Forecast
st.subheader(f"📈 Historical & Forecasted M-pox Cases in {selected_country}")
fig = px.line(combined_df, x='Date', y='Cases', color='Type',
              title=f"M-pox Cases Forecast using {model_choice}",
              template='plotly_white')
fig.update_layout(height=500, margin=dict(t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# Download Forecast
st.markdown("### 📅 Download Forecast")
st.download_button(
    label="Download Forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f"{selected_country.lower().replace(' ', '_')}_{model_choice.lower()}_forecast.csv",
    mime="text/csv"
)

# 🌎 Global Overview Section
global_stats = df_summary[['Confirmed_Cases', 'Suspected_Cases', 'Hospitalized']].sum()
st.markdown("### 🌐 Global Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Confirmed", int(global_stats['Confirmed_Cases']))
col2.metric("Total Suspected", int(global_stats['Suspected_Cases']))
col3.metric("Total Hospitalized", int(global_stats['Hospitalized']))

# 🔍 Word Cloud for Country Mentions
st.markdown("### ☁️ Country Mentions Word Cloud")
country_freq = df_summary.set_index("Country")["Confirmed_Cases"].to_dict()
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(country_freq)
fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig_wc)

# 🌍 Choropleth Map of Confirmed Cases
st.markdown("### 🌏 Confirmed Cases by Country")
fig_map = px.choropleth(df_summary, locations="Country", locationmode="country names",
                        color="Confirmed_Cases", hover_name="Country",
                        color_continuous_scale="Reds", title="Confirmed M-pox Cases by Country")
fig_map.update_layout(height=600)
st.plotly_chart(fig_map, use_container_width=True)

# ⏱️ First Detection Timeline
st.markdown("### ⏱️ First Detection Timeline")
# Detect date and country columns
date_col = next((col for col in df_worldwide.columns if 'date' in col), None)
country_col = next((col for col in df_worldwide.columns if 'country' in col), None)

if date_col and country_col:
    df_worldwide[date_col] = pd.to_datetime(df_worldwide[date_col])
    fig_detect = px.scatter(df_worldwide, x=date_col, y=country_col,
                            title="Timeline of First Detection by Country",
                            template='plotly_white')
    fig_detect.update_layout(height=600)
    st.plotly_chart(fig_detect, use_container_width=True)
else:
    st.error("Could not find appropriate 'date' or 'country' column in Worldwide Detection data.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub](https://github.com/) | [Contact](mailto:you@example.com)")