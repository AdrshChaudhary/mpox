import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from PIL import Image
import warnings
import os

warnings.filterwarnings("ignore")

# Streamlit page config
st.set_page_config(page_title="🩺 Skin & M-pox App", layout="wide")
st.title("🩺 Unified Health Intelligence Dashboard")

# Tabs for navigation
tab1, tab2 = st.tabs(["🖼️ Skin Disease Classification", "📈 M-pox Forecasting"])

# ---------------- TAB 1: Skin Disease Classification ------------------
with tab1:
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                            content:'This app is in its early stage. We recommend you to seek professional advice from a dermatologist. Thank you.'; 
                            visibility: visible;
                            display: block;
                            position: relative;
                            #background-color: red;
                            padding: 5px;
                            top: 2px;
                        }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    #Add CSS styling for center alignment
    st.markdown(
        """
        <style>
        .center {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Load the pre-trained model
    model_path = os.path.join("models", "model.h5")
    model = tf.keras.models.load_model(model_path)
    # Define the labels for the categories
    labels = {
        0: 'Chickenpox',
        1: 'Cowpox',
        2: 'HFMD',
        3: 'Healthy',
        4: 'Measles',
        5: 'MPOX'
    }
    # Function to preprocess the image
    def preprocess_image(image):
        # Resize the image to 224x224
        image = image.resize((224, 224))
        # Convert the image to a numpy array
        image_array = np.array(image)
        # Normalize the image array
        # image_array = image_array / 255.0
        # Add an extra dimension to match the model's input shape
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    # Function to make predictions
    def predict(image):
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make the prediction
        prediction = model.predict(processed_image)
        # Get the predicted label index
        label_index = np.argmax(prediction)
        # Get the predicted label
        predicted_label = labels[label_index]
        # Get the confidence level
        confidence = prediction[0][label_index] * 100
        return predicted_label, confidence
    
    # Center align the heading
    st.markdown("<h1 class='center'>Skin Lesion Classifier</h1>", unsafe_allow_html=True)
    # Display a file uploader widget
    number = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])
    if number == 'Capture by camera':
        uploaded_file = st.camera_input("Take a picture")
    else:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Process and classify the image
        predicted_label, confidence = predict(image)
        # Display the predicted label and confidence
        st.markdown("<h3 class='center'>This might be:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 class='center'>{predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='center'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

# ---------------- TAB 2: M-pox Forecast Dashboard ------------------
with tab2:
    st.header("📈 M-pox Global Case Forecasting Dashboard")

    # Load datasets
    df_cases = pd.read_csv("data/Daily_Country_Wise_Confirmed_Cases.csv")
    df_summary = pd.read_csv("data/Monkey_Pox_Cases_Worldwide.csv")
    df_worldwide = pd.read_csv("data/Worldwide_Case_Detection_Timeline.csv")

    df_worldwide.columns = df_worldwide.columns.str.strip().str.lower().str.replace(' ', '_')

    # Reshape daily cases
    df_long = df_cases.melt(id_vars='Country', var_name='Date', value_name='Cases')
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long['Cases'] = pd.to_numeric(df_long['Cases'], errors='coerce').fillna(0)

    st.sidebar.header("⚙️ Forecasting Settings")
    countries = sorted(df_long['Country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    model_choice = st.sidebar.radio("Model", ["ARIMA (Auto)", "Prophet"])
    forecast_days = st.sidebar.slider("Days to Forecast", 7, 180, 30)

    country_df = df_long[df_long['Country'] == selected_country]
    country_df = country_df.groupby('Date')['Cases'].sum().reset_index()
    summary = df_summary[df_summary['Country'] == selected_country]

    st.subheader(f"📊 Summary for {selected_country}")
    if not summary.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confirmed", f"{int(summary['Confirmed_Cases'].values[0])}")
        col2.metric("Suspected", f"{int(summary['Suspected_Cases'].values[0])}")
        col3.metric("Hospitalized", f"{int(summary['Hospitalized'].values[0])}")
        col4.metric("Travel History", f"{int(summary['Travel_History_Yes'].values[0])}")
    else:
        st.warning("No summary data available.")

    def run_arima(df, forecast_days):
        ts = df.set_index('Date')['Cases']
        model = ARIMA(ts, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        last_date = ts.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
        return forecast_df, ts[-forecast_days:], forecast
    
    def run_prophet(df, forecast_days):
        prophet_df = df.rename(columns={'Date': 'ds', 'Cases': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
        return forecast_df, prophet_df['y'][-forecast_days:], forecast['yhat'][-forecast_days:]

    try:
        if model_choice == "ARIMA (Auto)":
            forecast_df, y_true, y_pred = run_arima(country_df, forecast_days)
            ci_low, ci_high = None, None
        else:
            forecast_df, y_true, y_pred = run_prophet(country_df, forecast_days)
            ci_low, ci_high = forecast_df['yhat_lower'], forecast_df['yhat_upper']

        # mae = mean_absolute_error(y_true, y_pred)
        # rmse = mean_squared_error(y_true, y_pred)
        # st.success(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        plot_df = country_df.copy()
        plot_df['Type'] = 'Actual'
        forecast_df = forecast_df.rename(columns={'Forecast': 'Cases'})
        forecast_df['Type'] = 'Forecast'
        combined_df = pd.concat([plot_df, forecast_df])

        st.subheader("📈 Historical & Forecasted Cases")
        fig = px.line(combined_df, x='Date', y='Cases', color='Type',
                      title=f"{selected_country} M-pox Forecast ({model_choice})")

        if ci_low is not None:
            fig.add_traces([
                go.Scatter(x=forecast_df['Date'], y=ci_low, mode='lines',
                           line=dict(width=0), showlegend=False),
                go.Scatter(x=forecast_df['Date'], y=ci_high, mode='lines',
                           line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False)
            ])

        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download Forecast", forecast_df.to_csv(index=False),
                           file_name=f"{selected_country.lower()}_forecast.csv")

    except Exception as e:
        st.error(f"Forecast failed: {e}")

    # Global stats and word cloud
    global_stats = df_summary[['Confirmed_Cases', 'Suspected_Cases', 'Hospitalized']].sum()
    st.markdown("### 🌐 Global Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Confirmed", int(global_stats['Confirmed_Cases']))
    col2.metric("Total Suspected", int(global_stats['Suspected_Cases']))
    col3.metric("Total Hospitalized", int(global_stats['Hospitalized']))

    st.markdown("### ☁️ Word Cloud of Country Mentions")
    country_freq = df_summary.set_index("Country")["Confirmed_Cases"].to_dict()
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(country_freq)
    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)

    st.markdown("### 🗺️ Choropleth Map")
    fig_map = px.choropleth(df_summary, locations="Country", locationmode="country names",
                            color="Confirmed_Cases", hover_name="Country",
                            color_continuous_scale="Reds")
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### ⏱️ First Detection Timeline")
    date_col = next((col for col in df_worldwide.columns if 'date' in col), None)
    country_col = next((col for col in df_worldwide.columns if 'country' in col), None)
    if date_col and country_col:
        df_worldwide[date_col] = pd.to_datetime(df_worldwide[date_col])
        fig_detect = px.scatter(df_worldwide, x=date_col, y=country_col,
                                title="Timeline of First Detection by Country")
        st.plotly_chart(fig_detect)
    else:
        st.error("Missing 'date' or 'country' column in detection dataset.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub](https://github.com/AdrshChaudhary/mpox) | [Contact](mailto:im.aadrsh@gmail.com)")