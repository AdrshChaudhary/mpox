# M-Pox Forecast & Detection Web App

The global outbreak of the virus previously called **Monkeypox**, now referred to as **Mpox**, has caused widespread concern over the past years. In July 2022, the **World Health Organization (WHO)** declared it a *Public Health Emergency of International Concern (PHEIC)*. Timely diagnosis and trend analysis have become critical in tackling the virus’s spread.

To address this dual need, we present an integrated **M-Pox Forecast & Skin Lesion Detection Web Application**. The app combines **deep learning-based image classification** for skin lesion detection with **global case forecasting models** for proactive public health awareness.

---

## 🔬 App Functionalities

### 1. **Mpox Skin Lesion Detection**

Our model uses the **ResNet-50** architecture trained on a curated skin lesion image dataset to distinguish between various pox-type infections, including:

- **Mpox**
- **Chickenpox**
- **Measles**
- **Cowpox**
- **Hand, Foot, and Mouth Disease (HFMD)**
- **Healthy Skin**

The model provides both the predicted disease class and its **confidence score**, allowing users to get a quick and AI-assisted second opinion.

> **Frontend:** Built using **Streamlit** for ease of use and accessibility.  
> **Backend Model:** **ResNet-50**, fine-tuned for medical image classification.

👉 Try it out: [Skin Lesion Detector](https://m-pox-forecast.streamlit.app/)

### 2. **Global Mpox Case Forecasting**

The application also features a **global forecasting dashboard** using **ARIMA** and **Facebook Prophet** time series models, trained on WHO-reported Mpox case data. Users can:

- View historical case data across different countries
- See future predictions of confirmed cases
- Gain insight into trend changes

> **Forecasting Models Used:**
> - ARIMA
> - Facebook Prophet

> **Visualization:** Powered by `Plotly`, `Matplotlib`, and `Seaborn`.

---

## 📁 Dataset

We used the publicly available dataset on Kaggle:  
🔗 [Mpox Skin Lesion Dataset (MSLD V2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20)

### Class Distribution Summary:

| Class Label                   | Image Count | Unique Patients |
|------------------------------|-------------|------------------|
| Mpox                         | 284         | 143              |
