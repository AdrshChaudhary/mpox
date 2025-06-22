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
| Chickenpox                   | 75          | 62               |
| Measles                      | 55          | 46               |
| Cowpox                       | 66          | 41               |
| HFMD                         | 161         | 144              |
| Healthy                      | 114         | 105              |
| **Total**                    | **755**     | **541**          |

### Sample Images  
![Sample Images](https://github.com/AdrshChaudhary/mpox/blob/main/assets/samples.jpg)

---

## 🧠 Model Architecture (Lesion Detection)

The app utilizes **ResNet-50**, a deep CNN known for its skip connections and ability to prevent vanishing gradients in deep layers. We fine-tuned it on the skin lesion dataset using **data augmentation** and **transfer learning** for better generalization.

### Model Pipeline

![Working Pipeline](https://github.com/AdrshChaudhary/mpox/blob/main/assets/GA_aug.jpeg)

---

## 🖥️ Screenshots

### 🔍 Lesion Detection Interface
![Interface](https://github.com/AdrshChaudhary/mpox/blob/main/assets/Screenshot%202023-12-10%20202710.png)

### ✅ Prediction Result
![Prediction](https://github.com/AdrshChaudhary/mpox/blob/main/assets/Screenshot%202023-12-10%20202754.png)

---

### 📈 Global Mpox Forecasting Dashboard

#### 🌍 Country-wise Case Selection
![Country Select](https://github.com/AdrshChaudhary/mpox/blob/main/assets/country.PNG)

#### 📊 ARIMA Forecasting Graph
![ARIMA Forecast](https://github.com/AdrshChaudhary/mpox/blob/main/assets/arima.PNG)

#### 📉 Prophet Model Forecast
![Prophet Forecast](https://github.com/AdrshChaudhary/mpox/blob/main/assets/prophet.PNG)

---

## 🚀 Live Application

🔗 **Access the web app here**: [https://m-pox-forecast.streamlit.app/](https://m-pox-forecast.streamlit.app/)

Whether you're a healthcare professional, researcher, or simply a concerned user, this app helps with **instant detection** and **epidemic trend monitoring** — all in one place.

---

## 👨‍💻 Author

**Aadarsh Chaudhary**  
- 🌐 [Portfolio](https://aadrsh.netlify.app/)  
- 💼 [LinkedIn](https://www.linkedin.com/in/aadarshchaudhary/)  
- 💻 [GitHub](https://github.com/AdrshChaudhary)
