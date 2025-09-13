# 📰 Fake News Detection Using Machine Learning

A Machine Learning project to classify news as **Fake** or **Real** using Natural Language Processing (NLP) techniques and a Streamlit web interface.

---

## 📖 Project Overview
This project uses a dataset of **True** and **Fake** news articles from Kaggle to train a model that detects fake news based purely on writing style and text patterns.  
⚠️ **Disclaimer:** This model does **not verify facts**, it only analyzes textual patterns.

---

## 🚀 Features
- Preprocessing with tokenization, stopword removal, and lemmatization.
- TF-IDF vectorization for text representation.
- Hyperparameter tuning using RandomizedSearchCV.
- Interactive **Streamlit Web App** for real-time predictions.
- Outputs both prediction and **confidence score**.

---

## 🏗️ Tech Stack
- **Python 3.8+**
- **Libraries:** pandas, numpy, nltk, scikit-learn, matplotlib, seaborn, joblib, streamlit

---

## 2️⃣ Install Dependencies
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download Dataset
Place the Kaggle **True.csv** and **Fake.csv** files in the project directory.

---

## 4️⃣ Train the Model (Optional)
```bash
python main.py
```

This will generate:
- `Vectorizer.pkl`  
- `Fake_News_Detector_Model.pkl`

---

## 5️⃣ Run the Streamlit App
```bash
streamlit run user_interface.py
```

Open your browser at **http://localhost:8501/**.

---

## 🖼️ App Preview
The Streamlit app allows users to enter:
- **News Headline**  
- **News Body**

It returns:
- **Prediction:** 🛑 Fake or ✅ Real  
- **Confidence Score** (percentage)

---

## 📊 Model Performance
- TF-IDF + Logistic Regression / Random Forest
- Metrics achieved:
  - **Accuracy:** ~95%  
  - **Precision, Recall, F1-score**: High performance on both classes.

---

## 📌 Future Improvements
- Experiment with advanced NLP models (e.g., BERT, LSTM).  
- Add multilingual support.  
- Include live fact-check APIs for verification.  
- Build a Docker image for easy deployment.

---

## 👤 Author
**Saif Ullah Umar**  
- 📧 [GitHub Profile](https://github.com/SaifUllahUmar0317)  
- 💼 Machine Learning Enthusiast

---

## 📝 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute this code for educational and commercial purposes.
