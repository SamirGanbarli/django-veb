# 🧠 Multi-Module Web Mining Application with Python & Django

This project is a multi-functional web mining application built using **Python** and **Django**, integrating three main modules: **Sentiment Analysis**, **Topic Modeling**, and **Wikipedia Scraping**.

## 🚀 Features

### 1. 🔍 Sentiment Analysis
Automatically detects **positive**, **negative**, and **neutral** sentiments in user-generated content such as social media posts and product reviews.

- **Model Options**:
  - Naive Bayes
  - K-Means Clustering
  - Transformer-based (`CardiffNLP/twitter-roberta-base-sentiment`)

- **Output**:
  - Pie chart of sentiment distribution
  - Classification report (if ground truth labels are provided)

### 2. 🧾 Topic Modeling
Discovers the **main themes and topics** in large text datasets using **NMF (Non-negative Matrix Factorization)**.

- **Input**: Dataset with a text column
- **Output**: Table of dominant topics and associated keywords

### 3. 🌐 Wikipedia Scraping
Fetches **summarized information** from Wikipedia based on a user-provided keyword.

- Uses the `wikipedia` Python package
- Provides clean and concise topic summaries for downstream analysis

---

## 🗂 Datasets

- Sourced from **Kaggle**
- Includes user reviews from platforms like **Amazon** and **Starbucks**
- Preprocessing steps:
  - Missing value handling
  - Text cleaning & lowercasing
  - TF-IDF vectorization for classical models
  - Transformer models use internal tokenization and embedding

---

## 🛠️ System Architecture

- **Backend**: Django (Python)
- **Frontend**: Django Template Language (DTL)
- **Modules**:
  - Integrated within Django views
  - Modular structure for easy extension and maintenance

---

## 💡 Usage Scenarios

### Sentiment Analysis
1. Select model from dropdown.
2. Upload dataset and specify:
   - Text column name
   - (Optional) Label column name
3. Click **"Analyze Sentiment"**
4. View sentiment pie chart and (if labels provided) classification report.

### Topic Modeling
1. Upload dataset.
2. Specify text column name.
3. Click **"Upload and Analyze"**
4. View table of discovered topics and keywords.

### Wikipedia Scraping
1. Enter topic or keyword.
2. Click **"Scrape Wikipedia"**
3. View fetched summary content.

---

## ⚠️ Known Issues

- Datasets from various sources may have different structures.
- Manual column name input is required—invalid input triggers error messages.

---

## 📌 Planned Improvements

- Add export/download option for scraped Wikipedia summaries in structured formats (e.g., `.txt`, `.csv`, `.json`).

---
## 📚 Tech Stack

- Python
- Django
- scikit-learn (Naive Bayes, K-Means, NMF)
- Transformers (`HuggingFace`)
- Wikipedia API
- TF-IDF, preprocessing via `sklearn` and `nltk`

---

## 👨‍💻 Author

**Samir Ganbarli**  
This project was developed as part of my work in web mining and applied NLP.

---

## 📄 License

[MIT](LICENSE)

