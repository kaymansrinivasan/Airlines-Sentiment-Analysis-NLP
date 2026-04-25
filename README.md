# ✈️ Airline Sentiment Analysis using NLP & BERT

## 📖 Overview
This project focuses on analyzing airline customer reviews to classify sentiment into **Negative, Neutral, and Positive** using Natural Language Processing (NLP).

The goal is to compare **traditional machine learning models** with **advanced transformer-based models (BERT)** and understand where improvements actually occur.

---

## 🎯 Objectives
- Perform data cleaning and preprocessing on real-world airline reviews
- Conduct exploratory data analysis (EDA)
- Build baseline NLP models using TF-IDF
- Implement advanced deep learning using BERT
- Compare model performance and derive insights

---

## 📂 Dataset
- Source: Airline reviews dataset
- Size: ~22,000 reviews
- Features include:
  - Review text
  - Rating
  - Airline name
  - Service-related attributes (seat comfort, staff service, etc.)

---

## 🧹 Data Cleaning
- Removed unnecessary columns (e.g., index columns)
- Handled missing values:
  - Dropped critical missing rows (ratings, reviews)
  - Imputed categorical values as `"Unknown"`
  - Filled numerical features with median
- Converted:
  - Ratings → numeric
  - Dates → datetime
- Created new features:
  - `sentiment` (from rating)
  - `label` (encoded sentiment)
  - `full_text` (title + review)

---

## 📊 Exploratory Data Analysis (EDA)
Key insights:
- Dataset is **highly imbalanced** (majority negative reviews)
- Certain airlines consistently show higher negative sentiment
- Strong correlation:
  - **Value for Money (0.55)** with overall rating
- Common complaint patterns:
  - "flight delayed"
  - "customer service"
  - "worst experience"

---

## 🤖 Models Implemented

### 🔹 Baseline Models
- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes

### 🔹 Advanced Model
- DistilBERT (Transformer-based model)
- Fine-tuned for 3-class sentiment classification

---

## 📈 Results

### Logistic Regression
| Metric | Score |
|------|------|
| Accuracy | 0.75 |
| Neutral F1 | 0.27 |

---

### BERT
| Metric | Score |
|------|------|
| Accuracy | 0.82 |
| Neutral F1 | 0.26 |

---

## 🔍 Key Findings

### ✅ Improvements with BERT
- Higher overall accuracy
- Better performance on **positive and negative classes**
- Improved contextual understanding

---

### ⚠️ Limitation
- **Neutral class performance did not improve significantly**
  - Logistic Regression: 0.27
  - BERT: 0.26

👉 This suggests:
> The limitation lies in **data ambiguity**, not model capability.

---

## 🧠 Insight (Important)

Neutral reviews often contain:
- Mixed sentiment
- Weak polarity signals

Even advanced models struggle due to:
- Overlapping class boundaries
- Labeling strategy based on rating thresholds

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Transformers (Hugging Face)
- Matplotlib, Seaborn

---

## 📁 Project Structure
