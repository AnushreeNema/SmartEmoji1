# 🏆 SmartEmoji — Context-Aware Emoji Recommendation System

🥇 **Winner — UCI Heist Datathon 2026 (Best Use of StrataScratch Dataset)** *(completed solo)*

---

## 🌐 Overview

SmartEmoji is an NLP-powered **emoji recommendation system** that suggests the most relevant emojis for a given piece of text.

Unlike traditional approaches that treat emoji prediction as a **single-label classification problem**, SmartEmoji reframes the task as a **ranking/recommendation problem**, recognizing that multiple emojis can be valid depending on context.

👉 Instead of predicting a single “correct” emoji, the system returns the **top-3 most relevant emojis with confidence scores**, improving real-world usability.

---

## 🚀 Product

SmartEmoji is built as an **end-to-end product**, not just a model.

### ✨ Features
- 📝 Input text → receive **top-3 emoji suggestions instantly**
- 🧠 Context-aware predictions using a fine-tuned transformer model  
- 📊 Confidence scores for each recommendation  
- 😊 Sentiment-aware re-ranking for improved relevance  
- ⚡ Designed for real-world usage (chat, social media, UI integrations)

### 💻 Interface
- Web-based UI for real-time interaction  
- Clean input/output flow for emoji recommendations  

---

## 💡 Key Insight

Emoji prediction is inherently **ambiguous and subjective**.

- The same text can map to multiple valid emojis  
- Datasets often assign only one label → limiting  
- Traditional evaluation (top-1 accuracy) is misleading  

### ✅ Solution
- Reframed task from **classification → recommendation system**
- Evaluated using **top-k accuracy (k=3)** instead of top-1

---

## 🏗️ Approach

### 1. Baselines
- TF-IDF + Logistic Regression  
- TF-IDF + Random Forest  

### 2. Final Model
- Fine-tuned **BERTweet** (optimized for social media text)

### 3. Enhancements
- Sentiment-aware re-ranking  
- Class imbalance correction  
- Probability-based ranking  

---

## 📊 Evaluation

| Metric        | Description                          |
|--------------|--------------------------------------|
| Top-1 Accuracy | Traditional (less reliable)          |
| Top-3 Accuracy | Captures real-world relevance (used) |

👉 Many “incorrect” predictions were actually **semantically valid alternatives**

---


---

## 🛠️ Tech Stack

- **Backend:** Python, PyTorch, HuggingFace Transformers  
- **Model:** BERTweet (fine-tuned)  
- **Frontend:** React / Next.js  
- **NLP:** Tokenization, Embeddings, Sentiment Analysis  
- **Other:** Probability Calibration, Ranking Logic  

---

## 🚧 Challenges

- ❗ Ambiguous labels (multiple valid emojis per text)  
- ⚖️ Class imbalance (frequent emojis dominate)  
- 🧵 Noisy, short-form tweet data  
- 📉 Misleading traditional evaluation metrics  

---

## 🏆 Accomplishments

- 🥇 Winner — UCI Heist Datathon 2026 *(Best Use of StrataScratch Dataset)*  
- Built an **end-to-end system (model + evaluation + product)**  
- Introduced **top-k evaluation** aligned with real-world usage  
- Developed a **recommendation-based approach** instead of strict classification  

---

## 📚 What I Learned

- Problem formulation > model complexity  
- Evaluation should reflect **real user behavior**  
- Many “errors” in ML are actually **valid alternatives**

---

## 🔮 Future Work

- Multi-label training setup  
- Real-time emoji suggestions while typing  
- Personalized recommendations  
- Improved sentiment modeling  

---
## 👤 Author

**Anushree Nema**  
[https://www.linkedin.com/in/anushree-nema-609ab71b4/]  
[anushreenema624@gmail.com]