# 🌍 World News Classification with Deep Learning

This project focuses on classifying **world news** into different categories using various deep learning models. We trained and evaluated multiple architectures such as **CNN**, **LSTM**, **Transformer**, and a **Hybrid model**, and finally deployed the solution through a simple and functional **frontend web interface** for real-time prediction.

---

## 🚀 Project Highlights

- Multiclass classification on global news headlines and content  
- Comprehensive preprocessing pipeline (cleaning, tokenization, padding, etc.)
- Implementation of multiple deep learning architectures:
  - 📚 LSTM (`LSTM.py`)
  - 🧠 CNN (`CNN.py`)
  - 🔗 Hybrid (CNN + LSTM) (`Hybrid.py`)
  - 🧬 Transformer-based model (`Transformer.py`)
- Performance comparison with accuracy, loss curves, and confusion matrices
- Finalized with a user-friendly **web frontend** for live predictions
- Modular codebase for easy training, evaluation, and deployment

---


## 🔬 Model Evaluation

Each model was trained and tested with the same dataset to allow for fair comparison. Evaluation included:

- Accuracy & loss graphs for training vs. validation
- Confusion matrices per model
- Inference performance on unseen data

We found that the **Transformer-based model** showed the best balance between speed and accuracy, while the **Hybrid model** performed strongly on longer news content.

---

## 🌐 Frontend Web Application

After training the models, we developed a simple **web interface** that allows users to enter a news headline and body, and get real-time predictions.

**Features:**
- Minimal and clean interface
- Model is loaded from the backend
- Instant result display with predicted category

The frontend is designed to demonstrate how such a classification system could be used in a real-world application.

---

## ⚙️ Technologies Used

| Area            | Stack                               |
|-----------------|--------------------------------------|
| Programming     | Python 3.x, HTML/CSS/JS              |
| Deep Learning   | TensorFlow, Keras                    |
| NLP             | NLTK, Tokenizer, Padding             |
| Visualization   | Matplotlib, Seaborn                  |
| Frontend        | Flask (or Streamlit if used), JS     |
| Environment     | Google Colab, Jupyter Notebook       |

---

## 🏁 How to Run

1. Clone the repo:

```bash
git clone https://github.com/ilaydaakyuz/NewsClassification.git
cd NewsClassification
```

2. To train or test models, run any of the following:

```bash
python LSTM.py
python CNN.py
python Hybrid.py
python Transformer.py
```

3. To run the web interface (if using Flask):

```bash
cd frontend
python app.py
```

4. Open the browser at `http://localhost:5000` to test predictions live.

---

## 🧪 Sample Input & Output

**Input:**

```
Title: Ekonomide yeni reform paketi açıklandı
Content: Hükümet tarafından açıklanan yeni ekonomik reformlar...
```

**Output:**

```
Predicted Category: Economy
```

---

## 🤝 Contributions

We welcome any kind of contribution! If you want to add a new model, refactor existing code, or improve the frontend, feel free to fork and submit a PR.

---

## 📬 Contact

If you have questions or feedback, please open an issue or contact the maintainer at [GitHub Profile](https://github.com/ilaydaakyuz).
