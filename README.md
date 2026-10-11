# Text Analytics, Natural Language Processing & Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Natural Language Processing (NLP) and Sentiment Analysis framework implemented across two core investigation tracks:
- **Part A:** Linguistic foundations, comparative tokenization, morphological stemming/lemmatization, POS tagging, Named Entity Recognition (NER), and statistical N-Gram language modeling with Laplace smoothing.
- **Part B:** Supervised end-to-end sentiment classification on large-scale text reviews utilizing TF-IDF feature extraction, Grid Search hyperparameter optimization, and comparative benchmarking across multiple ML classifiers.

---

## 📌 Project Overview

### Part A: Text Preprocessing, NLP Pipelines & Statistical Language Models
* **Tokenization Benchmarking:** Comparative empirical evaluation of whitespace `split()`, RegEx patterns (`re.findall`), and Penn Treebank tokenization (`nltk.word_tokenize`).
* **Morphological Analysis:** Stemming algorithms (Porter vs. Snowball vs. Lancaster) benchmarked against WordNet Lemmatization with POS context awareness.
* **Grammatical & Entity Extraction:** Part-of-Speech tagging using the Averaged Perceptron Tagger and Named Entity Recognition (NER) via NLTK Maximum Entropy chunking.
* **Statistical Language Modeling:** Unigram, Bigram, and Trigram Maximum Likelihood Estimation (MLE) with Add-1 (Laplace) smoothing and Perplexity evaluation for synthetic text generation.

### Part B: Supervised Sentiment Analysis & Classification
* **Text Normalization:** Stopword removal, contraction expansion, noise filtering, and stemming.
* **Feature Representation:** Term Frequency-Inverse Document Frequency (TF-IDF) n-gram vectorization with sublinear scaling.
* **Classifier Benchmarking:** Systematic evaluation across:
  * Multinomial Naive Bayes (`MultinomialNB`)
  * Logistic Regression (`LogisticRegression`)
  * Linear Support Vector Classification (`LinearSVC`)
  * Random Forest Ensemble (`RandomForestClassifier`)
* **Optimization & Diagnostic Evaluation:** `GridSearchCV` hyperparameter tuning, Confusion Matrices, Precision-Recall trade-offs, and Word Cloud frequency visualizations.

---

## 🗂️ Repository Structure

```text
text-analytics-sentiment-analysis-nlp/
├── Part_A_TXSA.ipynb    # Linguistic preprocessing, POS, NER & N-gram language models
├── Part_B_TXSA.ipynb    # TF-IDF feature extraction, model tuning & sentiment classification
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes local checkpoints, datasets & bytecode
└── README.md            # Comprehensive project documentation
```

---

## 🔬 Key Methodologies & Formulations

### 1. TF-IDF Representation
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log \left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2. N-Gram Language Modeling with Add-1 (Laplace) Smoothing
$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebooks
```bash
jupyter notebook
```
Open either `Part_A_TXSA.ipynb` or `Part_B_TXSA.ipynb` to inspect and execute the pipelines.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
