[README_3.md](https://github.com/user-attachments/files/29213223/README_3.md)
# NLP News Intelligence System

A multi-paradigm NLP project on the **AG News** dataset that combines classical machine learning, neural sentence embeddings, and unsupervised topic modeling into one news classification and semantic search system.

The central finding: **classical TF-IDF features win at classification, while neural embeddings win at semantic search** — so the right system uses both.

## Overview

- **Dataset:** AG News — 120,000 training and 7,600 test articles
- **Categories:** World, Sports, Business, Sci/Tech
- **Paradigms compared:** Classical ML (TF-IDF), neural embeddings (Sentence-BERT), unsupervised (LDA)
- **Tasks:** text classification, semantic retrieval, topic discovery

## Results

### Classification (test set, 7,600 articles)

| Model | Accuracy | Paradigm |
|---|---|---|
| **SVM + TF-IDF** | **91.16%** | Classical |
| Logistic Regression + TF-IDF | 90.87% | Classical |
| Logistic Regression + SBERT | 88.58% | Neural |

The best classifier was a Linear SVM on TF-IDF features. TF-IDF beat the (non-fine-tuned) SBERT embeddings by ~2.6 points. AG News categories turn out to be lexically separable — discriminative keywords like *basketball*, *stocks*, and *NASA* carry most of the signal, which sparse TF-IDF captures directly.

### Semantic search (Precision@5)

| Method | Precision@5 |
|---|---|
| **SBERT retrieval** | **0.844** |
| TF-IDF retrieval | 0.776 |

For retrieval, Sentence-BERT improved Precision@5 by **8.8%**. TF-IDF breaks on vocabulary mismatch (the query *"Apple launches AI chip"* pulls in Beatles articles because of the word *Apple*), while SBERT retrieves AMD/Intel/NVIDIA articles that share no keywords but match in meaning.

**Takeaway:** a hybrid pipeline is justified — TF-IDF for classification, SBERT for search.

### Topic modeling (LDA, unsupervised)

LDA with 4 topics recovered the true news categories without ever seeing the labels:

| LDA Topic | Top terms | Maps to |
|---|---|---|
| Topic 0 | technology, computer, internet, software, microsoft | Sci/Tech |
| Topic 1 | million, stock, oil, percent, price | Business |
| Topic 2 | iraq, killed, minister, president, people | World |
| Topic 3 | game, team, season, victory, win | Sports |

Cross-tabulation against the real labels showed strong alignment, confirming the dataset has clear latent structure.

## Notebooks

Run in order:

1. **`01_exploration.ipynb`** — data loading, label mapping, class distribution
2. **`02_classical_nlp.ipynb`** — TF-IDF + Logistic Regression and Linear SVM
3. **`03_bert_embeddings.ipynb`** — Sentence-BERT embeddings, semantic similarity search
4. **`04_topic_modeling.ipynb`** — LDA topic modeling and label cross-tabulation
5. **`05_final_evaluation_and_integration.ipynb`** — model comparison, retrieval evaluation, final integration

## Tech Stack

Python, scikit-learn, sentence-transformers, gensim, NLTK, spaCy, pandas, NumPy, matplotlib, seaborn.

## Setup

```bash
# clone
git clone https://github.com/Chandana18G/nlp-news-intelligence.git
cd nlp-news-intelligence

# (recommended) virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# install
pip install -r requirements.txt

# run
jupyter notebook
```

## Project Structure

```
nlp-news-intelligence/
├── 01_exploration.ipynb
├── 02_classical_nlp.ipynb
├── 03_bert_embeddings.ipynb
├── 04_topic_modeling.ipynb
├── 05_final_evaluation_and_integration.ipynb
├── train_clean.csv
├── test_clean.csv
├── requirements.txt
└── README.md
```

## Key Learnings

- Newer isn't automatically better — a Linear SVM on TF-IDF beat general-purpose neural embeddings on this classification task.
- Match the method to the task: lexical features for classification, semantic embeddings for retrieval.
- Unsupervised LDA can recover human-interpretable categories, validating the dataset's structure.
- Evaluating across paradigms surfaces trade-offs a single-model project would miss.

## Author

**Chandana Gurusiddappa**
GitHub: [@Chandana18G](https://github.com/Chandana18G)
LinkedIn: [https://www.linkedin.com/in/chandana-gurusiddappa-785563223/]


