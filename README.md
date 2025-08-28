# Quora Insincere Questions Classification

This project tackles the [Kaggle competition](https://www.kaggle.com/competitions/quora-insincere-questions-classification) of detecting insincere questions on Quora. It is a **binary text classification** task:

- **0** → Sincere question  
- **1** → Insincere question  

---

## Dataset
- **train.csv** (1.3M labeled questions)  
- **test.csv** (unlabeled)  
- **sample_submission.csv**  

---

## Modeling Approaches

1. **TF-IDF + XGBoost**  
   - Preprocessing: stopword removal + stemming  
   - Features: top 50k TF-IDF terms  
   - Imbalance handled via `scale_pos_weight`  
   - **F1 Score:** ~0.37  

2. **Embedding + SimpleRNN**  
   - Tokenized & padded sequences (`max_len=50`, vocab=50k)  
   - Architecture: `Embedding → SimpleRNN(64) → Dropout(0.3) → Dense(sigmoid)`  
   - **F1 Score:** ~0.59  

3. **Embedding + Bidirectional GRU**  
   - Same preprocessing as RNN  
   - Architecture: `Embedding → Bi-GRU(64) → Dropout(0.3) → Dense(sigmoid)`  
   - **F1 Score:** ~0.53  

4. **DistilBERT (Fine-tuned)**  
   - Pretrained `distilbert-base-uncased` from HuggingFace  
   - Max length = 50 tokens, trained with AdamW + BCE loss  
   - **F1 Score:** ~0.67 ✅ *best performance*  

---

## Notes
- The dataset is highly **imbalanced** → oversampling, focal loss, or ensembles could improve results.  
- GRU underperformed compared to simple RNN. Possible reason: GRU’s gating made it harder to learn from this relatively small dataset, while RNN’s simpler structure generalized better.
- Longer fine-tuning or larger BERT variants could further boost performance.

---

## Acknowledgments
- [Kaggle](https://www.kaggle.com) for the dataset.  
- [HuggingFace Transformers](https://huggingface.co/transformers/) for pretrained models.  

---
