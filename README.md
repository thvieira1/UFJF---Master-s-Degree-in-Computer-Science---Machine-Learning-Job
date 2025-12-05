# Spam Classification Using Machine Learning and GPT

This project compares classical machine learning models with a modern large language model (GPT) for the task of spam detection. Two datasets of labeled messages were used to evaluate performance and generalization.

## Datasets

Two public datasets were used.  
Each contains a text message and a label ("spam" or "ham").  
The datasets differ in size, message length and variety, allowing comparison between different data conditions.

## Methodology

The project is divided into two stages: classical machine learning models and GPT zero-shot classification.

### Text Processing

All text messages were transformed using TF-IDF with up to 3000 features and n-grams (1 and 2).  
This converts each message into a numeric vector suitable for machine learning algorithms.

### Classical Machine Learning Models

The following models were trained:

- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- Soft Voting Ensemble  
- Weighted Ensemble  

Each model was trained and evaluated under the same conditions.

### Evaluation Protocol

All classical models were evaluated using 5-fold stratified cross-validation.  
This ensures that each fold maintains the same spam/ham proportion and that all models are evaluated on identical data splits.

Metrics collected for each fold:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Execution time  

Final performance is the average across the five folds.

## GPT Zero-Shot Classification

GPT was used without any training or fine-tuning.  
Each message was classified using a natural-language prompt instructing GPT to respond with either "spam" or "ham".  
The same 5-fold test splits were used, but GPT was evaluated only on the test portion of each fold.

This ensures a fair comparison even though GPT does not undergo supervised training.

Average GPT results across 5 folds:

- Accuracy around 0.9492  
- Precision around 0.7326  
- Recall around 0.9786  
- F1-score around 0.8379  

GPT showed very high recall but lower precision than the classical models.

## Summary of Results

Among classical models, SVM, Logistic Regression, Random Forest, and the Weighted Ensemble achieved the best performance, with F1-scores typically between 0.93 and 0.97 depending on the dataset.

GPT, despite not being trained, achieved strong recall and competitive overall performance, but with more false positives, leading to lower precision.

## Conclusion

Classical machine learning models remain highly effective for spam detection, especially when evaluated using TF-IDF representations.  
Ensemble methods provide additional robustness and often achieve the best overall performance.  
GPT performs well in zero-shot mode and can understand semantic context beyond surface-level word frequencies, but it is slower and less precise than the classical models.

A hybrid approach combining classical ML models with GPT may provide future improvements by leveraging strengths from both approaches.
