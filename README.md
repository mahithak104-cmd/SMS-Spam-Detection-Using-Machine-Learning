# SMS Spam Detection Using Machine Learning

This project uses machine learning and Natural Language Processing (NLP) to classify SMS messages as spam or legitimate (ham). The main goal was to analyze text messages, identify patterns associated with spam, and build machine learning models that can automatically distinguish unwanted messages from normal conversations.

## What I Worked With

The project uses the SMS Spam Collection dataset, which originally contained 5,572 SMS messages.

After removing duplicate records, the dataset contained 5,169 unique messages:

- **Ham messages:** 4,516
- **Spam messages:** 653

I explored the dataset to understand the distribution of spam and legitimate messages before building the classification models.

## What I Built

The project has four main parts:

- **Exploratory Data Analysis** – Analyzing the dataset and understanding the distribution of spam and ham messages.
- **Text Preprocessing and Feature Extraction** – Converting SMS text into numerical features using TF-IDF vectorization.
- **Model Comparison** – Training and evaluating Logistic Regression, Multinomial Naive Bayes, and Random Forest classifiers.
- **SMS Classification** – Testing the trained models on custom messages to predict whether they are spam or legitimate.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF Vectorization
- Google Colab
- Jupyter Notebook

## Text Preprocessing

Before training the models, I prepared the SMS messages for machine learning using:

- Text normalization
- Lowercase conversion
- Stopword removal
- TF-IDF vectorization
- Feature extraction with a maximum of 5,000 features

TF-IDF converts text messages into numerical representations so that machine learning algorithms can identify useful patterns in the message content.

## Exploratory Data Analysis

I analyzed the distribution of spam and legitimate messages to understand the dataset.

The dataset was imbalanced, with legitimate messages representing approximately 87.37% of the cleaned data and spam messages representing approximately 12.63%.

This imbalance is important because a model can achieve high accuracy while still missing some spam messages.

### Spam vs. Ham Distribution

<img width="1246" height="938" alt="Distribution of Spam and Ham Messages" src="https://github.com/user-attachments/assets/7d894534-3a62-4713-a4a2-e5214a296dda" />

## Model Training and Evaluation

I trained and compared three machine learning algorithms:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

The data was split into training and testing sets using an 80/20 split with stratification to preserve the class distribution.

I evaluated the models using accuracy, precision, recall, F1-score, and a confusion matrix.

## Model Performance

The Random Forest classifier achieved the highest accuracy among the three initial models in this experiment.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 97.58% | 97.32% | 83.21% | 89.71% |
| Multinomial Naive Bayes | 97.20% | 100.00% | 77.86% | 87.55% |
| Logistic Regression | 95.84% | 97.83% | 68.70% | 80.72% |

These results show that model performance involves trade-offs. For example, Multinomial Naive Bayes had high spam precision but missed more spam messages than Random Forest.

### Model Performance Comparison

<img width="1960" height="1164" alt="SMS Spam Detection Model Performance Comparison" src="https://github.com/user-attachments/assets/21a0a732-784b-4b4f-a439-46922d2b11c1" />

## Confusion Matrix

I used a confusion matrix to examine the Random Forest model's predictions on the test dataset.

The model correctly classified 900 legitimate messages and 109 spam messages. It incorrectly classified 3 legitimate messages as spam and missed 22 spam messages.

This analysis helped identify the types of classification errors made by the model.

### Random Forest Confusion Matrix

<img width="1010" height="772" alt="Random Forest Confusion Matrix" src="https://github.com/user-attachments/assets/b55fb477-1589-4632-857e-1af3a2f264dd" />

## Improving Spam Detection

I also experimented with Logistic Regression using balanced class weights to give greater importance to the minority spam class.

The balanced Logistic Regression model achieved:

- **Accuracy:** 97.29%
- **Spam precision:** 89.92%
- **Spam recall:** 88.55%
- **Spam F1-score:** 89.23%

Compared with the original Logistic Regression model, balancing the class weights improved spam recall. This demonstrates how handling class imbalance can help detect more spam messages, although it may also increase false-positive predictions.

## Custom SMS Prediction

The project also tests the models using custom SMS messages, including examples of promotional messages, urgent reward claims, everyday conversations, and meeting arrangements.

These tests demonstrate how the trained classifier can predict whether a new message is spam or legitimate. However, predictions are not always correct, and real-world deployment would require additional testing.

## Conclusion

This project gave me hands-on experience applying NLP and machine learning to text classification. It helped me understand text preprocessing, TF-IDF feature extraction, model evaluation, class imbalance, and the trade-offs between precision and recall.

By comparing multiple algorithms and experimenting with balanced class weights, I explored different approaches to improving SMS spam detection. The project provides a foundation for developing more advanced text classification systems and experimenting with additional feature engineering and model optimization.

## Author

Mahitha Kalinathabotla

Master of Science in Business Analytics  
University at Albany, SUNY

GitHub: https://github.com/mahithak104-cmd
