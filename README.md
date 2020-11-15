# Sentiment-Analysis-Via-ML-Based-Approach
Download the “Product Sentiment” dataset from the course portal: sentiment_train.csv and sentiment_test.csv.

-	Load, clean, and preprocess the data as you find necessary.
-	Using the training data, extract features from the text (i.e., BOW and/or Bag of N-Grams and/or topics and/or lexical features and/or whatever you want). 
-	Use your favorite ML algorithm to train a classification model.  Don’t forget everything that we’ve learned in our ML course: hyperparameter tuning, cross validation, handling imbalanced data, etc. Make reasonable decisions and try to create the best-performing classifier that you can.
-	Use the testing data to measure the accuracy and F1-score of your model. 

## Model Development
- Data EDA
- Preporcessing with stopwords, unidecode & Lemmatizer
- Pipeline via TfidfVectorizer condect via
  - Decision Tree
  - Random Forest
  - Knn
  - XGBoost
  - Regression 
  - ADBoost
  - GBT
  - MLP
  - NMF
  
## Results
-	Model F1: 75%
- Model Accuracy: 75%
-	From the business point of view, I am somewhat satisfied from the result above. A 75% of both F1 score and Accuracy score indicates for every 100 reviews that clients post, the model is able to capture 75 of their opinion towards to the company’s product and services successfully.
