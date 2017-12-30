Classification Using Apache Spark Please look at https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6974851263301701/550720149541967/6267955876615859/latest.html

[Using NaiveBayes in Spark] Accuracy of model at predicting the Post was: 0.796022991694

# Classify_Facebook-_posts-fake_seller-re-seller_and_no-seller
Classify Facebook posts into three categories: fake_seller, re-seller, no-seller.

Using sklearn : 

MultinomialNB accuracy_score  : 0.83983383337179784
SVC accuracy_score            : 0.82183244864989613
Logistic Regression_tf_idf    : 0.59912300946226638
Logistic_Regression_No_tf_idf : 0.62474036464343408


Useful Resources 

10 Minutes to pandas
https://pandas.pydata.org/pandas-docs/stable/10min.html

Language_Detector : Port of Google's language-detection library to Python.
https://github.com/Mimino666/langdetect

Naive Bayes classifier for multinomial models
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

Convert a collection of text documents to a matrix of token counts
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

C-Support Vector Classification : 
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Split arrays or matrices into random train and test subsets
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Accuracy classification score.
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

Accessing Text Corpora and Lexical Resources
http://www.nltk.org/book/ch02.html

Tokenizer Interface
http://www.nltk.org/api/nltk.tokenize.html

A processing interface for removing morphological affixes from words
http://nullege.com/codes/search/nltk.stem.lancaster.LancasterStemmer


