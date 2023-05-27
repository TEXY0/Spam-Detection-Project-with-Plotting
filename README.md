This repository contains code for email spam classification using a Naive Bayes classifier in Python. The code performs various steps, including data preprocessing, model training, evaluation, and visualization.

Segment 1: Importing Libraries and Loading Data
The necessary libraries such as numpy, pandas, nltk, sklearn, matplotlib.pyplot, and seaborn are imported. The email dataset is loaded from a CSV file using pd.read_csv().

Segment 2: Data Preprocessing
Duplicates are removed from the dataset using df.drop_duplicates(inplace=True). The process_text function is defined to preprocess the text by removing punctuation and stopwords. The text is tokenized by applying the process_text function to the 'text' column and storing the result in a new 'text_tokens' column.

Segment 3: Splitting Data and Vectorizing Text
The data is split into training and testing sets using train_test_split(). The text data is vectorized using CountVectorizer, which converts the text into a matrix of token counts. The vectorization is performed separately for the training and testing sets.

Segment 4: Training and Evaluating the Model
A Naive Bayes Classifier (MultinomialNB) is trained on the training data (X_train_bow and y_train). The model is evaluated on both the training and testing datasets using metrics such as classification report, confusion matrix, and accuracy score.

Segment 5: Visualization
Dimensionality reduction is performed using Principal Component Analysis (PCA) to reduce the dimensionality of the vectorized text data (X_test_bow) to 2 components for visualization purposes. A heatmap is plotted to visualize the confusion matrix.
A pie chart is created to show the distribution of predicted values. Linear regression is performed on the predicted values (test_predictions) to generate a regression line. Finally, a scatter chart is plotted to visualize the actual values vs. predicted values with the regression line.

This repository serves as a useful resource for anyone interested in email spam classification using a Naive Bayes classifier. 
The code demonstrates the steps involved in data preprocessing, model training, evaluation, and visualization, providing insights into the performance of the classifier and the distribution of predicted values.
