# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import joblib

# Loading both datasets
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Giving Labels to each dataset
true_news["Label"] = np.ones((21417,1), dtype=int)
fake_news["Label"] = np.zeros((23481,1), dtype=int)

# Concatenating both datasets
dataset = pd.concat([true_news,fake_news], axis=0)
dataset = dataset[["title", "text", "Label"]]

# Preprocessing
dataset.isnull().sum()
dataset.duplicated().sum()
dataset = dataset.drop_duplicates()

# (TF-IDF) VECTORIZATION
# Conversion to lowercase
dataset["title"] = dataset["title"].str.lower()
dataset["text"] = dataset["text"].str.lower()

# Tokenization, Stopwords and punctuation removal
def cleaning_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    clean_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return clean_tokens
dataset["title"] = dataset["title"].apply(cleaning_text)
dataset["text"] = dataset["text"].apply(cleaning_text)

# Lemmatization
def lemmatizing_text(clean_token):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clean_token]
    return lemmatized_tokens
dataset["title"] = dataset["title"].apply(lemmatizing_text)
dataset["text"] = dataset["text"].apply(lemmatizing_text)

# Converting tokens into strings and concatenating (title, text)
dataset['text_str'] = dataset['text'].apply(lambda x: " ".join(x))
dataset['title_str'] = dataset['title'].apply(lambda x: " ".join(x))
dataset['combined_text'] = dataset['title_str'] + " " + dataset['text_str']

# Train-Test-Spliting
X = dataset["combined_text"]
Y = dataset["Label"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vectorizing Xtrain and Xtest
tfidf = TfidfVectorizer(max_features=5000)
Xtrain_tfidf = tfidf.fit_transform(Xtrain)
Xtest_tfidf = tfidf.transform(Xtest)

# Model Selection and Hyperparameter Tunning
models = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100,150,200],
            "max_depth": [5,7,9,10],
            "min_samples_split": [2,4,6,8]
        }
    },
    
    "LogisticRegression":{
        "model": LogisticRegression(),
        "params":{
            "max_iter": [100,150,200,250]
        }
    }
}
scores = []
final_model = None
best_score = 0
for name,model_params in models.items():
    # Model Training and prediction
    best_model_params = RandomizedSearchCV(estimator = model_params["model"],    
                                           param_distributions=model_params["params"], cv=5)
    best_model_params.fit(Xtrain_tfidf, Ytrain)
    if best_model_params.best_score_ > best_score:
        best_score = best_model_params.best_score_
        final_model = best_model_params.best_estimator_
   
    scores.append({
        "model": name,
        "best parameters": best_model_params.best_params_,
        "score": best_model_params.best_score_
    })

params_scores_df = pd.DataFrame(scores)
print(params_scores_df)

# Prediction and Evaluation
Ypred = final_model.predict(Xtest_tfidf)

cm = confusion_matrix(Ytest, Ypred)
acc = accuracy_score(Ytest, Ypred)
pre = precision_score(Ytest, Ypred)
recall = recall_score(Ytest, Ypred)
F1_score = f1_score(Ytest, Ypred)

print(f"Accuracy score: {acc:.2f}")
print(f"Precision score: {pre:.2f}")
print(f"Recall score: {recall:.2f}")
print(f"F1 score: {F1_score:.2f}")

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake", "Real"],
           yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.show()

# Saving TF-IDF Vectorizer and Model
with open("Vectorizer.pkl", 'wb') as vectorizer:
    joblib.dump(tfidf, vectorizer)
with open("Fake_News_Detector_Model.pkl", 'wb') as detector:
    joblib.dump(final_model, detector)
