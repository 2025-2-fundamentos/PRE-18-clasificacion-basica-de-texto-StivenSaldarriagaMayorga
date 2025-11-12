
#%%
import pandas as pd

dataframe = pd.read_csv(
    "../files/input/sentences.csv.zip",
    index_col=False,
    compression="zip",
)

#%%
#
# Tamaño del dataset
#
dataframe.shape

#%%
#
# Data
#
dataframe.head()

#%%
#
# Sentimientos (clases)
#
dataframe.target.value_counts()

#%%
# Ejemplos de frases positivas
#
for i in range(5):
    print(dataframe[dataframe.target == "positive"]["phrase"].iloc[i])
    
    
#%%
# Ejemplos de frases negativas
#
for i in range(5):
    print(dataframe[dataframe.target == "negative"]["phrase"].iloc[i])

# %%
# Ejemplos de frases neutrales
#
for i in range(5):
    print(dataframe[dataframe.target == "neutral"]["phrase"].iloc[i])
# %%


#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataframe.phrase,
    dataframe.target,
    test_size=0.3,
    shuffle=False,
)

#%%
# Construcción de la matriz documento-término
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    lowercase=True,
    analyzer="word",
    token_pattern=r"\b[a-zA-Z]\w+\b",
    stop_words="english",
    max_df=0.99,
    min_df=2,
    binary=True,
)
vectorizer.fit(X_train)


vectorizer.get_feature_names_out()

#%%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf
# %%
X_train_vectorized = vectorizer.transform(X_train)

clf.fit(
    X_train_vectorized,
    y_train,
)


# %%
from sklearn.metrics import accuracy_score

#
# Muestra de entrenamiento
#
accuracy_score(
    y_true=y_train,
    y_pred=clf.predict(X_train_vectorized),
)
# %%
from sklearn.metrics import accuracy_score

#
# Muestra de entrenamiento
#
accuracy_score(
    y_true=y_train,
    y_pred=clf.predict(X_train_vectorized),
)

# %%
from sklearn.metrics import accuracy_score

#
# Muestra de entrenamiento
#
accuracy_score(
    y_true=y_train,
    y_pred=clf.predict(X_train_vectorized),
)

# %%
# Muestra de prueba
#
X_test_vectorized = vectorizer.transform(X_test)
predictions = clf.predict(X_test_vectorized)

accuracy_score(
    y_true=y_test,
    y_pred=predictions,
)




# %%
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Greens",
)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")



# %%
import pickle

with open("clf.pkl", "wb") as file:
    pickle.dump(clf, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
    
    
    
    
    
    # %%
with open("clf.pkl", "rb") as file:
    new_clf = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    new_vectorizer = pickle.load(file)

accuracy_score(
    y_true=dataframe.target,
    y_pred=new_clf.predict(new_vectorizer.transform(dataframe.phrase)),
)

# %%
