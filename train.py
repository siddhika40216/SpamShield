import pandas as pd
import nltk
import string
import pickle
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


rfc = RandomForestClassifier()

mnb = MultinomialNB()

etc = ExtraTreesClassifier()


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()

    text = text.split()  

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


x= pd.read_csv("sms-spam.csv")
e = pd.read_csv("emails.csv")
x = x[['v1','v2']]
x.columns = ['label','text']

x['label'] = x['label'].map({
    'ham':0,
    'spam':1
})

e = e[['text','spam']]
e.columns = ['text','label']
e = e[['label','text']]

df = pd.concat([x,e],ignore_index=True)
# keep only first 2 columns
df = df.iloc[:, :2]

# rename columns properly
df.columns = ['result', 'input']

# convert labels
df['result'] = df['result'].map({'ham': 0, 'spam': 1})


# ----------------------------
# APPLY TRANSFORMATION
# ----------------------------
df['transformed_text'] = df['input'].apply(transform_text)

# ----------------------------
# VECTORIZE
# ----------------------------
tk = TfidfVectorizer(max_features=7000)

X = tk.fit_transform(df['transformed_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = VotingClassifier(
    estimators=[
        ('rf', rfc),
        ('nb', mnb),
        ('et', etc)
    ],
    voting='soft'
)

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tk, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")