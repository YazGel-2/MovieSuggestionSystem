import pandas as pd
import ast
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


def select_columns():
    df = pd.read_csv("DataSets/MovieReviews_Raw.csv")
    df = df.drop(columns=["Ratings", "Resenhas","genres","Description"])
    df.rename(columns={df.columns[0]: "Index"}, inplace=True)
    df.to_csv("DataSets/MovieReviews_CleanedColumns.csv", index=False)

def drop_duplicates():
    df = pd.read_csv("DataSets/MovieReviews_CleanedColumns.csv")
    df = df.drop_duplicates(subset=["Reviews"])
    df.to_csv("DataSets/MovieReviews_NoDuplicates.csv", index=False)

def drop_empties():
    df = pd.read_csv("DataSets/MovieReviews_NoDuplicates.csv")
    df_clean = df.dropna()
    df_clean.to_csv("DataSets/MovieReviews_Cleaned.csv", index=False)

def do_lowercase():
    df = pd.read_csv("DataSets/MovieReviews_Cleaned.csv")
    df["Reviews"] = df["Reviews"].str.lower()
    df.to_csv("DataSets/MovieReviews_Lowercase.csv", index=False)

def clean_noise():
    df = pd.read_csv("DataSets/MovieReviews_Lowercase.csv")

    def clean_text(text):
        if pd.isna(text):
            return text
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'<.*?>', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    df["Reviews"] = df["Reviews"].apply(clean_text)
    
    df.to_csv("DataSets/MovieReviews_NoNoise.csv", index=False)


def remove_punctuation():
    df = pd.read_csv("DataSets/MovieReviews_NoNoise.csv")
    
    df["Reviews"] = df["Reviews"].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    
    df.to_csv("DataSets/MovieReviews_NoPunctuation.csv", index=False)


def remove_stopwords():
    df = pd.read_csv("DataSets/MovieReviews_NoPunctuation.csv")
    
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    def clean_stopwords(text):
        if pd.isna(text):
            return text
        return " ".join([word for word in str(text).split() if word not in stop_words])

    df["Reviews"] = df["Reviews"].apply(clean_stopwords)
    df.to_csv("DataSets/MovieReviews_NoStopwords.csv", index=False)

def tokenize():
    df = pd.read_csv("DataSets/MovieReviews_NoStopwords.csv")
    
    df["Reviews"] = df["Reviews"].apply(lambda x: str(x).split())
    
    df.to_csv("DataSets/MovieReviews_Tokenized.csv", index=False)

def lemmatize():
    df = pd.read_csv("DataSets/MovieReviews_Tokenized.csv")

    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def lemmatize_tokens(text):
        if pd.isna(text):
            return text
        
        if isinstance(text, str):
            text = ast.literal_eval(text)

        pos_tags = pos_tag(text)

        return [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

    df["Reviews"] = df["Reviews"].apply(lemmatize_tokens)

    df.to_csv("DataSets/MovieReviews_Lemmatized.csv", index=False)