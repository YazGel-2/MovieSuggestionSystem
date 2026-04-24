import pandas as pd


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