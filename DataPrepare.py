import pandas as pd


def select_columns():
    df = pd.read_csv("DataSets/MovieReviews_Raw.csv")
    df = df.drop(columns=["Ratings", "Resenhas","genres","Description"])
    df.rename(columns={df.columns[0]: "Index"}, inplace=True)
    df.to_csv("DataSets/MovieReviews_CleanedColumns.csv", index=False)