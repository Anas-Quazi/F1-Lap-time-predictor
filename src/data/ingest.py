import pandas as pd

URL = ("https://raw.githubusercontent.com/KathirVelan11/F1-Race-Data-Optimization/"
       "main/data/processed/combined_dataset.csv")

def load_raw():
    return pd.read_csv(URL)

if __name__ == "__main__":
    df = load_raw()
    df.to_csv("data/raw_laps.csv", index=False)   # local copy, so no re-download
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
    print(df["Compound"].value_counts())
    print(df.groupby("Year")["raceId"].nunique())