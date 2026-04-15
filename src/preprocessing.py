import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

    df['lead_time'] = (df['Ship Date'] - df['Order Date']).dt.days

    return df