import pandas as pd

df = pd.read_csv("data_training.csv")

print(df.iloc[:, 1].value_counts())