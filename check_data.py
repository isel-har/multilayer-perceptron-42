

import pandas as pd

df = pd.read_csv("data/data_train.csv")

print(df[df.iloc[:, 1] == 'M'].shape[0])
print(df[df.iloc[:, 1] == 'B'].shape[0])