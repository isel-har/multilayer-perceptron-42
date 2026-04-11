import pandas as pd
import numpy as np


from keras.losses import CategoricalCrossentropy

from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import random

seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)



df = pd.read_csv("data_training.csv")
df = df.drop(columns=df.columns[0])

y = df.iloc[:, 0].to_numpy()
X = df.iloc[:, 1:].to_numpy()

del df

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

scaler = StandardScaler()
ohe = OneHotEncoder(sparse_output=False)

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)

y_train = ohe.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
y_val   = ohe.transform(y_val.reshape(-1, 1)).astype(np.float32)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# convert sparse -> labels

y_train_labels = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weights = dict(enumerate(class_weights))


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    shuffle=False,
    batch_size=32,
    class_weight=class_weights,
    epochs=30
)


df = pd.read_csv("data_test.csv")
df = df.drop(columns=df.columns[0])


y_test = df.iloc[:, 0].to_numpy()
X_test = df.iloc[:, 1:].to_numpy()

X_test = scaler.transform(X_test).astype(np.float32)
y_test = ohe.transform(y_test.reshape(-1, 1)).astype(np.float32)

y_pred = model.predict(X_test)

cce = CategoricalCrossentropy()

# compute loss

loss = cce(y_test, y_pred).numpy()
with open('results.txt', 'w') as f:
    f.write(f"keras loss : {loss}\n")
