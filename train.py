import json
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dropout, Embedding, Flatten, Dense, LSTM
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report


def build_model():
    with open('data/processed/data-info.json', 'r') as f:
        data_info = json.load(f)

    lstm = True

    vocab_size = data_info['vocab_size']
    max_doc_len = data_info['max_doc_len']
    embed_dim = 32

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_doc_len))
    if lstm:
        model.add(LSTM(100))
    else:
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    return model


def run():
    model = build_model()
    print(model.summary())

    model.compile(
        optimizer=RMSprop(lr=1e-3), loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=2)
    filepath = 'data/saved-model-epoch{epoch:02d}-val_loss{val_loss:.4f}.hdf5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')

    train_data = np.load('data/processed/train_v2.npz')
    X_train, y_train = train_data['X'], train_data['y']
    model.fit(X_train, y_train, epochs=1, batch_size=256, validation_split=0.3,
              callbacks=[early_stop, checkpoint]
              )

    print('Loading test data...')
    test_data = np.load('data/processed/test_v2.npz')
    X_test, y_test = test_data['X'], test_data['y']
    y_pred = model.predict_classes(X_test, batch_size=256, verbose=1)

    output_file_path = f'data/predictions-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
    print(f'Persisting predictions: {output_file_path}')
    df_pred = pd.DataFrame(y_pred)
    df_pred.to_csv(output_file_path)

    print(classification_report(y_test, y_pred))

    print('Done')


if __name__ == '__main__':
    run()
