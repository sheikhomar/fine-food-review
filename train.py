import json

import numpy as np
from tensorflow.keras.layers import (
    Dropout, Embedding, Flatten, Dense,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


def run():
    with open('data/processed/data-info.json', 'r') as f:
        data_info = json.load(f)

    vocab_size = data_info['vocab_size']
    max_doc_len = data_info['max_doc_len']
    embed_dim = 300

    train_data = np.load('data/processed/train.npz')
    X_train, y_train = train_data['X'], train_data['y']

    test_data = np.load('data/processed/test.npz')
    X_test, y_test = test_data['X'], test_data['y']

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_doc_len))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer=RMSprop(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    print('Done')


if __name__ == '__main__':
    run()
