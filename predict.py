import numpy as np
import pandas as pd
from datetime import datetime

from train import build_model
from sklearn.metrics import classification_report


def run():
    model = build_model()
    print(model.summary())

    model_weights_path = "data/saved-model-01-0.29.hdf5"
    print(f'Loading weights from {model_weights_path}')
    model.load_weights(model_weights_path)

    print('Loading test data...')
    test_data = np.load('data/processed/test_v2.npz')
    X_test, y_test = test_data['X'], test_data['y']

    print('Generating predictions for ')
    y_pred = model.predict_classes(X_test, batch_size=256, verbose=1)

    output_file_path = f'data/y_pred{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
    print(f'Persisting predictions: {output_file_path}')
    df_pred = pd.DataFrame(y_pred)
    df_pred.to_csv(output_file_path)

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Done')


if __name__ == '__main__':
    run()
