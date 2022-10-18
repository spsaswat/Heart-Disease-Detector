from __future__ import print_function

import argparse
import os
import sklearn
import pandas as pd
import numpy as np

import joblib
import os

from sklearn.ensemble import RandomForestClassifier

# model selection
from sklearn.model_selection import train_test_split
# evaluation
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "heart.csv")
    data_hdy = pd.read_csv(file, engine="python")

    # labels are in the first column
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    features_name = ['cp', 'thalach', 'slope', 'restecg', 'trestbps', 'age', 'sex', 'thal', 'ca', 'oldpeak', 'exang', 'chol']
    features = data_hdy[features_name]
    target = data_hdy['target']
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 20, stratify=target)

    rfc = RandomForestClassifier(random_state = 20, max_depth = 3) # default criterion is gini
    rfc = rfc.fit(x_train,y_train)
    y_pred = rfc.predict(x_test)
    print(classification_report(y_test, y_pred))

    # saving the model
    joblib.dump(rfc, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    rfc = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rfc