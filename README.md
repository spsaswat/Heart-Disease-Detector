# Heart Disease Detector
Using Data visualization/analysis, and machine learning models to build a model for detecting heart disease. Different machine learning models were trained and their performances were compared to find the best model.
After finding out that <b>Random Forest and XGBoost</b> performs the best, now it is time to deploy the model.

# Deploying the model using AWS
For deploying I chose the XGBoost model. An xgboost deployment pipeline is better because retraining XGBoost is faster than random forest.
First I created a S3 bucket to store the training data. Then I retrained the model using the same process and parameters, while building the model locally.
### References
1) Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset <br>
2) AWS documentations: <br>
&emsp;2.1) https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/ <br>
&emsp;2.2) https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#deploy-sklearn-models

