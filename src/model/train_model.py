import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

import json
from utils import *

# loads config parameters from config.json as a Python dictionary
with open("src/model/config.json","r") as f:
    config = json.load(f)

# maps model names from config to classes
models = {
    "Logistic Regression": LogisticRegression(max_iter = 500),
    "SVC": SVC(),
    "Ridge": RidgeClassifier(),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state = 42),
    "Random Forest": RandomForestClassifier(random_state = 42),
    "XGB Classifier": XGBClassifier(random_state = 42), 
    "LGBM Classifier": LGBMClassifier(random_state = 42, verbose = -1),
    "AdaBoost Classifier": AdaBoostClassifier(random_state = 42, algorithm = "SAMME")
}

# maps model names from config to respective model hyperparameter tuning functions
tuners = {"LGBClassifier": lgb_tune,
          "XGBClassifier": xgb_tune,
          "AdaBoostClassifier": ada_tune,
          "DecisionTreeClassifier": dt_tune}

# loads pandas dataframes as features (x) and labels (y) from the output of process_data.py
df_copy2 = pd.read_csv("data/processed_data.csv")
model_df = create_processed_df(**config["features"], df=df_copy2)
feature_names,x, y = create_xy(**config["features"],model_df=model_df, **config["create_xy"])

# creates a train (for training), test (for validation) and dev dataset (for evaluation), split in a 9:1:1 ratio
x, x_dev, y, y_dev = split_df(x,y,0.0909)
x_train, x_test, y_train, y_test = split_df(x,y,0.1)

# if enabled, creates or overwrites a new csv file for logging modelname, metrics and includes a remarks column
if config["run_modes"]["reset_log"]:
    create_output_log('output.csv',"w",["Model","QWK","Accuracy","F2","Remarks"])

# if enabled, trains the given list of models on the training dataset
if config["run_modes"]["baseline_train"]:
    for model_name in config["list_of_models"]:
        model = models[model_name]
        run_model(x_train,y_train,x_test, y_test, model,
                        logging = config["baseline_train_params"]["logging"],
                        log_filename = config["baseline_train_params"]["log_filename"],
                        model_name = model_name + "_"+ config["baseline_train_params"]["model_name"], 
                        remarks = model_name + "_" + config["baseline_train_params"]["remarks"]
                        )

# if enabled, runs a k-fold CV with the given list of models on the processed dataset
if config["run_modes"]["baseline_cv"]:
    for model_name in config["list_of_models"]:
        model = models[model_name]
        cross_val_model(model,x,y,
                        logging = config["baseline_cv_params"]["logging"],
                        log_filename = config["baseline_cv_params"]["log_filename"],
                        model_name = model_name + "_" + config["baseline_cv_params"]["model_name"], 
                        remarks = model_name + "_" + config["baseline_cv_params"]["remarks"],
                        folds = config["baseline_cv_params"]["folds"])

# if enabled, performs hyperparameter tuning on the given list of models on the processed dataset
if config["run_modes"]["HP_tuning"]:
    for tuner_name in config["HP_tuning_params"]["tuners"]:
        best = start_HP_tuning(tuners[tuner_name],x,y,
                                study_name = tuner_name + " " + config["HP_tuning_params"]["study_name"], 
                                n_trials = config["HP_tuning_params"]["n_trials"])
        print("*"*30)
        print(f"Best params for {tuner_name}:{best.best_params}")

if config["run_modes"]["voting_classifier"]:

    """
    If enabled, trains a VotingClassifier on the given list of models on the processed dataset and uses x.dev to predict the final output. 
    Saves a .png of the resulting confusion matrix and feature importances in the logs directory.
    """

    estimator_list = [('LGBM', LGBMClassifier(**config["voting_classifier_params"]["best_lgb_params"],objective = "binary",boosting_type = "gbdt",random_state = 42, verbose=-1)),
                    ('XGB', XGBClassifier(**config["voting_classifier_params"]["best_xgb_params"],objective = "binary:logistic",eval_metric = "logloss",booster = "gbtree",random_state = 42)),
                    ('DecisionTree', DecisionTreeClassifier(**config["voting_classifier_params"]["best_dt_params"],random_state = 42))
    ] 
    vot_hard = VotingClassifier(estimators = estimator_list, voting ='hard')

    vot_hard, final_preds,_,_,_ = run_model(x,y,x_dev, y_dev, vot_hard,
                                  **config["voting_classifier_train_params"]
                        )
    show_figs(vot_hard, 
              config["voting_classifier_train_params"]["model_name"],
              feature_names, 
              y_dev, 
              final_preds)

if config["run_modes"]["use_existing_model"]:
    """
    If enabled, runs a named saved model on the processed dataset and uses x.dev to predict the final output. 
    Saves a .png of the resulting confusion matrix and feature importances in the logs directory.
    """
    loaded_model = pickle.load(open(config["use_existing_model"]["model_path"], 'rb'))
    loaded_model_preds = loaded_model.predict(x_dev)
    show_figs(loaded_model, 
              config["use_existing_model"]["model_name"],
              feature_names, 
              y_dev, 
              loaded_model_preds)