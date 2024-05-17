import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import csv
import pickle
import optuna
from optuna.samplers import TPESampler

def create_processed_df(cats, conts, df):
    """
    Creates a copy of the original dataframe with cats 
    columns dtype converted to 'category'. 
    """
    model_df = df.copy()
    model_df[cats] = model_df[cats].apply(lambda x: x.astype('category'))
    return model_df
def create_xy(cats,conts,model_df, apply_scale = False, apply_ohe = False):
    """
    Creates two dataframes of features and labels to be used for modeling. 
    If apply_scale is set to True, the conts columns are scaled using StandardScaler from scikit-learn library. 
    If apply_ohe is set to True, the cats columns are converted to one-hot encoded columns.
    """
    x, y = model_df[cats+conts], model_df.Adopted
    
    if apply_scale or apply_ohe:
        scale = StandardScaler()
        ohe = OneHotEncoder(sparse_output = False)
        preprocessor_list = []
        if apply_ohe: preprocessor_list.append(("OneHotEncoder", ohe, cats))
        if apply_scale: preprocessor_list.append(("StandardScaler", scale, conts))
        preprocessor = ColumnTransformer(preprocessor_list, verbose_feature_names_out = False)
        x = preprocessor.fit_transform(x)

    return preprocessor.get_feature_names_out(),x,y
def split_df(x,y,test_size = 0.2, random_state = 42):
    """
    Splits the features (x) and labels (y) into the test size ratio specified. 
    """
    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=test_size, 
                                                    random_state = random_state,
                                                    stratify=y
                                                   )
    return x_train, x_test, y_train, y_test
def create_output_log(filename, mode = "a",fill = []):
    """
    Creates or appends to the csv file "filename" which is created in the logs directory.
    """
    location = "logs/" + filename
    with open(location, mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fill)
def run_model(x_train,y_train,x_test, y_test, model, logging = False,log_filename = "output.csv",model_name = "None", remarks = "None"):
    """
    Runs the given model with the given train and test labels with logging.
    """
    model.fit(x_train, y_train)
    predictionsx = model.predict(x_test)
    accuracyx = accuracy_score(y_test, predictionsx)
    kappa = cohen_kappa_score(y_test, predictionsx, weights = "quadratic")
    f2 = fbeta_score(y_test, predictionsx,pos_label=0, beta =2)
    print("Model name:", model_name)
    print("Accuracy:", accuracyx)
    print("Quadratic weighted kappa:",kappa)
    print("Fbeta score:", f2)
    if logging:
        create_output_log(log_filename,fill = [model_name,kappa,accuracyx,f2,remarks])
        pickle_name = "saved_model/"+ model_name + ".sav"
        pickle.dump(model, open(pickle_name, 'wb'))
    return model,predictionsx, accuracyx, kappa, f2
def cross_val_model(estimator,x,y,logging = True,
                    log_filename = "output.csv",
                    model_name = "XGB", 
                    remarks = "",
                    folds = 10):
    """
    Runs a stratified k-fold validation of the given model.
    """
    val_scores = cross_validate(estimator, x, y, scoring = {"Kappa": make_scorer(cohen_kappa_score,weights = 'quadratic'), 
                                                            "Accuracy": make_scorer(accuracy_score),
                                                            "F2": make_scorer(fbeta_score, pos_label=0,beta = 2)},cv=folds)
    mean_kappa, mean_acc, mean_f2 = val_scores["test_Kappa"].mean(), val_scores["test_Accuracy"].mean(), val_scores["test_F2"].mean(),
    print("Model name:", model_name)
    print("Accuracy:", mean_acc)
    print("Quadratic weighted kappa:",mean_kappa)
    print("Fbeta score:",mean_f2)

    if logging:
        create_output_log(log_filename,fill = [model_name,mean_kappa,mean_acc,mean_f2,remarks])
    return val_scores["test_Kappa"]
def start_HP_tuning(model_used,x_data,y_data, study_name = "", n_trials = 200):
    """
    Runs hyperparameter tuning of the selected model using optuna with the specified number of trials. 
    Using the same name study continues the tuning from when it was previously stopped.
    """
    sqlite_db = "sqlite:///sqlite.db"
    study = optuna.create_study(storage=sqlite_db, study_name=study_name, 
                                sampler=TPESampler(n_startup_trials=100, multivariate=True, seed=0),
                                direction="maximize", load_if_exists=True)

    study.optimize(lambda trial: model_used(trial, x_data, y_data), n_trials=n_trials)
    return study
def lgb_tune(trial, x_data,y_data):
    """
    Tunes hyperparameters for a LGBMCLassifier.
    """
    kfold = StratifiedKFold(10, shuffle=True, random_state=42)
    max_depth = trial.suggest_int('max_depth', 4, 30)
    n_estimators = trial.suggest_int('n_estimators', 500, 2000)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
    min_child_weight = trial.suggest_int('min_child_weight', 0, 10)
    subsample = trial.suggest_float('subsample', 0, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
    learning_rate = trial.suggest_float('learning_rate', 0, 1)
    params = {'learning_rate': learning_rate,
              'n_estimators': n_estimators,
              'max_depth': max_depth,
              'lambda_l1': reg_alpha,
              'lambda_l2': reg_lambda,
              'colsample_bytree': colsample_bytree, 
              'subsample': subsample,    
              'min_child_samples': min_child_weight}
    
    lgbopt = LGBMClassifier(**params,objective = "binary",
                            boosting_type = "gbdt",
                            random_state = 42,
                            verbose=-1)
    cv_splits = kfold.split(x_data, y=y_data)

    cv = cross_val_score(lgbopt, x_data, y_data, cv = cv_splits, scoring=make_scorer(fbeta_score, pos_label=0, beta = 2)).mean()
    return cv
def xgb_tune(trial, x_data,y_data):
    """
    Tunes hyperparameters for a XGBCLassifier.
    """
    kfold = StratifiedKFold(10, shuffle=True, random_state=42)
    grow_policy = trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"])
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    gamma = trial.suggest_float('gamma', 1e-9, 1.0)
    subsample = trial.suggest_float('subsample', 0.25, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.25, 1.0)
    max_depth = trial.suggest_int('max_depth', 0, 24)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 30)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True)

    params = {'grow_policy': grow_policy,
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'gamma' : gamma,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'reg_lambda': reg_lambda,
    'reg_alpha': reg_alpha}
    
    xgbopt = XGBClassifier(**params,objective = "binary:logistic",
                           eval_metric = "logloss",
                            booster = "gbtree",
                            random_state = 42)
    cv_splits = kfold.split(x_data, y=y_data)

    cv = cross_val_score(xgbopt, x_data, y_data, cv = cv_splits, scoring=make_scorer(fbeta_score, pos_label=0, beta = 2)).mean()
    return cv
def ada_tune(trial, x_data, y_data):
    """
    Tunes hyperparameters for a AdaBoostCLassifier.
    """
    kfold = StratifiedKFold(10, shuffle=True, random_state=42)
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)

    params = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate}
    
    adaopt = AdaBoostClassifier(**params,
                                algorithm = "SAMME",
                                random_state = 42)
    cv_splits = kfold.split(x_data, y=y_data)

    cv = cross_val_score(adaopt, x_data, y_data, cv = cv_splits, scoring=make_scorer(fbeta_score, pos_label=0, beta = 2)).mean()
    return cv
def dt_tune(trial, x_data, y_data):
    """
    Tunes hyperparameters for a DecisionTreeCLassifier.
    """
    kfold = StratifiedKFold(10, shuffle=True, random_state=42)
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 3, 50)

    params = {
    'criterion': criterion,
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split}
    
    dtopt = DecisionTreeClassifier(**params,
                                random_state = 42)
    cv_splits = kfold.split(x_data, y=y_data)

    cv = cross_val_score(dtopt, x_data, y_data, cv = cv_splits, scoring=make_scorer(fbeta_score, pos_label=0, beta = 2)).mean()
    return cv
def compute_feature_importance(voting_clf, weights):
    """
    Computes feature importance of Voting Classifier
    """
    feature_importance = dict()
    for est in voting_clf.estimators_:
        feature_importance[str(est)] = est.feature_importances_
    
    fe_scores = [0]*len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score*weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores

def show_figs(model,model_name, feature_names, y_dev, final_preds):
    vot_classifier = model
    feat_importances = pd.Series(compute_feature_importance(vot_classifier, [1, 1, 1]), index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh').get_figure().savefig(f'logs/{model_name}_feat_importances.png',bbox_inches ="tight")
    
    cm = confusion_matrix(y_dev,
                        final_preds,
                        normalize=None).round(2)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot().figure_.savefig(f'logs/{model_name}_confusion_matrix.png',bbox_inches ="tight")