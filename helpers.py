import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def upsample_train(df, target_col, replace=True, random_state=20224740):
    unique_vals = df[target_col].value_counts()
    max_count = max(unique_vals)
    final_df = pd.DataFrame()
    for class_name in unique_vals.index:
        if unique_vals.loc[class_name] < max_count:
            upsampled_class = resample(
                df[df[target_col] == class_name],
                replace=replace,
                n_samples=max_count,
                random_state=random_state
            )
        else:
            upsampled_class = df[df[target_col] == class_name]

        assert len(upsampled_class) == max_count
        final_df = pd.concat([final_df, upsampled_class])

    return final_df


def kfold_cv(model, X, y, n_splits=5, upsample=False, random_state=20224740, verbose=True, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scores = {'f1':[], 'auc':[], 'acc':[]}

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):

        # split the data
        train_data = X.iloc[train_idx]
        test_data = X.iloc[test_idx]

        train_labels = y.iloc[train_idx]
        test_labels = y.iloc[test_idx]

        # upsampling the training data
        if upsample:
            train_data = pd.concat([train_data, train_labels], axis=1)
            upsampled = upsample_train(
                train_data, 
                target=y.name, 
                replace=True, 
                random_state=random_state
                )
            train_data = upsampled.drop(y.name, axis=1)
            train_labels = upsampled[y.name]

        # train the model
        model.fit(train_data, train_labels)

        # make predictions
        preds_class = model.predict(test_data)
        preds_proba = model.predict_proba(test_data)

        # calculate the scores
        f1 = f1_score(preds_class, test_labels, pos_label='>50K')
        auc = roc_auc_score(test_labels, preds_proba[:, 1])
        acc = accuracy_score(test_labels, preds_class)

        # save the scores
        scores['f1'].append(f1)
        scores['auc'].append(auc)
        scores['acc'].append(acc)

        if verbose:
            print(f'Fold {i+1}.')
            print(f'F1 score: {f1:.3f}. AUC: {auc:.3f}. Accuracy: {acc:.3f}.\n')

    return model, scores


def plot_roc_curve(classifier, X, y, pos_label):
    """
    plots the roc curve based of the probabilities
    """
    # get the name of the classifier
    clasifier_name = classifier.__class__.__name__

    # predict class and probabilities
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)[:, 1]

    # calculate the scores
    fpr, tpr, thresholds = roc_curve(y, y_prob, pos_label=pos_label)
    auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred, pos_label=pos_label)
    acc = accuracy_score(y, y_pred)
    baseline = np.arange(0, 1, 0.01)

    # plot the roc curve
    plt.plot(baseline, baseline, color='red', linestyle='dashed', label='baseline (random classifier AUC=0.5)')
    plt.plot(fpr, tpr, linewidth=3, label=f'{clasifier_name}|AUC={auc:.2f}|F1={f1:.2f}|ACC={acc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc="lower right")




def preprocess_tabnet_data(df, for_prediction=False, encoder_path='tabnet/tabnet_classes_encoder.npy'):
    train = df.copy()
    target = 'class'
    np.random.seed(20224740)
    # Split into train and test
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "test"], p =[.8, 0.2], size=(train.shape[0],))

    # Drop the RowID column -- irrelevant -- only for train data
    if not for_prediction:
        train.drop(['RowID'], axis=1, inplace=True)

    train_indices = train[train.Set=="train"].index
    test_indices = train[train.Set=="test"].index

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims =  {}

    if not for_prediction:
        encoders = {}
        for col in train.columns:
            if (types[col] == 'object' or nunique[col] < 200) and col not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                l_enc = LabelEncoder()
                rep_value = str(train[col].mode()[0])
                # train[col].fillna(rep_value, inplace=True)
                for i in range(len(train[col])):
                    if type(train[col].iloc[i]) != str:
                        train[col].iloc[i] = rep_value
                train[col] = l_enc.fit_transform(train[col].values)
                encoders[col] = l_enc  # to save the encoder
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
            else:
                train.fillna(train.loc[train_indices, col].mean(), inplace=True)
        np.save(encoder_path, encoders)
    else:
        encoders = np.load(encoder_path, allow_pickle=True).item()
        for col in train.columns:
            if (types[col] == 'object' or nunique[col] < 200) and col not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                if col == 'RowID': continue
                l_enc = encoders[col]
                rep_value = str(train[col].mode()[0])
                # train[col].fillna(rep_value, inplace=True)
                for i in range(len(train[col])):
                    if type(train[col].iloc[i]) != str:
                        train[col].iloc[i] = rep_value
                train[col] = l_enc.transform(train[col].values)
                encoders[col] = l_enc  # to save the encoder
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
            else:
                train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    if for_prediction:
        unused_feat = ['Set']
        train.drop(unused_feat, axis=1, inplace=True)
        index = train['RowID']
        train.drop('RowID', axis=1, inplace=True)
        return train, index

    unused_feat = ['Set']

    features = [ col for col in train.columns if col not in unused_feat+[target]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    # X_train = train[features].values[train_indices]
    # y_train = train[target].values[train_indices]


    # Upsample
    if not for_prediction:
        train_final = upsample_train(
            pd.concat([train[features].iloc[train_indices], train[target].iloc[train_indices]], axis=1),
            target
        )
        X_train = train_final.drop(target, axis=1).values
        y_train = train_final[target].values
    # else:
    #     X_train = train[features].values[train_indices]
    #     y_train = train[target].values[train_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    return X_train, y_train, X_test, y_test, cat_idxs, cat_dims, encoders, train

def preprocess_catboost(df, for_prediction=False):
    train = df.copy()

    target_col = 'class'
    cols_to_drop = ['RowID', 'education']
    categorical_cols = ['marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation', 'workclass']


    # one of the beauties of catboost is that it does not require us to deal with missing values or preprocess categorical variables
    if for_prediction:
        index = train['RowID']
        train = train.drop(cols_to_drop, axis=1)
        train[categorical_cols] = train[categorical_cols].astype(str)
        return train, index
        
    train = train.drop(cols_to_drop, axis=1)

    X = train.drop(target_col, axis=1)
    y = train[target_col]

    X[categorical_cols] = X[categorical_cols].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20224740, stratify=y)

    return X_train, X_test, y_train, y_test