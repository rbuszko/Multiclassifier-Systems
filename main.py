# Libraries
from random import seed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import yellowbrick as yb
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neural_network import MLPClassifier
from multi_imbalance.resampling.mdo import MDO
from sklearn.metrics import confusion_matrix

# My libraries
import classifiers.pool_classifiers as pool
from pruning_methods.margin_ordering import margin_ordering                   # Document 1
from pruning_methods.divp import divp                                         # Document 4
from pruning_methods.agglomerative_clustering import agglomerative_clustering # Document 2
from pruning_methods.ensemble_prunning import ensemble_prunning               # Document 3
from pruning_methods.interclass_competences import interclass_competences     # Document 7
from pruning_methods.consensus_enora import consensus_enora                   # Document 6
#########################
### Interesting links ###
#########################

# Example implementation of my dataset
# https://www.kaggle.com/akshat0007/fetalhr/code
# https://phuongdelrosario.medium.com/uci-cardiotocography-data-set-fetal-states-classification-part-1-data-summary-and-eda-e0cec8a61eff
# Training, validation, test datasets
# https://machinelearningmastery.com/difference-test-validation-datasets/
# Unbalanced data methods
# https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
# Feature Selection methods
# https://www.upgrad.com/blog/how-to-choose-a-feature-selection-method-for-machine-learning/
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/
# https://www.kdnuggets.com/2021/12/alternative-feature-selection-methods-machine-learning.html
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

if __name__ == '__main__':


    #########################
    ### Data description ####
    #########################


    # Load file from excel
    df = pd.read_excel(r'CTG.xls')
    # Drop last n rows (Nan)
    df.drop(df.tail(3).index, inplace=True)
    # Drop first n rows (Nan)
    df.drop(df.head(1).index, inplace=True)
    # Drop unused columns
    df.drop(columns=["FileName", "Date", "SegFile", "b", "e", "LBE", "NSP", "A",
                     "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP", "DR"], axis=1, inplace=True)
    # Check if there is any Null value
    # print(df.isnull().sum())
    # Data distribution
    # yb.target.class_balance(df.CLASS, labels=["A", "B", "C", "D", "SH", "AD", "DE", "LD", "FS", "SUSP"])
    # Changing format of data
    data = df.to_numpy()
    # Splitting data into features and labels
    X = df.iloc[:, 0: -1]
    y = df.iloc[:, -1]


    #####################
    ### Normalization ###
    #####################
    
    # Honestly I am not sure if this is a good idea. Probably should ask.
    # This takes information from data. In my case it is very important.


    #########################
    ### Feature selection ###
    #########################

    # Not sure if I should do this before before oversampling.
    # Talked with supervisor and he told me to do this after oversampling on all data.
    # Need to change values in future.

    # Need to check other models too.
    # model = KNeighborsClassifier(n_neighbors=5)
    # Search for scoring https://scikit-learn.org/stable/modules/model_evaluation.html
    # sfs = SFS(model,
    #           k_features=(1,21),
    #           forward=False,
    #           floating=True,
    #           scoring='balanced_accuracy',
    #           cv=10,
    #           n_jobs=-1)
    # sfs.fit(X,y)    
    # Print results
    # print(sfs.k_feature_names_)
    # fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    # plt.title('Sequential Forward Selection')
    # plt.grid()
    # plt.show()
    # print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))

    # Use only choosed features
    # choosed_features = list(sfs.k_feature_names_)
    choosed_features = ['LB', 'AC', 'MLTV', 'DL', 'DP', 'Nmax', 'Mean']
    X = X[choosed_features]
    # Train, validation, test split
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 1 - train_ratio - validation_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio), stratify=y_test)


    ###############################
    ### Unbalanced data methods ###
    ###############################

    # Should check smote algorithm, compare and pick the best one!

    # Not sure if I am using this right in case of arguments!
    mdo = MDO(k=5, k1_frac=0.4, seed=0, prop=1, maj_int_min={
        'maj': [2],
        'min': [1,3,4,5,6,7,8,9,10]
    })
    X_train_resampled, y_train_resampled = mdo.fit_transform(np.copy(X_train), np.copy(y_train))
    # Numpy to pandas conversion
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    y_train_resampled = pd.Series(y_train_resampled, name=y_train.name)
    # Bonding variables
    samples_training = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    samples_validation = pd.concat([X_valid, y_valid], axis=1)
    samples_test = pd.concat([X_test, y_test], axis=1)

    ############
    ### Main ###
    ############
    pool_size = 6
    pool_classifiers = pool.PoolClassifiers("Pool", pool_size, 1, samples_training, samples_validation, number_samples_training=100).pool
    # Honestly not sure about size of final pool
    # margin_ordering(pool_classifiers, int(pool_size * 0.2), samples_validation, samples_test)
    # divp(pool_classifiers, samples_validation, samples_test)
    # agglomerative_clustering(pool_classifiers, samples_validation, samples_test, pool_size * 0.2)
    # ensemble_prunning(pool_classifiers, samples_training, samples_validation, samples_test, 0.3)
    # interclass_competences(pool_classifiers, samples_validation, samples_test)
    consensus_enora(pool_classifiers, samples_training, samples_validation, samples_test)
    


