# Libraries
from random import seed
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import yellowbrick as yb
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from multi_imbalance.resampling.mdo import MDO
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import copy
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from scipy import stats
import scikit_posthocs as sp
from combination_methods.majority_voting import majority_voting
from statsmodels.stats.contingency_tables import mcnemar

# My libraries
import classifiers.pool_classifiers as pool
from pruning_methods.margin_ordering import margin_ordering                   # Document 1
from pruning_methods.divp import divp                                         # Document 4
from pruning_methods.agglomerative_clustering import agglomerative_clustering # Document 2
from pruning_methods.ensemble_prunning import ensemble_prunning               # Document 3
from pruning_methods.interclass_competences import interclass_competences     # Document 7
from pruning_methods.consensus_enora import consensus_enora                   # Document 6
from pruning_methods.myprop import friendship
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


    X = pd.DataFrame(preprocessing.normalize(X), columns=X.columns)
    X.index += 1

    ########################################
    # Repeated stratified cross validation #
    ########################################

    # List to check
    Values_accuracy = []
    Values_f1 = []
    # Values_roc = []

    # Change X and y to numpy array, need to rskf
    Xnp = np.array(X)
    ynp = np.array(y)
    # Rskf implementation
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=8)
    # Main loop
    classifier_type = 3 ##### TYP KLASYFIKATORA 1, 2, 3
    for train_index, test_index in rskf.split(Xnp, ynp):
        X_train, X_test = Xnp[train_index], Xnp[test_index]
        y_train, y_test = ynp[train_index], ynp[test_index]

        # Going for 80 | 10 | 10 ratio
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
        for vi, ti in sss.split(X_test, y_test):
            X_valid, X_test = X_test[vi], X_test[ti]
            y_valid, y_test = y_test[vi], y_test[ti]
            break
        
        ###########################
        # Unbalanced data methods #
        ###########################
        
        # Maybe will search for better results
        mdo = MDO(k=5, k1_frac=0.4, seed=1, prop=1, maj_int_min={
            'maj': [2],
            'min': [1,3,4,5,6,7,8,9,10]
        })
        
        X_train_resampled, y_train_resampled = mdo.fit_transform(X_train, y_train)
        X_valid_resampled, y_valid_resampled = mdo.fit_transform(X_valid, y_valid)
       
        sm = SMOTE(random_state=2, k_neighbors=4)
        X_valid_resampled, y_valid_resampled = sm.fit_resample(X_valid_resampled, y_valid_resampled)

        # Numpy to pandas conversion
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
        y_train_resampled = pd.Series(y_train_resampled, name=y.name)
        X_valid_resampled = pd.DataFrame(X_valid_resampled, columns=X.columns)
        y_valid_resampled = pd.Series(y_valid_resampled, name=y.name)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        y_test = pd.Series(y_test, name=y.name)
        
        #####################
        # Feature selection #
        #####################


        X_fs = pd.concat([X_train_resampled, X_valid_resampled], axis=0)
        y_fs = pd.concat([y_train_resampled, y_valid_resampled], axis=0)

        # Need to check other models too.
        if classifier_type == 1:
            model = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 2:
            model = MLPClassifier(random_state=0)
        else:
            model = DecisionTreeClassifier(random_state=0)
        # Search for scoring https://scikit-learn.org/stable/modules/model_evaluation.html
        sfs = SFS(model,
                  k_features=(1,21),
                  forward=False,
                  floating=True,
                  scoring='f1_weighted',
                #   scoring='accuracy',
                  cv=5,
                  n_jobs=-1)
        sfs.fit(X_fs,y_fs)    
        choosed_features = list(sfs.k_feature_names_)
        # fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
        # plt.title('Sequential Forward Selection')
        # plt.grid()
        # plt.show()

        X_fs = X_fs[choosed_features]
        X_train_resampled = X_train_resampled[choosed_features]
        X_valid_resampled = X_valid_resampled[choosed_features]
        X_test = X_test[choosed_features]

        # Proper format
        samples_training = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        samples_validation = pd.concat([X_valid_resampled, y_valid_resampled], axis=1)
        samples_test = pd.concat([X_test, y_test], axis=1)

    #     ####################
    #     # Check classifier #
    #     ####################
        
        
    #     # Model to check
    #     if classifier_type == 1:
    #         model = KNeighborsClassifier(n_neighbors=5)
    #     elif classifier_type == 2:
    #         model = MLPClassifier(random_state=0)
    #     else:
    #         model = DecisionTreeClassifier(random_state=0)
        
    #     model.fit(X_fs, y_fs)
    #     prediction = model.predict(X_test)
    #     Values_accuracy.append(accuracy_score(y_test, prediction))
    #     Values_f1.append(f1_score(y_test, prediction, average='weighted'))

        
    #     ###############################
    #     # Check pool size + n_samples #
    #     ###############################


        # for pool_size in [50, 100, 150]:
        #     for n_samples in [2000, 3500, 5000]:
        #         pool_classifiers = pool.PoolClassifiers("Pool", pool_size, classifier_type, samples_training, samples_validation, number_samples_training=n_samples).pool
        #         # friendship(pool_classifiers, 0.8, classifier_type, 5, samples_training, samples_validation, samples_test, 1)
        #         # friendship(pool_classifiers, 0.7, classifier_type, 20, samples_training, samples_validation, samples_test, 1)
        #         # new_pool = margin_ordering(pool_classifiers, samples_validation, samples_test)
        #         Values_accuracy.append(majority_voting(pool_classifiers, samples_test, 'accuracy'))
        #         Values_f1.append(majority_voting(pool_classifiers, samples_test, 'f1'))

        #############################
        # Check prunning algorithms #
        #############################

        pool_size = 100
        ns_samples = 2000
        y_true = samples_test.iloc[:, -1]
        pool_classifiers = pool.PoolClassifiers("Pool", pool_size, classifier_type, samples_training, samples_validation, number_samples_training=ns_samples).pool

        # predictions = margin_ordering(pool_classifiers, samples_validation, samples_test)
        # predictions = agglomerative_clustering(pool_classifiers, samples_validation, samples_test)
        # predictions = divp(pool_classifiers, samples_validation, samples_test)
        # predictions = interclass_competences(pool_classifiers, samples_validation, samples_test, 2, 1)
        # predictions = consensus_enora(pool_classifiers, 5, samples_training, samples_validation, samples_test, 0.5)
        predictions = friendship(pool_classifiers, 0.75, classifier_type, 5, samples_training, samples_validation, samples_test)
        print(accuracy_score(y_true, predictions))
        print(f1_score(y_true, predictions, average='weighted'))
        Values_accuracy.append(accuracy_score(y_true, predictions))        
        Values_f1.append(f1_score(y_true, predictions, average='weighted'))
        
        print("GOT")

    # pools_size = [50, 100, 150]
    # ns_samples = [2000, 3500, 5000]
    # i = 0
    # for pool_size in pools_size:
    #         for n_samples in ns_samples:
    #             print(f'Pool size {pool_size}')
    #             print(f'n_samples {n_samples}')
    #             acc = [Values_accuracy[i], Values_accuracy[i + 9], Values_accuracy[i + 18], Values_accuracy[i + 27], Values_accuracy[i + 36], Values_accuracy[i + 45], Values_accuracy[i + 54], Values_accuracy[i + 63], Values_accuracy[i + 72], Values_accuracy[i + 81]]
    #             f1 = [Values_f1[i], Values_f1[i + 9], Values_f1[i + 18], Values_f1[i + 27], Values_f1[i + 36], Values_f1[i + 45], Values_f1[i + 54], Values_f1[i + 63], Values_f1[i + 72], Values_f1[i + 81]]
    #             print(f'Acc: {acc}')
    #             print(f'F1 : {f1}')
    #             i += 1

    
    ###################
    # Statistic tests #
    ###################

    #########
    # Model #
    #########

    # rng = np.random.default_rng()
    # x = stats.norm.rvs(loc=5, scale=3, size=100, random_state=rng)
    # shapiro_test = stats.shapiro(x)
    # print(x)
    # print(Values_accuracy)
    # print(Values_f1)

    # Knn_acc = [0.7089201877934272, 0.7323943661971831, 0.784037558685446, 0.7417840375586855, 0.704225352112676, 0.7370892018779343, 0.7323943661971831, 0.6948356807511737, 0.7089201877934272, 0.755868544600939]
    # Knn_F1 = [0.6982163734587999, 0.7247727715952234, 0.7759793528531653, 0.7388894040880262, 0.6977345579733822, 0.7326555270331174, 0.7198793027121679, 0.6886257412653721, 0.6947472889286967, 0.7562326873827456]
    # Dtc_acc = [0.7746478873239436, 0.7652582159624414, 0.7464788732394366, 0.8075117370892019, 0.812206572769953, 0.755868544600939, 0.755868544600939, 0.7464788732394366, 0.6948356807511737, 0.784037558685446]
    # Dtc_F1 = [0.7594365751648675, 0.7689810645169921, 0.753020086489862, 0.8088487752701542, 0.8148841177334258, 0.7579865825097166, 0.7589509128448647, 0.7468300379683914, 0.6932309888087095, 0.7869022264671485]
    # Mlp_acc = [0.6056338028169014, 0.6197183098591549, 0.6619718309859155, 0.6619718309859155, 0.5868544600938967, 0.6525821596244131, 0.6525821596244131, 0.5633802816901409, 0.6619718309859155, 0.6431924882629108]
    # Mlp_f1 = [0.6102540403948855, 0.6194092599217208, 0.6731180564776525, 0.6640091063320843, 0.5969573011647301, 0.6536673020772165, 0.6593191422413189, 0.5733903698272892, 0.6742551156773468, 0.6540411316239094]

    # round_to_tenths = [round(num, 3) for num in Mlp_f1]
    # print(round_to_tenths)

    # print(stats.wilcoxon(Knn_F1, Mlp_acc))

    ##################
    # Pool + Samples #
    ##################


    # # Pool size 50
    # # n_samples 200
    # A_p50_n200 = [0.7417840375586855, 0.755868544600939, 0.7652582159624414, 0.7746478873239436, 0.8169014084507042, 0.7370892018779343, 0.784037558685446, 0.7323943661971831, 0.7417840375586855, 0.7981220657276995]
    # F_p50_n200 = [0.7316014282589854, 0.7525752010971595, 0.7697502947109445, 0.7694424611001033, 0.8121320353180687, 0.726314314451715, 0.7897873381699428, 0.7152063632099097, 0.7247459671077632, 0.798753132343971]
    # # Pool size 50
    # # n_samples 600
    # A_p50_n600 = [0.7605633802816901, 0.7511737089201878, 0.8215962441314554, 0.8028169014084507, 0.8873239436619719, 0.7464788732394366, 0.8497652582159625, 0.7793427230046949, 0.7652582159624414, 0.812206572769953]
    # F_p50_n600 = [0.7555399945578195, 0.7442720034437729, 0.8238719257902456, 0.79899031708528, 0.8834358244648902, 0.733734157014974, 0.8467181055209224, 0.7747520476393716, 0.7554827582175215, 0.8132149624906615]
    # # Pool size 50
    # # n_samples 1000
    # A_p50_n1000 = [0.7652582159624414, 0.784037558685446, 0.8262910798122066, 0.8215962441314554, 0.8967136150234741, 0.7887323943661971, 0.8403755868544601, 0.7981220657276995, 0.7793427230046949, 0.8403755868544601]
    # F_p50_n1000 = [0.7566582864300846, 0.7787495265650003, 0.8284976252429612, 0.8204790240608597, 0.8955564635439609, 0.779408060087486, 0.8388345131302878, 0.7870895412166293, 0.7727228507309472, 0.8421027434617749]
    # # Pool size 50
    # # n_samples 1500
    # A_p50_n1400 = [0.7652582159624414, 0.7887323943661971, 0.8591549295774648, 0.8403755868544601, 0.892018779342723, 0.7699530516431925, 0.8591549295774648, 0.8028169014084507, 0.7887323943661971, 0.8544600938967136]
    # F_p50_n1400 = [0.7531124165169817, 0.7895079308286005, 0.8579621115638134, 0.8394271687009671, 0.886092139334698, 0.7606148491300557, 0.8575354535521423, 0.7932111241243143, 0.7837058562781928, 0.8563647256993507]
    # # Pool size 50
    # # n_samples 2000
    # Acc: [0.7746478873239436, 0.7981220657276995, 0.8779342723004695, 0.8403755868544601, 0.892018779342723, 0.7887323943661971, 0.8497652582159625, 0.812206572769953, 0.7887323943661971, 0.8685446009389671]
    # F1 : [0.7640090068947097, 0.7950441581866963, 0.8782902542959392, 0.8398734784272888, 0.8889750447025394, 0.7797407339660861, 0.8505856244337001, 0.8057146481631698, 0.7881183388409714, 0.8676337238736461]
    
    # # Pool size 100
    # # n_samples 200
    # A_p100_n200 = [0.7417840375586855, 0.7511737089201878, 0.7887323943661971, 0.784037558685446, 0.8309859154929577, 0.7417840375586855, 0.7981220657276995, 0.755868544600939, 0.755868544600939, 0.8028169014084507]
    # F_p100_n200 = [0.7314283746520364, 0.74541129652686, 0.7899184410219838, 0.7783332674541619, 0.8263033835144986, 0.7278779334915435, 0.8033939847691628, 0.7369859168891363, 0.7348888473318269, 0.8024824947229787]
    # # Pool size 100
    # # n_samples 600
    # A_p100_n600 = [0.7652582159624414, 0.7652582159624414, 0.8215962441314554, 0.812206572769953, 0.8732394366197183, 0.755868544600939, 0.8403755868544601, 0.7981220657276995, 0.7746478873239436, 0.8262910798122066]
    # F_p100_n600 = [0.7519663015809089, 0.7623490732281407, 0.8206440990363777, 0.8097778631818059, 0.8688920700704584, 0.7445613935855042, 0.8387505733477616, 0.7836681715489446, 0.7672497176066805, 0.825916923324082]
    # # Pool size 100
    # # n_samples 1000
    # A_p100_n1000 = [0.7746478873239436, 0.784037558685446, 0.8262910798122066, 0.8262910798122066, 0.9014084507042254, 0.7746478873239436, 0.8356807511737089, 0.7981220657276995, 0.7746478873239436, 0.8591549295774648]
    # F_p100_n1000 = [0.7662616639054639, 0.7802364557865022, 0.8284731396430719, 0.8251565866587103, 0.9000143964115925, 0.7654165216576873, 0.8371417385020054, 0.7871042851140833, 0.7681463722522317, 0.8579272571358103]
    # # Pool size 100
    # # n_samples 1400
    # # Pool size 100
    # # n_samples 1500
    # Acc: [0.7793427230046949, 0.7934272300469484, 0.8732394366197183, 0.8403755868544601, 0.8967136150234741, 0.784037558685446, 0.8685446009389671, 0.8028169014084507, 0.784037558685446, 0.8732394366197183]
    # F1 : [0.7692658365652855, 0.7871169247437464, 0.8703426578437862, 0.8398734784272888, 0.8918341137918055, 0.7748392980868204, 0.868592510709921, 0.7912462527259746, 0.7805332863725146, 0.8737201373513869]
    # # Pool size 100
    # # n_samples 2000
    # Acc: [0.7934272300469484, 0.7981220657276995, 0.863849765258216, 0.8403755868544601, 0.8873239436619719, 0.784037558685446, 0.8544600938967136, 0.7981220657276995, 0.7887323943661971, 0.8685446009389671]
    # F1 : [0.7863175273536193, 0.7945967246375037, 0.8636525169987157, 0.8398734784272888, 0.8830551414292992, 0.7749310602643236, 0.8564856797754183, 0.786848862118516, 0.7850525883178449, 0.86813543085533]

    # # Pool size 150
    # # n_samples 200
    # A_p150_n200 = [0.7417840375586855, 0.7323943661971831, 0.7793427230046949, 0.7934272300469484, 0.8450704225352113, 0.7370892018779343, 0.8028169014084507, 0.7276995305164319, 0.7605633802816901, 0.8075117370892019]
    # F_p150_n200 = [0.7284845019971674, 0.725513478521517, 0.7847874759528531, 0.7883168543509013, 0.8388892729508178, 0.7223720772825247, 0.8072818253693829, 0.7046825124296594, 0.7417225578349218, 0.8078597465791039]
    # # Pool size 150
    # # n_samples 600
    # A_p150_n600 = [0.7652582159624414, 0.7699530516431925, 0.8403755868544601, 0.8169014084507042, 0.8732394366197183, 0.7605633802816901, 0.8497652582159625, 0.7887323943661971, 0.7793427230046949, 0.8309859154929577]
    # F_p150_n600 = [0.7555127722319105, 0.7665100236799818, 0.8397228983599738, 0.8144544704508322, 0.8688920700704584, 0.7482629984911074, 0.8471113069915663, 0.7732031019289455, 0.7701513451477324, 0.8280023052331212]
    # # Pool size 150
    # # n_samples 1000
    # A_p150_n1000 = [0.7652582159624414, 0.7746478873239436, 0.8403755868544601, 0.8262910798122066, 0.9061032863849765, 0.7699530516431925, 0.8450704225352113, 0.7981220657276995, 0.7793427230046949, 0.8497652582159625]
    # F_p150_n1000 = [0.75586942512245, 0.7678695468970471, 0.8422060257482245, 0.8251565866587103, 0.9043930993760311, 0.7630456076510954, 0.8441539981702008, 0.785417923694643, 0.7750917574861238, 0.8488073076606862]
    # # Pool size 150
    # # n_samples 1400
    # A_p150_n1400 = [0.7793427230046949, 0.7793427230046949, 0.8403755868544601, 0.8403755868544601, 0.9107981220657277, 0.7887323943661971, 0.8544600938967136, 0.7934272300469484, 0.7887323943661971, 0.8779342723004695]
    # F_p150_n1400 = [0.7690483932169186, 0.7760790821018877, 0.8410318637340537, 0.8391225945670946, 0.9032589299822237, 0.7795464984763829, 0.8549226084571485, 0.7809664612157914, 0.7815019081593559, 0.8768337460302854]
    # # Pool size 150
    # # n_samples 1500
    # Acc: [0.7746478873239436, 0.7887323943661971, 0.8591549295774648, 0.8356807511737089, 0.8967136150234741, 0.7746478873239436, 0.8544600938967136, 0.7981220657276995, 0.7793427230046949, 0.8732394366197183]
    # F1 : [0.7661617419118951, 0.781815211839081, 0.8603600348329942, 0.8349204466408428, 0.8923043871755039, 0.7650924177850441, 0.8551279888565847, 0.7851342944765058, 0.7743108437581577, 0.8737201373513869]
    # # Pool size 150
    # # n_samples 2000
    # Acc: [0.784037558685446, 0.7981220657276995, 0.863849765258216, 0.8403755868544601, 0.8967136150234741, 0.7981220657276995, 0.8403755868544601, 0.812206572769953, 0.7887323943661971, 0.8685446009389671]
    # F1 : [0.7773960289243128, 0.7943346505060731, 0.8643146092101036, 0.8398734784272888, 0.8920247566304901, 0.7894267682288043, 0.8415062211573071, 0.8000285494697933, 0.7846201968144146, 0.8676433315176253]

    # namesA = ['A_p50_n200', 'A_p50_n600', 'A_p50_n1000', 'A_p100_n200', 'A_p100_n600', 'A_p100_n1000', 'A_p150_n200', 'A_p150_n600', 'A_p150_n1000']
    # namesF = ['F_p50_n200', 'F_p50_n600', 'F_p50_n1000', 'F_p100_n200', 'F_p100_n600', 'F_p100_n1000', 'F_p150_n200', 'F_p150_n600', 'F_p150_n1000']
    # tableA = [A_p50_n200, A_p50_n600, A_p50_n1000, A_p100_n200, A_p100_n600, A_p100_n1000, A_p150_n200, A_p150_n600, A_p150_n1000]
    # tableF = [F_p50_n200, F_p50_n600, F_p50_n1000, F_p100_n200, F_p100_n600, F_p100_n1000, F_p150_n200, F_p150_n600, F_p150_n1000]
    # for t1_i in range(len([A_p50_n200, A_p50_n600, A_p50_n1000, A_p100_n200, A_p100_n600, A_p100_n1000, A_p150_n200, A_p150_n600, A_p150_n1000])):
    #     for t2_i in range(len([A_p50_n200, A_p50_n600, A_p50_n1000, A_p100_n200, A_p100_n600, A_p100_n1000, A_p150_n200, A_p150_n600, A_p150_n1000])):
    #         print("____________________")
    #         if t1_i == t2_i:
    #             continue
    #         print(f'Average {namesF[t1_i]}: {np.average(tableF[t1_i])}')
    #         print(f'Average {namesF[t2_i]}: {np.average(tableF[t2_i])}')
    #         print(stats.wilcoxon(tableF[t1_i], tableF[t2_i]))

    # print(stats.wilcoxon(A_p150_n1000, A_p50_n1400))

    # # round_to_tenths = [round(num, 3) for num in F_p150_n1000]
    # # print(round_to_tenths)


    #################################
    # Result of prunning algorithms #
    #################################


    print(Values_accuracy)
    print(Values_f1)

    # # Margin ordering, [16, 11, 63, 15, 9, 33, 19, 9, 24, 15] | GIT
    # Acc_1 = [0.7934272300469484, 0.7793427230046949, 0.8732394366197183, 0.8450704225352113, 0.8591549295774648, 0.7934272300469484, 0.8591549295774648, 0.8075117370892019, 0.7793427230046949, 0.8591549295774648]
    # F1_1 = [0.7873802100397584, 0.7758773187355984, 0.8730421883602181, 0.8466368326342176, 0.8564079323427172, 0.7877430844701682, 0.8599143324998457, 0.8042315448418734, 0.7800340360447009, 0.8588502170605584]
    # # Agglomerative hierarchical clustering, [10, 5, 4, 4, 5, 90, 3, 4, 4, 13] | :()
    # Acc_2 = [0.7699530516431925, 0.7699530516431925, 0.7746478873239436, 0.784037558685446, 0.8403755868544601, 0.7746478873239436, 0.8356807511737089, 0.7887323943661971, 0.7370892018779343, 0.8544600938967136]
    # F1_2 = [0.7634925914117472, 0.7623918799329794, 0.7805077088162147, 0.7817970199454968, 0.8392482015202395, 0.7659343551355449, 0.8337686186943664, 0.7865814569983357, 0.7318998561569032, 0.8542306143553982]
    # # DivP, [6, 11, 20, 7, 13, 16, 13, 6, 14, 9] # TO TRZEBA OBLICZYĆ | GIT
    # Acc_3 = [0.7887323943661971, 0.7934272300469484, 0.8497652582159625, 0.8169014084507042, 0.8826291079812206, 0.7699530516431925, 0.8591549295774648, 0.812206572769953, 0.812206572769953, 0.8356807511737089]
    # F1_3 = [0.7764858833086136, 0.7905752475517027, 0.8501234349930629, 0.815175780867293, 0.881280270645104, 0.764969787187554, 0.8593059413570717, 0.8091357748235184, 0.804403069496711, 0.8337539038374352]
    # # Interclass competences, [9, 9, 9, 9, 9, 9, 8, 9 ,9, 8] |GIT
    # Acc_4 = [0.7652582159624414, 0.7887323943661971, 0.8262910798122066, 0.8169014084507042, 0.8732394366197183, 0.7793427230046949, 0.8497652582159625, 0.8169014084507042, 0.7746478873239436, 0.8309859154929577]
    # F1_4 = [0.7627198123122293, 0.7820546505517495, 0.8282567044293774, 0.8148586883409613, 0.871713246384054, 0.7793427230046949, 0.8501282127920201, 0.8115055981652145, 0.7689552076606223, 0.8279782950940331]
    # # Consensus enora, [5, 7, 4, 4, 5, 5, 4, 3 ,7, 5] | GIT
    # Acc_5 = [0.7934272300469484, 0.7981220657276995, 0.863849765258216, 0.8403755868544601, 0.8873239436619719, 0.784037558685446, 0.8544600938967136, 0.7981220657276995, 0.7887323943661971, 0.8685446009389671]
    # F1_5 = [0.7863175273536193, 0.7945967246375037, 0.8636525169987157, 0.8398734784272888, 0.8830551414292992, 0.7749310602643236, 0.8564856797754183, 0.786848862118516, 0.7850525883178449, 0.86813543085533]
    # # Friendship, [14, 12, 16, 18, 14, 18, 12, 14 ,14, 14] | GIT
    # Acc_6 = [0.7605633802816901, 0.755868544600939, 0.7793427230046949, 0.7981220657276995, 0.8215962441314554, 0.755868544600939, 0.8262910798122066, 0.7464788732394366, 0.755868544600939, 0.812206572769953]
    # F1_6 = [0.7503354022775143, 0.7506513205098738, 0.7731694152177212, 0.7970770380795387, 0.8125977251357602, 0.7470228461027985, 0.8216026904045426, 0.7269326378294971, 0.7465725024410471, 0.8093880396947472]
    # # Bagging | GIT
    # Acc_7 = [0.7746478873239436, 0.784037558685446, 0.8262910798122066, 0.8262910798122066, 0.9014084507042254, 0.7746478873239436, 0.8356807511737089, 0.7981220657276995, 0.7746478873239436, 0.8591549295774648]
    # F1_7 = [0.7662616639054639, 0.7802364557865022, 0.8284731396430719, 0.8251565866587103, 0.9000143964115925, 0.7654165216576873, 0.8371417385020054, 0.7871042851140833, 0.7681463722522317, 0.8579272571358103]
    # # Dtc 1 | GIT
    # Acc_8 = [0.7746478873239436, 0.7652582159624414, 0.7464788732394366, 0.8075117370892019, 0.812206572769953, 0.755868544600939, 0.755868544600939, 0.7464788732394366, 0.6948356807511737, 0.784037558685446]
    # F1_8 = [0.7594365751648675, 0.7689810645169921, 0.753020086489862, 0.8088487752701542, 0.8148841177334258, 0.7579865825097166, 0.7589509128448647, 0.7468300379683914, 0.6932309888087095, 0.7869022264671485]
    # # Dtc 2 | GIT
    # Acc_9 = [0.6666666666666666, 0.6807511737089202, 0.7089201877934272, 0.6572769953051644, 0.7652582159624414, 0.6713615023474179, 0.7417840375586855, 0.6666666666666666, 0.7370892018779343, 0.7323943661971831]
    # F1_9 = [0.6589253778757157, 0.6746034392561302, 0.6952659829074097, 0.6498838883643218, 0.7458422471534464, 0.6627072939842219, 0.7316628132597394, 0.6485049513631167, 0.7308448138782636, 0.7280587105736839]
    # # round_to_tenths = [round(num, 3) for num in F1_6]
    # # print(round_to_tenths)

    # # print(np.average([14, 12, 16, 18, 14, 18, 12, 14 ,14, 14]))
    # namesA = ['Acc_1', 'Acc_2', 'Acc_3', 'Acc_4', 'Acc_5', 'Acc_6', 'Acc_7', 'Acc_8', 'Acc_9']
    # namesF = ['F1_1', 'F1_2', 'F1_3', 'F1_4', 'F1_5', 'F1_6', 'F1_7', 'F1_8', 'F1_9']
    # tableA = [Acc_1, Acc_2, Acc_3, Acc_4, Acc_5, Acc_6, Acc_7, Acc_8, Acc_9]
    # tableF = [F1_1, F1_2, F1_3, F1_4, F1_5, F1_6, F1_7, F1_8, F1_9]
    # for t1_i in range(len([Acc_1, Acc_2, Acc_3, Acc_4, Acc_5, Acc_6, Acc_7, Acc_8, Acc_9])):
    #     for t2_i in range(len([Acc_1, Acc_2, Acc_3, Acc_4, Acc_5, Acc_6, Acc_7, Acc_8, Acc_9])):
    #         print("____________________")
    #         print("____________________")
    #         if t1_i == t2_i:
    #             continue
    #         print(f'Average {namesF[t1_i]}: {np.average(tableF[t1_i])}')
    #         print(f'Average {namesF[t2_i]}: {np.average(tableF[t2_i])}')
    #         print(stats.wilcoxon(tableF[t1_i], tableF[t2_i]))
