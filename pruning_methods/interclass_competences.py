from sklearn.metrics import confusion_matrix

# vl - classifier
# M - class labels
# x - features
# dl1(x) - vector support for class 1 <0,1>
# L - number of base classifiers
# i - classifier decision
# j - true label
# cii(vl | x) - classifier vl is capable to the correct classification of object from class i
# cij(cl | x) - classifier vl is not capable to the correct classificication of object from class i
# We gain this info from confusion matrix


# Vj - element from validation set of j'th class 
# Dvl_i - validation set assigned by classifier vl as i (i is the class)


def interclass_competences(pool_classifiers, samples_validation, samples_test):
    X = samples_validation.iloc[:, 0:-1]
    y_true = samples_validation.iloc[:, -1]

    print(confusion_matrix(y_true, pool_classifiers[0].model.predict(X)))