import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Simple approach
def majority_voting(classifiers_pool, samples_test, type):
    result = []
    votes_classifier = []
    X = samples_test.iloc[:, 0:-1]
    y_true = samples_test.iloc[:, -1]

    # Object votes for classifier
    for classifier in classifiers_pool:
        votes_classifier.append(classifier.model.predict(X))
    # Classifier votes for object
    votes_object = np.array(votes_classifier).T
    # Calculate votes
    for votes in votes_object:
        result.append(max(set(votes), key = list(votes).count))

    if type == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, result)
    elif type == 'accuracy':
        return accuracy_score(y_true, result)
    elif type == 'f1':
        return f1_score(y_true, result, average='weighted')
    else:
        return result
    
