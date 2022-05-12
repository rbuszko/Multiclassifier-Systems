import numpy as np
import pandas as pd
import math

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from combination_methods.majority_voting import majority_voting
from sklearn.metrics import accuracy_score

def calculate_oo_distance(x1, x2, c1, c2):
    return c1 * math.exp(-c2 * euclidean(np.array(x1), np.array(x2)))


def create_local_confusion_matrix(object, classifier, data, classes_number):
    local_confusion_matrix = np.zeros((classes_number, classes_number), dtype=float)
    X = data.iloc[:, 0:-1]
    y_true = data.iloc[:, -1]

    # Clasify objects based on specyfic classifier
    objects_classes = classifier.model.predict(X)
    # Complete matrix with weightned values
    for i in range(len(objects_classes)):
        local_confusion_matrix[int(y_true.iloc[i]) - 1][int(objects_classes[i]) - 1] += calculate_oo_distance(object.iloc[0:-1], X.iloc[i], 1, 1)

    return local_confusion_matrix


def create_weights_final_ensemble_index(object, label, pool, data, classes_number, k_mistake, k_top):
    final_ensemble = []
    clusters_row_indexes = [[] for _ in range(classes_number)]
    matrix_label_rows_classifier = []
    
    # Get local confusion matrix for each classifier
    for i in range(len(pool)):
        local_confusion_matrix = create_local_confusion_matrix(object, pool[i], data, classes_number)
        # Label - 1 cause first index is 0
        matrix_label_rows_classifier.append((local_confusion_matrix[int(label - 1)], i))
    # Create clusters of k_mistake classifier indexes
    for i in range(classes_number):
        matrix_label_rows_classifier.sort(key = lambda x: x[0][i], reverse=True)
        for j in range(k_mistake):
            clusters_row_indexes[i].append(matrix_label_rows_classifier[j])
    # Delete label cluster
    clusters_row_indexes.pop(int(label - 1))
    # From each cluster pick classifier with best accuracy, 
    for cluster in clusters_row_indexes:
        cluster.sort(key = lambda x: x[0][int(label - 1)], reverse=True)
        for i in range(k_top):
            if cluster[i][1] not in final_ensemble:
                final_ensemble.append(cluster[i][1])

    return matrix_label_rows_classifier, final_ensemble


def interclass_competences(pool_classifiers, samples_validation, samples_test, k_mistake, k_top):
    result = []
    classes_number = 10

    # Find "classes" for each sample
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(samples_validation.iloc[:, 0:-1], samples_validation.iloc[:, -1])
    knn_labels = model.predict(samples_validation.iloc[:, 0:-1])
    # Dynamic classification
    for i in range(len(samples_test)):
        matrix_label_rows_classifier, final_ensemble = create_weights_final_ensemble_index(
            samples_test.iloc[i],
            knn_labels[i],
            pool_classifiers,
            samples_validation,
            classes_number,
            k_mistake,
            k_top)
        # Sort matrix_label_rows_classifier
        matrix_label_rows_classifier.sort(key = lambda x: x[1])
        # Calculate support from each classifier
        final_support = [0 for _ in range(classes_number)]
        for c_index in final_ensemble:
            support = pool_classifiers[c_index].model.predict_proba(samples_test.iloc[i, 0:-1].to_frame().T)
            # weights = matrix_label_rows_classifier[c_index][0]
            final_support += support #* weights
        result.append(np.argmax(final_support) + 1)
    
    print(f"ACCURACY ALL CLASSIFIERS: {majority_voting(pool_classifiers, samples_test, 'accuracy')}")
    print(f"ACCURACY BY SOME: {accuracy_score(samples_test.iloc[:, -1], result)}")
