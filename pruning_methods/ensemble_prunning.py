from dataclasses import dataclass
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from combination_methods.majority_voting import majority_voting

# Why there are wrong formulas for measures?
# Should I use such document?

# Assumptions:
# 1. Creation of basic classifiers is not the same cause of different approaches | X
# 2. During classifier prunning I used majority voting to check accuracy after classifier removal

@dataclass
class contingency_structure:
    a: int # c1 true,  c2 true
    b: int # c1 true,  c2 false
    c: int # c1 false, c2 true
    d: int # c1 false, c2 false


def diversity_measures_centroids(centroids, data):
    contingency_table = np.ndarray((len(centroids), len(centroids)), dtype=contingency_structure)
    y_true = data.iloc[:, -1]

    # Create contingency table
    for ip1 in range(len(centroids)):
        for ip2 in range(len(centroids)):
            object = contingency_structure(0, 0, 0, 0)
            for index in range(len(data)):
                if (centroids[ip1][index] == y_true.iloc[index]) & (centroids[ip2][index] == y_true.iloc[index]):
                    object.a += 1
                elif (centroids[ip1][index] == y_true.iloc[index]) & (centroids[ip2][index] != y_true.iloc[index]):
                    object.b += 1
                elif (centroids[ip1][index] != y_true.iloc[index]) & (centroids[ip2][index] == y_true.iloc[index]):
                    object.c += 1
                else:
                    object.d += 1
            contingency_table[ip1][ip2] = object

    return calculate_measures(contingency_table)


def diversity_measures_classifiers(classifiers_pool, data):
    contingency_table = np.ndarray((len(classifiers_pool), len(classifiers_pool)), dtype=contingency_structure)
    predictions = []
    X = data.iloc[:, 0:-1]
    y_true = data.iloc[:, -1]

    # Calculate predictions for each classifier
    for classifier in classifiers_pool:
        predictions.append(classifier.model.predict(X))
    # Create contingency table
    for ip1 in range(len(predictions)):
        for ip2 in range(len(predictions)):
            object = contingency_structure(0, 0, 0, 0)
            for index in range(len(data)):
                if (predictions[ip1][index] == y_true.iloc[index]) & (predictions[ip2][index] == y_true.iloc[index]):
                    object.a += 1
                elif (predictions[ip1][index] == y_true.iloc[index]) & (predictions[ip2][index] != y_true.iloc[index]):
                    object.b += 1
                elif (predictions[ip1][index] != y_true.iloc[index]) & (predictions[ip2][index] == y_true.iloc[index]):
                    object.c += 1
                else:
                    object.d += 1
            contingency_table[ip1][ip2] = object

    return calculate_measures(contingency_table)


def calculate_measures(contingency_table):
    M_corelation = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    M_kappa = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    
    m = contingency_table[0][0].a + contingency_table[0][0].b + contingency_table[0][0].c + contingency_table[0][0].d

    # Correlation Coefficient = (ad - bc) / sqrt((a + b)(a + c)(c + d)(b + d))
    # Kappa-statistic         = (((a + d) / m) - ((a + b)(a + c) + (c + d) (b + d) / m ^ 2)) / m
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            a = contingency_table[i][j].a
            b = contingency_table[i][j].b
            c = contingency_table[i][j].c
            d = contingency_table[i][j].d
            M_corelation[i][j] = ((a * d) - (b * c)) / math.sqrt((a + b) * (a + c) * (c + d) * (b + d))
            o1 = (a + d) / m
            o2 = (((a + b) * (a + c)) + ((c + d) * (b + d))) / math.pow(m, 2)
            M_kappa[i][j] = (o1 - o2) / (1 - o2)

    return M_corelation, M_kappa


def optimal_clusters(classifiers_pool, data_train):
    predictions = []

    # Data for clustering
    for classifier in classifiers_pool:
        predictions.append(classifier.model.predict(data_train.iloc[:, 0:-1]))
    # Find optimal number of clusters
    previous_disagreement = 1
    for cluster_number in range(len(classifiers_pool) + 1)[2:]:
        kmeans = KMeans(n_clusters=cluster_number).fit(predictions)
        # We take data closest to centroid as centroid
        centroids = [predictions[i] for i in vq(kmeans.cluster_centers_, predictions)[0]]
        correlation, kappa = diversity_measures_centroids(centroids, data_train)
        average_disagreement = ((np.sum(correlation) + np.sum(kappa) - (2 * len(centroids))) / 2) / (len(centroids) * len(centroids) - len(centroids))
        # I take clusters before their disagreement detoriate
        if average_disagreement > previous_disagreement:
            return list(labels)
        previous_disagreement = average_disagreement
        labels = kmeans.labels_
    
    return list(kmeans.labels_)


def prune_clusters(classifiers_pool, clusters, data_validation, threshold):
    print(clusters)
    X = data_validation.iloc[:, 0:-1]
    y_true = data_validation.iloc[:, -1]

    # Calculate disagreement between  classifiers
    correlation, kappa = diversity_measures_classifiers(classifiers_pool, data_validation)
    measure = (correlation + kappa) / 2
    # Prune clusters
    for cluster in clusters:
        # Dictionary cluster - value
        classifier_accuracy = {}
        for classifier in cluster:
            classifier_accuracy[classifier] = classifiers_pool[classifier].model.score(X, y_true)
        # Sort dictionary by accuracies
        classifier_accuracy = dict(sorted(classifier_accuracy.items(), key=lambda item: item[1]))
        # Most accuracy classifier
        top = list(classifier_accuracy.keys())[-1]
        for classifier, accuracy in list(classifier_accuracy.items())[:-1]:
            # Accuracy of cluster
            old_accuracy = majority_voting([classifiers_pool[i] for i in cluster], data_validation, 'accuracy')
            # Get disagreement between current and best classifier
            disagreement = measure[classifier][top]
            # Remove classifier
            cluster.remove(classifier)
            # TODO: Not sure if should act on threshold, saw in examples that it may lead
            # to better solution
            if disagreement <= threshold:
                current_cluster_accuracy = majority_voting([classifiers_pool[i] for i in cluster], data_validation, 'accuracy')
                if old_accuracy > current_cluster_accuracy:
                    cluster.append(classifier)

    print(clusters)
                


        


def ensemble_prunning(classifiers_pool, data_train, data_validation, data_test, threshold):
    # Get optimal clusters
    cluster_labels = optimal_clusters(classifiers_pool, data_train)
    clusters = [[] for i in range(np.max(cluster_labels) + 1)]
    for i in range(len(cluster_labels)):
        clusters[cluster_labels[i]].append(i)
    # Prune clusters
    prune_clusters(classifiers_pool, clusters, data_validation, threshold)
    # Get diversity measures
    # M_corelation, M_Kappa = diversity_measures(classifiers_pool, data_validation)
    print("Need to implement classifier weightening at the start if I want to implement this")

        


