import numpy as np
from combination_methods.majority_voting import majority_voting
import copy
from sklearn.metrics import f1_score

# Assumptions:
# 1. Shouldn't be homogeneous                                                         | X
# 2. Classifier can't be in more than one subset at the same time                     | :)
# 3. Ci = {cl, cm}, Cj = {cn} => prob(cl fails, cm fails) > prob(cl fails, cn fails)  | :)
#    Two classifiers from the same subset have bigger chance of fail classify of objects than those in separate
#    Classifiers in the same subsets are highly error correlated and in different error independent

# Info Agglomerative hierarchical clustering - https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019

def classifier_cluster_distance(classifier, cluster, classifier_distance_table):
    maximum = 0

    # Find distance between classifier and cluster
    for i in cluster:
        # Distance between
        distance = classifier_distance_table[classifier][i]
        maximum = distance if maximum < distance else maximum

    return maximum


def choose_classifier(cluster_number, clusters, classifier_distance_table):
    average_distances = []

    for classifier in clusters[cluster_number]:
        distances = []
        # Calculate distance between single classifier and clusters
        for cluster in clusters:
            # Dont measure distance to own cluster
            if cluster == clusters[cluster_number]:
                continue
            else:
                distances.append(classifier_cluster_distance(
                    classifier, cluster, classifier_distance_table))
        average_distances.append(np.average(distances))
    return clusters[cluster_number][np.argmax(average_distances)]


def cluster_distance_measure(clusters, classifier_distance_table):
    cluster_distance_table = np.ndarray((len(clusters), len(clusters)), dtype=float)

    # Calculate cluster distance table
    for ic1 in range(len(clusters)):
        for ic2 in range(len(clusters)):
            # Wont add the same clusters
            if ic1 != ic2:
                cluster_combinations = [(x, y) for x in clusters[ic1] for y in clusters[ic2]]
                # Find maximum distance in combinations
                maximum = 0
                for combination in cluster_combinations:
                    distance = classifier_distance_table[combination[0]][combination[1]]
                    maximum = distance if maximum < distance else maximum
                cluster_distance_table[ic1][ic2] = maximum
            else:
                cluster_distance_table[ic1][ic2] = 1
    
    return cluster_distance_table


def classifiers_clustering(classifiers_pool, classifier_distance_table):
    # Initial clusters
    clusters = [[i] for i in range(len(classifiers_pool))]
    cluster_distances = [0 for i in range(len(classifiers_pool))]
    max_distances = [0 for i in range(len(classifiers_pool))]
    combinations = []
    max_distance = []

    # Calculate clusters
    while len(clusters) > 2:
        # Calculate distances between clusters
        distance_table = cluster_distance_measure(clusters, classifier_distance_table)
        # Find coordinates of smallest distance
        x, y = np.unravel_index(distance_table.argmin(), distance_table.shape)
        # Remove cluster with higher index
        deleted_cluster = clusters.pop(np.max([x, y]))
        deleted_cluster_distances = cluster_distances.pop(np.max([x, y]))
        deleted_max_distances = max_distances.pop(np.max([x, y]))
        # Merge clusters
        clusters[np.min([x, y])] += deleted_cluster
        max_distances[np.min([x, y])] = distance_table[x][y] - np.min([cluster_distances[np.min([x, y])], deleted_cluster_distances])
        cluster_distances[np.min([x, y])] = distance_table[x][y]
        combinations.append(copy.deepcopy(clusters))
        max_distance.append(np.max(max_distances))

    best_combination = combinations[np.argmax(max_distance) - 1]
    return best_combination


def classifier_distance_measure(classifiers_pool, data_validation):
    compound_error_table = np.ndarray((len(classifiers_pool), len(classifiers_pool)), dtype=float)
    predictions = []
    X = data_validation.iloc[:, 0:-1]
    y_true = data_validation.iloc[:, -1]

    # Calculate predictions for each classifier
    for classifier in classifiers_pool:
        predictions.append(classifier.model.predict(X))
    # Create double false distance table
    for ip1 in range(len(predictions)):
        for ip2 in range(len(predictions)):
            compound_error = 0
            for index in range(len(data_validation)):
                # Both classifiers made bad decision
                if (predictions[ip1][index] != y_true.iloc[index]) & (predictions[ip2][index] != y_true.iloc[index]):
                    compound_error += 1
                # Not relevant
                else:
                    continue
            # Dont need result of the same classifiers
            compound_error_table[ip1][ip2] = 1 - (compound_error / len(data_validation)) if ip1 != ip2 else 1

    return compound_error_table


def agglomerative_clustering(classifiers_pool, data_validation, data_test):
    # Calculate distances between classifiers
    distance_table = classifier_distance_measure(classifiers_pool, data_validation)
    # Create ensembles of classifiers
    clusters = classifiers_clustering(classifiers_pool, distance_table)
    # Create candidate ensemble
    candidate_ensemble = []
    for i in range(len(clusters)):
        candidate_ensemble.append(choose_classifier(i, clusters, distance_table))
    clusters.append(candidate_ensemble)
    # Choose best ensemble
    scores = []
    for cluster in clusters:
        candidate_cluster = []
        # Create pool from indexes
        for classifier_index in cluster:
            candidate_cluster.append(classifiers_pool[classifier_index])
        scores.append(majority_voting(candidate_cluster, data_validation, 'accuracy'))
    new_pool = []
    for i in clusters[np.argmax(scores)]:
        new_pool.append(classifiers_pool[i])

    # To compare, probably will change in future
    # print(majority_voting(classifiers_pool, data_test, 'accuracy'))
    # print(majority_voting(new_pool, data_test, 'accuracy'))
    print(f'New pool size: {len(new_pool)}')
    return majority_voting(new_pool, data_test, 'probes')
