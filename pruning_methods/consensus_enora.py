from random import randrange
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_som.som import SOM
from sklearn.model_selection import train_test_split
from combination_methods.majority_voting import majority_voting
from sklearn.metrics import accuracy_score
from operator import itemgetter
from dataclasses import dataclass
from itertools import combinations
from geneticalgorithm import geneticalgorithm as ga

import pandas as pd
import numpy as np
import math

algorithm_param = {'max_num_iteration': 600,\
                   'population_size':22,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


@dataclass
class contingency_structure:
    a: int # c1 true,  c2 true
    b: int # c1 true,  c2 false
    c: int # c1 false, c2 true
    d: int # c1 false, c2 false


def construct_classifier_instance_matrix(pool_classifiers, data):
    result = [[] for i in range(len(pool_classifiers))]
    X = data.iloc[:, 0:-1]
    y_true = data.iloc[:, -1]

    for i in range(len(pool_classifiers)):
        prediction = pool_classifiers[i].model.predict(X)
        for j in range(len(prediction)):
            result[i].append(1) if (prediction[j] == y_true[j]) else result[i].append(0)

    return result


def multi_classifier_clustering(pool_classifiers, clusters_number, data):
    clustering_results = []
    X = construct_classifier_instance_matrix(pool_classifiers, data)

    r_number = randrange(100000)
    # Few clustering algorithms
    kmenas = KMeans(n_clusters=clusters_number, init='random', random_state=r_number).fit(X)
    kmenas_pp = KMeans(n_clusters=clusters_number, init='k-means++', random_state=r_number).fit(X)
    em = GaussianMixture(n_components=clusters_number, random_state=r_number).fit(X)
    som = SOM(m=clusters_number, n=1, dim=len(data), random_state=r_number)
    som.fit(np.array(X))
    clustering_results.append(list(kmenas.labels_))
    clustering_results.append(list(kmenas_pp.labels_))
    clustering_results.append(list(som.predict(np.array(X))))
    clustering_results.append(list(em.predict(X)))
    # To have different results
    r_number += 1
    kmenas = KMeans(n_clusters=clusters_number, init='random', random_state=r_number).fit(X)
    kmenas_pp = KMeans(n_clusters=clusters_number, init='k-means++', random_state=r_number).fit(X)
    em = GaussianMixture(n_components=clusters_number, random_state=r_number).fit(X)
    som = SOM(m=clusters_number, n=1, dim=len(data), random_state=r_number)
    som.fit(np.array(X))
    clustering_results.append(list(kmenas.labels_))
    clustering_results.append(list(kmenas_pp.labels_))
    clustering_results.append(list(som.predict(np.array(X))))
    clustering_results.append(list(em.predict(X)))
    
    return clustering_results


def construct_co_association_matrix_and_propositions(pool_classifiers, cluster_number, data):
    clustering_propositions = multi_classifier_clustering(pool_classifiers, cluster_number, data)
    co_association_matrix = np.ndarray((len(clustering_propositions), len(clustering_propositions)), dtype=float)

    # Calculate co association matrix
    for i in range(len(clustering_propositions)):
        for j in range(len(clustering_propositions)):
            # Count elements repetition
            sum = 0
            for k in range(len(pool_classifiers)):
                if clustering_propositions[i][k] == clustering_propositions[j][k]:
                    sum += 1
            co_association_matrix[i][j] = sum / len(pool_classifiers)
    
    return co_association_matrix, clustering_propositions


def minimum_spanning_tree(pool_classifiers, cluster_number, data_train, data_validation, treshold):
    candidates = []
    co_association_matrix, clustering_propositions = construct_co_association_matrix_and_propositions(pool_classifiers, cluster_number, data_train)

    # Get candidats within threshold range
    for i in range(len(co_association_matrix)):
        for j in range(len(co_association_matrix)):
            if (co_association_matrix[i][j] > treshold) & (co_association_matrix[i][j] < 1):
                if (j, i) not in candidates:
                    candidates.append((i, j))
    # If there was no candidats add every cluster
    if len(candidates) < 1:
        for i in range(len(co_association_matrix)):
            for j in range(len(co_association_matrix)):
                if i == j:
                    candidates.append((i, j))
    clusters_accuracy = []
    # Combine clusters
    for candidate in candidates:
        clusters = [[] for i in range(cluster_number)]
        # If there is classifier which wasnt choosen by both clustering algorithms
        special_cluster = []
        for index in range(len(pool_classifiers)):
            if clustering_propositions[candidate[0]][index] == clustering_propositions[candidate[1]][index]:
                clusters[clustering_propositions[candidate[0]][index]].append(index)
            else:
                special_cluster.append(index)
        if special_cluster:
            clusters.append(special_cluster)
        # Remove empty ensembles
        clusters = list(filter(None, clusters))
        # TODO: Choose best clustering based on accuracy | Maybe will think about better idea in future
        votes = []
        for cluster in clusters:
            pool = []
            for label in cluster:
                pool.append(pool_classifiers[label])
            votes.append(majority_voting(pool, data_validation, 'labels'))
        votes_object = np.array(votes).T
        final_votes = []
        # Calculate votes from every cluster
        for votes in votes_object:
            final_votes.append(max(set(votes), key = list(votes).count))
        clusters_accuracy.append((clusters, accuracy_score(data_validation.iloc[:, -1], final_votes)))

    return max(clusters_accuracy, key=itemgetter(1))[0]


def calculate_measures(contingency_table):
    measure = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    
    m = contingency_table[0][0].a + contingency_table[0][0].b + contingency_table[0][0].c + contingency_table[0][0].d

    # Q-statistic = (ad - bc) / (ad + bc)
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            a = contingency_table[i][j].a
            b = contingency_table[i][j].b
            c = contingency_table[i][j].c
            d = contingency_table[i][j].d
            measure[i][j] = ((a * d) - (b * c)) / ((a * d) + (b * c))

    return measure


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


def choose_candidates_index(classifiers_pool, clusters, data):
    candidates = []
    cluster_combos = []
    q_statistic_matrix = diversity_measures_classifiers(classifiers_pool, data)

    # Get all possible combinations of classifiers in clusters
    for cluster in clusters:
        if len(cluster) > 1:
            cluster_combos.append(list(combinations(cluster, 2)))
    # Find most diverse options from clusters
    for cluster in cluster_combos:
        best_combination = (0, 0)
        best_value = 1
        for combo in cluster:
            q_value = q_statistic_matrix[combo[0]][combo[1]]
            if q_statistic_matrix[combo[0]][combo[1]] < best_value:
                best_combination = combo
                best_value = q_value
        candidates.append(best_combination[0])
        candidates.append(best_combination[1])
    
    return candidates


def genetic_algorithm(data, candidates):
    q_statistic = diversity_measures_classifiers(candidates, data)
    # Need to define there, because cant pass the rest of arguments to function
    def fitness_function(X):
        ensemble = []
        classifier_indexes = []
        # Majority voting error and averaged pairwise measure (Q-statistic)
        # Add classifiers to ensemble
        for index in range(len(X)):
            if X[index] == 1:
                ensemble.append(candidates[index])
                classifier_indexes.append(index)
        # Find combinations of choosen classifiers
        combos = list(combinations(classifier_indexes, 2))
        # Find average pairwise measure (Q-statistic)
        average_measure = 0
        voting_error = 0
        if len(combos) > 0:
            for combo in combos:
                average_measure += q_statistic[combo[0]][combo[1]]
            average_measure = average_measure / len(combos)
            # Find majority voting error
            voting_error = 1 - majority_voting(ensemble, data, 'accuracy')

        # Wont look after 1 element ensembles cause I will check this separately and getting weird error
        return average_measure + voting_error if len(combos) > 0 else 2

    varbound=np.array([[0, 1]] * len(candidates))
    vartype =np.array([['int']] * len(candidates))
    model = ga(function=fitness_function, dimension=len(candidates), variable_type_mixed=vartype, variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    solution = [i for i in model.output_dict['variable']]
    prunded_ensemble = []
    for index in range(len(solution)):
        if solution[index] == 1:
            prunded_ensemble.append(candidates[index])

    return prunded_ensemble


def consensus_enora(pool_classifiers, clusters_number, samples_train, samples_validation, samples_test, threshold):
    # Two sets of validation sets, so need to separate
    X = samples_validation.iloc[:, 0:-1]
    y = samples_validation.iloc[:, -1]
    Xv1, Xv2, yv1, yv2 = train_test_split(X, y, test_size=0.5, stratify=y)
    # Merge X and y
    samples_validation1 = pd.concat([Xv1, yv1], axis=1)
    samples_validation2 = pd.concat([Xv2, yv2], axis=1)

    clusters = minimum_spanning_tree(pool_classifiers, clusters_number, samples_train, samples_validation1, threshold)
    candidates_indexes = choose_candidates_index(pool_classifiers, clusters, samples_validation1)
    candidates = [pool_classifiers[index] for index in candidates_indexes]
    # Not sure how ENORA works so I used genetic algorithm instead
    new_pool = genetic_algorithm(samples_validation2, candidates)

    print(majority_voting(pool_classifiers, samples_test, 'accuracy'))
    print(majority_voting(candidates, samples_test, 'accuracy'))
    print(majority_voting(new_pool, samples_test, 'accuracy'))
