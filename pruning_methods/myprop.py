# 1) Na podstawie zbioru treningowego tworzymy klasyfikator specjalny
# 2) Wybieramy threshold, który określi klasy słabo rozpoznawane przez klasyfikator specjalny.
# 3) Tworzymy tyle grup klasyfikatorów ile jest wyróżnionych klas słabych i do każdego wybieramy klasyfikatory najlepsze w rozpoznawaniu,
# 4) Z każdej grupy wybieramy najlepszy klasyfikator, który rozpoznaje obiekty nierozpoznawalne przez klasyfikator główny
# 5) Klasyfikator rozpoznaje obiekt, jeżeli klasa nie należy do słabych to przyjmujemy jego odpowiedź, w przeciwnym przypadku
#    pytamy odpowiedniego eksperta o radę, jeżeli ich opinia jest zgodna super, inaczej głosowanie większościowe całego zespołu decyzyjnego.
#    A może wsparcia?

from typing import final
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
import math
from combination_methods.majority_voting import majority_voting
import classifiers.base_classifier as bs
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Upgrade in feature
def create_special_classifier(classifier_type, data):
    if classifier_type == 1:
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif classifier_type == 2:
        model = MLPClassifier(random_state=0)
    else:
        model = DecisionTreeClassifier(random_state=0)
    
    model.fit(data.iloc[:, 0:-1], data.iloc[:, -1])

    return model

def find_bad_classes(classifier, threshold, data):
    bad_classes = []
    
    # Create confusion matrix
    special_confusion = confusion_matrix(data.iloc[:, -1], classifier.predict(data.iloc[:, 0:-1]))
    # Find bad classes
    for row_i in range(len(special_confusion)):
        row = special_confusion[row_i]
        row = row / np.sum(row, axis=0)
        if row[row_i] < threshold:
            bad_classes.append(row_i + 1)
    
    return bad_classes

def create_cluster_of_experts(label, number_of_experts, data, pool_of_classifiers):
    # Find samples from searched label
    class_samples = data.loc[data['CLASS'] == label]
    X = class_samples.iloc[:, 0:-1]
    y = class_samples.iloc[:, -1]
    accuracies = {}
    for i in range(len(pool_of_classifiers)):
        accuracies[i] = accuracy_score(y_true=y, y_pred=pool_of_classifiers[i].model.predict(X))
    # Find experts
    best_classifiers = dict(sorted(accuracies.items(), key=lambda item: item[1]))
    top = list(best_classifiers.keys())[-number_of_experts:]
    
    return top

def find_representants(pool_classifiers, clusters, bad_classes, data):
    representants = [[] for _ in range(len(clusters))]

    # Add best classifier to each gropus of representans
    for i in range(len(clusters)):
        representants[i].append(clusters[i].pop())
    # Find best hard examples representants
    for  i in range(len(clusters)):
        # Create pool of real classifiers
        pool = [pool_classifiers[c] for c in clusters[i]]
        class_samples = data.loc[data['CLASS'] == bad_classes[i]]
        margins = margins_calculate(pool, class_samples)
        criterions = criterions_calculate(pool, margins, class_samples)
        expert_index = ranking_classifiers(criterions)
        representants[i].append(clusters[i][expert_index])

    return representants
###############################################################################

def margins_calculate(pool_classifiers, prunning_set):
    data_margins = []
    X = prunning_set.iloc[:, 0:-1]
    labels_data = [[] for i in range(len(prunning_set))]
    
    # Calculate predictions from each classifier
    for classifier in pool_classifiers:
        predictions = classifier.model.predict(X)
        for index in range(len(prunning_set)):
            labels_data[index].append(int(predictions[index]))
    # Calculate margins for each instance
    for labels in labels_data:
        votes = defaultdict(int)
        for label in labels:
            votes[label] += 1
        votes_sorted_reverse = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        vc1 = votes_sorted_reverse[0][1]
        vc2 = votes_sorted_reverse[1][1] if len(votes_sorted_reverse) > 1 else 0
        data_margins.append((vc1 - vc2) / len(pool_classifiers))

    return data_margins


def criterions_calculate(pool_classifiers, margins, prunning_set):
    criterions = []
    X = prunning_set.iloc[:, 0:-1]
    y_true = prunning_set.iloc[:, -1]
    set_length = len(prunning_set)

    for classifier in pool_classifiers:
        H = 0
        predictions = classifier.model.predict(X)
        # Calculate criterion
        for index in range(set_length):
            # Check if label well balanced
            if predictions[index] == y_true.iloc[index]:
                # Margin can't be 0
                H += math.log(margins[index], 10) if margins[index] != 0 else 0
        criterions.append(-1/set_length * H)

    return criterions


def ranking_classifiers(criterions):
    participants = []
    
    # Create (index, criterion) list
    index = 0
    for criterion in criterions:
        participants.append((index, criterion))
        index += 1
    # Sort participants due to criterions
    participants.sort(key = lambda x: x[1], reverse=True)

    return participants[0][0]

def friendship(pool_classifiers, threshold, classifier_type, number_of_experts, data_training, data_validation, data_test):    
    # Create special classifier
    special = create_special_classifier(classifier_type, data_training)
    # special = 
    # Find troublesome classes
    bad_classes = find_bad_classes(special, threshold, data_validation)
    # Create clusters of experts
    clusters_indexes = [[] for _ in range(len(bad_classes))]
    for i in range(len(clusters_indexes)):
        label = bad_classes[i]
        clusters_indexes[i] = create_cluster_of_experts(label, number_of_experts, data_validation, pool_classifiers)
    # Find representants of each group
    experts = find_representants(pool_classifiers, clusters_indexes, bad_classes, data_validation)
    # Proper representation of clusters
    real_experts = [[] for _ in range(10)]
    # cluster of all representants
    final_ensemble = []
    # Add special classifier
    special_base_classifier = bs.BaseClassifier('1',1, 3, 1, data_training, data_validation, 1, special)
    # final_ensemble.append(special_base_classifier)
    for i in range(len(bad_classes)):
        # Find cluster
        class_index = bad_classes[i] - 1
        # Add classifiers to cluster
        for classifier_index in experts[i]:
            real_experts[class_index].append(pool_classifiers[classifier_index])
            final_ensemble.append(pool_classifiers[classifier_index])
    
    # CLASSIFICATION
    # Classify by special
    predictions = special.predict(data_test.iloc[:, 0:-1])
    result = []
    for i in range(len(predictions)):
        ensemble_index = int(predictions[i]) - 1
        # Check if this is good class
        if not real_experts[ensemble_index]:
            result.append(predictions[i])
        else:
            p_1 = real_experts[ensemble_index][0].model.predict(pd.DataFrame(data_test.iloc[i, 0:-1]).T)
            p_2 = real_experts[ensemble_index][1].model.predict(pd.DataFrame(data_test.iloc[i, 0:-1]).T)
            if (predictions[i] != p_1) and (predictions[i] != p_2):
                # result.append(majority_voting(final_ensemble, pd.DataFrame(data_test.iloc[i]).T, 'prediction')[0])
                final_support = [0 for _ in range(10)]
                for classifier in final_ensemble:
                    support = classifier.model.predict_proba(pd.DataFrame(data_test.iloc[i, 0:-1]).T)
                    final_support += support
                result.append(np.argmax(final_support) + 1)
            else:
                result.append(predictions[i])
            # print("---------------------------")
            # print(f'Special: {predictions[i]}')
            # print(f'Expert 1: {p_1}')
            # print(f'Expert 2: {p_2}')
            # print(f'True: {data_test.iloc[i, -1]}')
            # print(majority_voting(final_ensemble, pd.DataFrame(data_test.iloc[i]).T, 'prediction')[0])

    print(f'SpecialA: {accuracy_score(data_test.iloc[:, -1], predictions)}')
    x = f1_score(data_test.iloc[:, -1], predictions, average='weighted')
    print(f'SpecialF: {x}')
    # print(f'Group: {accuracy_score(data_test.iloc[:, -1], result)}')
    print(f'New pool size: {len(final_ensemble)}')
    return result

    