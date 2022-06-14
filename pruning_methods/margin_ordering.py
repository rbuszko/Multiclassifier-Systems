from collections import defaultdict
from combination_methods.majority_voting import majority_voting
import math
import numpy as np

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


def ranking_classifiers(pool_classifiers, criterions, data):
    accuracies = []
    new_pool = []
    participants = []
    
    # Create (index, criterion) list
    index = 0
    for criterion in criterions:
        participants.append((index, criterion))
        index += 1
    # Sort participants due to criterions
    participants.sort(key = lambda x: x[1], reverse=True)
    # Create new pool of classifiers
    for index in range(len(pool_classifiers)):
        new_pool.append(pool_classifiers[participants[index][0]])
        accuracies.append(majority_voting(new_pool, data, 'accuracy'))

    return [pool_classifiers[participants[index][0]] for index in range(np.argmax(accuracies) + 1)]


def margin_ordering(pool_classifiers, data_validation, data_test):
    margins = margins_calculate(pool_classifiers, data_validation)
    criterions = criterions_calculate(pool_classifiers, margins, data_validation)
    new_pool = ranking_classifiers(pool_classifiers, criterions, data_validation)

    # To compare, probably will change in future
    # print(majority_voting(pool_classifiers, data_test, 'accuracy'))
    # print(majority_voting(new_pool, data_test, 'accuracy'))
    # return new_pool
    print(f'New pool size: {len(new_pool)}')
    return majority_voting(new_pool, data_test, 'probes')
