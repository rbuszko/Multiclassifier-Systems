from dataclasses import dataclass
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import train_test_split
import numpy as np
import math
import pandas as pd
from combination_methods.majority_voting import majority_voting

# Genetic algorithm doc:                                  https://pypi.org/project/geneticalgorithm/
# Graph coloring algorithm:                               https://www.geeksforgeeks.org/graph-coloring-set-2-greedy-algorithm/
def greedyColoring(adj, V):
    result = [-1] * V
    result[0] = 0
    available = [False] * V
    for u in range(1, V):
        for i in adj[u]:
            if (result[i] != -1):
                available[result[i]] = True
        cr = 0
        while cr < V:
            if (available[cr] == False):
                break
            cr += 1
        result[u] = cr
        for i in adj[u]:
            if (result[i] != -1):
                available[result[i]] = False
    
    return result


algorithm_param = {'max_num_iteration': 600,\
                   'population_size':22,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


def add_node(graph, visited, node, nodes_connections):
    # Check node as visited
    visited[node] = True
    # Add node to graph
    graph.append(node)
    # Loop trought connections in node to add not visited nodes to graph
    for item in nodes_connections[node]:
        if visited[item] == False:
            add_node(graph, visited, item, nodes_connections)


def graphs_separate(nodes_connections):
    # To know if we checked node before
    visited = [False for i in range(len(nodes_connections))]
    # List of connected graphs
    result = []
    # Loop trough every loop and find connections
    for node in range(len(nodes_connections)):
        if (visited[node] == False) & (len(nodes_connections[node]) != 0):
            graph = []
            add_node(graph, visited, node, nodes_connections)
            graph.sort()
            result.append(graph)
    
    return result


def graphs_combinations(groups_colors):
    result = []

    # If there are more graphs than 1 we combine the possibilities
    if len(groups_colors) > 1:
        for i in range(len(groups_colors) - 1):
            if result:
                result.append([x + y for x in result[i - 1] for y in groups_colors[i + 1]])
            else:
                result.append([x + y for x in groups_colors[i] for y in groups_colors[i + 1]])

        return result[-1]
    # If there is only 1 we got the answer
    else:
        return groups_colors[-1]


def genetic_algorithm(M1, M2, M3, M4, M5, data_validation, classifiers_pool):
    def best_pool(w1, w2, w3, w4, w5, T):
        # Create combined diversity matrix
        H = w1 * M1 + w2 * M2 + w3 * M3 + w4 * M4 + w5 * M5
        # Create adjacency matrix
        A = np.ndarray((H.shape[0], H.shape[1]), dtype=int)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                A[i][j] = 1 if H[i][j] < T else 0
        # If A is array of zeros go next
        if not np.any(A):
            return [], 0
        # Get nodes number
        nodes_number = H.shape[0]
        # Get vertex connections
        nodes_connections = [[] for i in range(nodes_number)]
        for i in range(nodes_number):
            for j in range(nodes_number):
                if A[i][j] == 1:
                    nodes_connections[i].append(j)
        # Get separated graphs
        graphs = graphs_separate(nodes_connections)
        # Separate groups by color and graph
        colors = greedyColoring(nodes_connections, len(nodes_connections))
        max_color = np.max(colors)
        groups_colors = [[[] for i in range(max_color + 1)] for j in range(len(graphs))]
        for index in range(len(graphs)):
            for node in graphs[index]:
                groups_colors[index][colors[node]].append(node)
        # Delete empty groups if any
        for i in range(len(groups_colors)):
            groups_colors[i] = list(filter(None, groups_colors[i]))
        # Get all possible combinations of classifiers
        classifier_combinations = graphs_combinations(groups_colors)
        # Get classifier pools
        pools = [[] for i in range(len(classifier_combinations))]
        for i in range(len(classifier_combinations)):
            for classifier_number in classifier_combinations[i]:
                pools[i].append(classifiers_pool[classifier_number])
        # Get pools result
        pools_result = []
        for pool in pools:
            pools_result.append(majority_voting(pool, data_validation, 'accuracy'))

        return pools[np.argmax(pools_result)], np.max(pools_result)

    # Need to define there, because cant pass the rest of arguments to function
    def fitness_function(X):
        w1 = X[0]
        w2 = X[1]
        w3 = X[2]
        w4 = X[3]
        w5 = X[4]
        T = X[5]

        best_pruned_pool, best_result = best_pool(w1, w2, w3, w4, w5, T)

        return 1 - best_result

    # Definition of weight and treshold bounds
    varbound=np.array([[0, 10],[0, 10],[0, 10],[0, 10],[0, 10],[0, 10]])
    vartype=np.array([['real'],['real'],['real'],['real'],['real'],['real']])

    model = ga(function=fitness_function, dimension=6, variable_type_mixed=vartype, variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()

    # convergence = model.report
    solution = model.output_dict
    best_pruned_pool, best_result = best_pool(solution['variable'][0], solution['variable'][1], solution['variable'][2], solution['variable'][3], solution['variable'][4], solution['variable'][5])
    
    print(f'Result the same i hope {best_result}')
    return best_pruned_pool


@dataclass
class contingency_structure:
    a: int # c1 true,  c2 true
    b: int # c1 true,  c2 false
    c: int # c1 false, c2 true
    d: int # c1 false, c2 false


def diversity_measures(classifiers_pool, data_validation):
    contingency_table = np.ndarray((len(classifiers_pool), len(classifiers_pool)), dtype=contingency_structure)
    predictions = []
    X = data_validation.iloc[:, 0:-1]
    y_true = data_validation.iloc[:, -1]

    # Calculate predictions for each classifier
    for classifier in classifiers_pool:
        predictions.append(classifier.model.predict(X))
    # Create contingency table
    for ip1 in range(len(predictions)):
        for ip2 in range(len(predictions)):
            object = contingency_structure(0, 0, 0, 0)
            for index in range(len(data_validation)):
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
    M1 = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    M2 = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    M3 = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    M4 = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    M5 = np.ndarray((contingency_table.shape[0], contingency_table.shape[1]), dtype=float)
    
    m = contingency_table[0][0].a + contingency_table[0][0].b + contingency_table[0][0].c + contingency_table[0][0].d

    # Disagreement            = (b + c) / m
    # Q-statistic             = (ad - bc) / (ad + bc)
    # Correlation Coefficient = (ad - bc) / sqrt((a + b)(a + c)(c + d)(b + d))
    # Kappa-statistic         = (((a + d) / m) - ((a + b)(a + c) + (c + d) (b + d) / m ^ 2)) / m
    # Double-fault            = d / m
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            a = contingency_table[i][j].a
            b = contingency_table[i][j].b
            c = contingency_table[i][j].c
            d = contingency_table[i][j].d
            M1[i][j] = (b + c) / m
            M2[i][j] = ((a * d) - (b * c)) / ((a * d) + (b * c))
            M3[i][j] = ((a * d) - (b * c)) / math.sqrt((a + b) * (a + c) * (c + d) * (b + d))
            o1 = (a + d) / m
            o2 = (((a + b) * (a + c)) + ((c + d) * (b + d))) / math.pow(m, 2)
            M4[i][j] = (o1 - o2) / (1 - o2)
            M5[i][j] = d / m

    return M1, M2, M3, M4, M5


def divp(classifiers_pool, data_validation, data_test):
    # Two sets of validation sets, so need to separate
    X = data_validation.iloc[:, 0:-1]
    y = data_validation.iloc[:, -1]
    Xv1, Xv2, yv1, yv2 = train_test_split(X, y, test_size=0.5, stratify=y)
    # Merge X and y
    samples_validation1 = pd.concat([Xv1, yv1], axis=1)
    samples_validation2 = pd.concat([Xv2, yv2], axis=1)

    # Get diversity measures
    M1, M2, M3, M4, M5 = diversity_measures(classifiers_pool, samples_validation1)
    # I optimize by accuracy as in document, if want to change this change line 133
    new_pool = genetic_algorithm(M1, M2, M3, M4, M5, samples_validation2, classifiers_pool)

    # To compare, probably will change in future
    print(majority_voting(classifiers_pool, data_test, 'accuracy'))
    print(majority_voting(new_pool, data_test, 'accuracy'))
