from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import yellowbrick as yb
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

class BaseClassifier:

    def __init__(self, name, classifier_type, number_samples_training, number_samples_validation,
                 samples_training, samples_validation, seed, classifier=None):
        self.__name = name
        self.__type = classifier_type
        if classifier_type == 1:
            self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        elif classifier_type == 2:
            self.model = MLPClassifier(random_state=1, hidden_layer_sizes=(10,10), solver='adam', max_iter=10000)
        else:
            self.model = DecisionTreeClassifier(random_state=0)
        self.seed = seed
        # Actions
        # self.__check_created()
        self.__train_model(number_samples_training, samples_training)
        if classifier:
            self.model = classifier
        # self.__check_model(number_samples_validation, samples_validation)

    def __check_created(self):
        if self.__type == 1:
            print(f"Hello, I am KNeighborsClassifier: {self.__name}")

    def __train_model(self, amount, training_samples):
        # Maybe I should change weights of samples to be different???
        data = training_samples.sample(n=amount, replace=True, ignore_index=True, random_state=self.seed)
        self.model.fit(data.iloc[:, 0:-1], data.iloc[:, -1])
        #yb.target.class_balance(data.CLASS, labels=["A", "B", "C", "D", "SH", "AD", "DE", "LD", "FS", "SUSP"])
        
    def __check_model(self, amount, validation_samples):
        # data = validation_samples.sample(n=amount, replace=True, ignore_index=True)
        data = validation_samples
        X = data.iloc[:, 0:-1]
        y_true = data.iloc[:, -1]
        # print(f"Balanced accuracy score: {balanced_accuracy_score(y_true, self.model.predict(X))}")
        print(f"Accuracy score: {self.model.score(X, y_true)}")
        # print(f"Hope: {self.model.predict_proba(X)}")
