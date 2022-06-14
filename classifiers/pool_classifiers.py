import classifiers.base_classifier as bs

# Only to create pool of homogeneus classifiers, should refactor later
class PoolClassifiers:

    def __init__(self, name, classifier_number, classifier_type, samples_training, samples_validation, number_samples_training):
        self.name = name
        self.type = classifier_type
        self.pool = [bs.BaseClassifier(
            name=i,
            classifier_type=classifier_type,
            number_samples_training=number_samples_training,
            number_samples_validation=100,
            samples_training=samples_training,
            samples_validation=samples_validation,
            seed=i) for i in range(classifier_number)]
        # Will do combination rules in there
