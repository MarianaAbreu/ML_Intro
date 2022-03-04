
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def correlation_(data, dec_data, labels, cs):
    """pearson correlation between autoencoder's input and output

    Args:
        data (array): input points
        dec_data (array): output points
        labels (array): labels
        cs (int): segment size 

    Returns:
        list, list: correlation points, labels
    """
    # compute pearson correlation

    new_data, new_labels = [], []
    for d in range(len(dec_data)):
        corr_train = [pearsonr(dec_data[d][i: i + cs], data[d][i: i + cs])[0] for i in
                      range(0, len(dec_data[d]), cs)]
        if np.isfinite(corr_train).all():
            new_data += [corr_train]
            new_labels += [labels[d]]

    return new_data, new_labels


def score_classifier_pred(c, n, train, test, y_train, y_test):
    """Compute classification score

    Args:
        c (classifier): classifier to use
        n (str): name of classifier
        train (array): train values
        test (array): test_values
        y_train (list or array): label values
        y_test (list or array): label test values

    Returns:
        float, list: accuracy score, list of predictions
    """

    # Train the classifier
    c.fit(train, y_train.ravel())

    # Predict test data
    y_pred = c.predict(test)

    score = accuracy_score(y_test, y_pred)

    # Get the classification accuracy

    print(f" --- Classifier {n} Final Score: " + str(np.round(score * 100, 2)) + '%')
    print('-----------------------------------------')

    return np.round(score * 100, 2), y_pred

names = ["KNN", "SVM", "Decision Tree", "Random Forest", "MLP", \
         "Adaboost", "GradBoost", "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

if __name__ == '__main__':

    for cl, classif in enumerate(classifiers):
        _score, pred_label = score_classifier_pred(classif, names[cl], train, test, np.array(y_train), np.array(y_test))

