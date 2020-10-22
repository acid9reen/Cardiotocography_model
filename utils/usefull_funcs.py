from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import plot_confusion_matrix


def calc_precision_recall(clf, X, y) -> dict:
    y_pred = cross_val_predict(clf, X, y, cv=3)

    return {
        "precision": precision_score(y, y_pred, average='weighted'),
        "recall": recall_score(y, y_pred, average='weighted')
    }


def confusion_matrix(clf, X_train, y_train, X_test, y_test) -> None:
    clf.fit(X_train, y_train)
    plot_confusion_matrix(clf, X_test, y_test)
