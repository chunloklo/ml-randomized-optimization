import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
import pickle

from sklearn import metrics, model_selection, base

def test_accuracy(clf, X_test, y_test):
    y_ = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_)
    return accuracy

def accuracy_vs_num_train_samples(clf, X_train, y_train, scoring=None, X_val=None, y_val=None, min_train_num=1, num_step=10, cross_val=True, folds=10):
    #TODO number of increment is not implemented
    train_acc = []
    test_acc = []
    train_samples = []
    i = 0
    print(min_train_num)
    for num_train in np.linspace(min_train_num, X_train.shape[0], num=num_step):
        i += 1
        num_train = int(num_train)
        print("iteration {}: training size: {}".format(i, num_train), end='\r')

        cX_train = X_train[0:num_train]
        cy_train = y_train[0:num_train]

        clf = base.clone(clf)

        if cross_val:
            cv_result = model_selection.cross_validate(clf, cX_train, cy_train, cv=folds, return_train_score=True, scoring=scoring)
            #print("Time taken to cross validate: {}".format(cv_result['fit_time']))
            train_acc.append(np.mean(cv_result['train_score']))
            test_acc.append(np.mean(cv_result['test_score']))
        else:
            clf.fit(cX_train, cy_train)
            train_acc.append(test_accuracy(clf, cX_train, cy_train))
            test_acc.append(test_accuracy(clf, X_val, y_val))


        train_samples.append(num_train)
    return train_acc, test_acc, train_samples


def show_confusion_matrix(clf, X_test, y_test, classes):
    y_ = clf.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_)
    print(cnf_matrix)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix')
    accuracy = metrics.accuracy_score(y_test, y_)
    return cnf_matrix, accuracy

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def train_val_test_split(X, y, train_size=0.6, val_size=0.2, random_state=None):
    train_val_size = train_size + val_size
    X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(X, y, train_size=train_val_size, random_state=random_state)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, train_size=train_size/train_val_size, random_state=random_state)
    return X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, y_train_val

def saveNpArray(arr, filename, descriptors=None):
    np.save("npData/{}".format("-".join([datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), filename, descriptors])), arr)

def saveObj(obj, filename, descriptors=None):
    with open("objData/{}.pkl".format("-".join([datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), filename, descriptors])), "wb") as f:
        pickle.dump(obj, f)

def plot_accuracy(train_acc, test_acc, x_vals, detail=False):
    plt.plot(x_vals, test_acc, label='test accuracy', marker='.')
    plt.plot(x_vals, train_acc, label='train accuracy', marker='.')
    if detail:
        plt.xticks(x_vals)
    plt.legend()
    #plt.ylim(0, 1)
    plt.show()

def paramTest(classifier, parameter, X_train, y_train, X_val=None, y_val=None, cross_val=True, folds=10, scoring=None):
    train_acc = []
    test_acc = []
    for param_val in parameter[1]:
        cur_param = { parameter[0] : param_val}
        print("Processing Param Val: {}".format(param_val), end='\r')
        clf = base.clone(classifier)
        clf.set_params(**cur_param)

        if cross_val:
            cv_result = model_selection.cross_validate(clf, X_train, y_train, cv=folds, return_train_score=True, scoring=scoring)
            train_acc.append(np.mean(cv_result['train_score']))
            test_acc.append(np.mean(cv_result['test_score']))
        else:
            if (scoring == None):
                print("SCORING NOT IMPLEMENTED FOR NOT CROSSVAL")
            train_acc.append(test_accuracy(clf, cX_train, cy_train))
            test_acc.append(test_accuracy(clf, X_val, y_val))
    print()
    return train_acc, test_acc, parameter[1]