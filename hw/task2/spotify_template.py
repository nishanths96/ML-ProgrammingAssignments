"""
Code to build decision tree classifier for spotify data
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
# import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns

def load_data():
    """ Load Data Set, making sure to import the index column correctly
        Arguments:
            None
        Returns:
            Training data dataframe, training labels, testing data dataframe,
            testing labels, features list
    """
    # TODO: Finish this function
    df = pd.read_csv("spotify_data.csv")
    # df['artist'] = pd.Categorical(df['artist'], categories=df['artist'].unique()).codes
    # drop the first unamed column that denotes the index of the datapoint
    # The title of the song is again not important ; it is another form of an identification ; so we can drop that as well
    # The names of the artist is a text-column and we are dropping that too.
    # Instead of this, we can convert the categorical variable into corresponding numerical representation
    df.drop(['Unnamed: 0', 'song_title', 'artist'], axis=1, inplace=True)

    print("Useful Features:")
    print(list(df.columns))
    # Most important features will be the ones with high high entropy -
        # "loudness": there are 1808 unique "loudness" values => high entropy
        # "Instrumentalness": there are 1107 unique values
        # "speechiness": there are 792 unique values
        # "energy": there are 719 unique values
    # Similarly, vice-versa:
    # Least important features: mode-(2) ; time_signature-(4) ; key-(12) ;
    # because these have very less number of unique values ; in other words lot of values are repeated, thus
    # these features will have lesser entropy and hence, can be considered as least important features from decision tree perspective.

    y = list(df['target'].values)
    df.drop(['target'], axis=1, inplace=True)
    X = list(df.values)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    return (X_train, X_test, y_train, y_test), list(df.columns)

def cv_grid_search(training_table, training_labels):
    """ Run grid search with cross-validation to try different
    hyperparameters
        Arguments:
            Training data dataframe and training labels
        Returns:
            Dictionary of best hyperparameters found by a grid search with
            cross-validation
    """
    # TODO: Finish this function
    params = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 13)),
              'class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 1, 1: 2}, {0: 0, 1: 1}, {0: 1, 1: 0}],
              'min_samples_split': [2, 3, 4, 5]}
              # 'max_leaf_nodes': list(range(2, 100)), }

    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=43), params, verbose=1, cv=5, n_jobs=1)
    grid_search_cv.fit(training_table, training_labels)
    print("Best Parameters are: ")
    print(grid_search_cv.best_estimator_)

    return grid_search_cv.best_params_



def plot_confusion_matrix(test_labels, pred_labels):
    """Plot confusion matrix
        Arguments:
            ground truth labels and predicted labels
        Returns:
            Writes image file of confusion matrix
    """
    # TODO: Finish this function
    cf_matrix = confusion_matrix(test_labels, pred_labels)
    print("Raw Confusion Matrix: ")
    print(cf_matrix)
    
    plt.figure(1)
    plt.title("Confusion Matrix")
    sns.heatmap(cf_matrix, annot=True, fmt='g')
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    # plt.show()
    plt.savefig("./Spotify_Confusion_Matrix.png")

def graph_tree(model, training_features, class_names):
    """ Plot the tree of the trained model
        Arguments:
            Trained model, list of features, class names
        Returns:
            Writes PDF file showing decision tree representation
    """
    # TODO: Finish this function
    export_graphviz(model, out_file='./dtree.dot', feature_names=training_features, class_names = class_names)
    return

def print_results(predictions, test_y):
    """Print results
        Arguments:
            Ground truth labels and predicted labels
        Returns:
            Prints precision, recall, F1-score, and accuracy
    """
    # TODO: Finish this function
    report = classification_report(test_y, predictions, labels=[0, 1], output_dict=True)
    print("Results/Metrics: ")
    print("Accuracy:", report['accuracy'])
    print("Class 0:", report['0'])
    print("Class 1:", report['1'])

    return


def print_feature_importance(model, features):
    """Print feature importance
        Arguments:
            Trained model and list of features
        Returns:
            Prints ordered list of features, starting with most important,
            along with their relative importance (percentage).
    """
    # TODO: Finish this function
    print("Feature Importances:")
    temp = zip(list(features), list(model.feature_importances_))
    temp = sorted(temp, key=lambda t: t[1], reverse=True)
    for fname, fval in temp:
        print("%s - %.3f%%" % (fname, 100*fval))

    return

def main():
    """Run the program"""
    # Load data
    (train_x, test_x, train_y, test_y), features = load_data()

    # Cross Validation Training
    params = cv_grid_search(train_x, train_y)
    # params = ['entropy', 4, 'balanced']

    # Train and test model using hyperparameters
    # TODO: Finish this function
    model = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'],
                           min_samples_split=params['min_samples_split'], class_weight=params['class_weight'])

    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    # Confusion Matrix
    plot_confusion_matrix(test_y, list(predictions))

    # Graph Tree
    graph_tree(model, features, ['hate', 'love'])

    # Accuracy, Precision, Recall, F1
    print_results(predictions, test_y)

    # Feature Importance
    print_feature_importance(model, features)


if __name__ == '__main__':
    main()
