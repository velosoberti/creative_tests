import numpy as np
import matplotlib.pyplot as plt



class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    ps = counts/total_samples
    gini = 1 - sum(ps**2)
    return gini


def find_best_split(X, y, metric_func):
    num_features = X.shape[1]
    best_feature = None
    best_threshold = None
    start_score = metric_func(y)
    best_info_gain = 0

    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            left_index = X[:, feature_index] <= threshold
            right_index = ~left_index
            if np.any(left_index) and np.any(right_index):
                score_left = metric_func(y[left_index])
                score_right = metric_func(y[right_index])
                score = (np.sum(left_index) / len(y)) * score_left + \
                    (np.sum(right_index)/len(y)) * score_right
                info_gain = start_score - score
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold
    return best_feature, best_threshold


def build_tree_recursive(X, y, depth=1, max_depth=-1, min_samples_split=-1, min_samples_leaf=-1,
                         metric_func=gini):
    if len(np.unique(y)) == 1 or depth == max_depth or len(y) < min_samples_split:
        # pure leaf
        return Node(value=np.bincount(y).argmax())

    best_feature, best_threshold = find_best_split(X, y, metric_func)

    if best_feature is None:

        return Node(value=np.bincount(y).argmax())

    left_index = X[:, best_feature] <= best_threshold
    right_index = ~left_index

    if len(y[left_index]) >= min_samples_leaf and len(y[right_index]) >= min_samples_leaf:
        left_subtree = build_tree_recursive(X[left_index], y[left_index], depth + 1, max_depth, min_samples_split
                                            )
        right_subtree = build_tree_recursive(X[right_index], y[right_index], depth + 1, max_depth, min_samples_split
                                             )
    else:
        return Node(value=np.bicount(y).argmax())

    return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


def predict_single(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict(tree, X):
    y_hat = []
    for x in X:
        prediction = predict_single(tree, x)
        y_hat.append(prediction)
    y_hat = np.array(y_hat)

    return y_hat


def decision_boundary(tree, X, y, X_train, y_train):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1x1, x2x2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    Z = predict(tree, np.c_[x1x1.ravel(), x2x2.ravel()])
    Z = Z.reshape(x1x1.shape)
    plt.contourf(x1x1, x2x2, Z, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=50,
                linewidth=1, cmap=plt.cm.Paired)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
