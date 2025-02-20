import matplotlib.pyplot as plt
from construtores import build_tree_recursive
from construtores import predict
from construtores import decision_boundary
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=2,
                           n_classes=2, n_informative=2, n_redundant=0, random_state=24)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
            marker='o', s=50, linewidth=1, cmap=plt.cm.Paired)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
tree = build_tree_recursive(X_train, y_train, max_depth=3, min_samples_leaf=10)


y_pred = predict(tree, X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Set Accuracy: {accuracy:.2f}')


decision_boundary(tree, X, y, X_train, y_train)
