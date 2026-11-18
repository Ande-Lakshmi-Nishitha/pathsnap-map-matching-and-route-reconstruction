from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier


def registry(seed=42):
    return {
        "logreg": dict(
            family="Linear / probabilistic",
            config="StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=2000)",
            make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
        "gaussian_nb": dict(
            family="Linear / probabilistic",
            config="GaussianNB(var_smoothing=1e-9)",
            make=lambda: GaussianNB()),
        "knn": dict(
            family="Instance-based",
            config="StandardScaler + KNN(k=25, distance-weighted), fit on <=120k rows",
            make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)),
            max_rows=120000),
        "decision_tree": dict(
            family="Tree-based",
            config="DecisionTree(max_depth=12, min_samples_leaf=20, gini)",
            make=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=seed)),
        "random_forest": dict(
            family="Ensemble (bagging)",
            config="RandomForest(200 trees, max_depth=18, min_samples_leaf=5, sqrt features)",
            make=lambda: RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=2, random_state=seed)),
        "extra_trees": dict(
            family="Ensemble (randomised trees)",
            config="ExtraTrees(200 trees, max_depth=18, min_samples_leaf=5)",
            make=lambda: ExtraTreesClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=2, random_state=seed)),
        "lightgbm": dict(
            family="Boosting",
            config="LightGBM(400 trees, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8)",
            make=lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=seed, n_jobs=2, verbose=-1)),
        "mlp": dict(
            family="Neural network",
            config="StandardScaler + MLP(64,32), ReLU, Adam lr=1e-3, early stopping",
            make=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3,
                                                                        early_stopping=True, max_iter=80, random_state=seed))),
    }
