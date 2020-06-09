from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
def build_ml_models():
    models = [
        # hinge loss Linear SVM
        ("SGD_hinge",
         SGDClassifier(loss='hinge', alpha=0.001, class_weight=None, penalty='l2', l1_ratio=0.15, max_iter=5, tol=None,
                       shuffle=True, verbose=0, epsilon=0.1, n_jobs=5, random_state=None, learning_rate='optimal',
                       eta0=0.0, power_t=0.5, average=False)),
        # logistic regression
        ("SGD_log",
         SGDClassifier(loss='log', alpha=0.0001, class_weight=None, penalty='l2', l1_ratio=0.15, max_iter=20, tol=None,
                       shuffle=True, verbose=0, epsilon=0.1, n_jobs=5, random_state=None, learning_rate='optimal',
                       eta0=0.0, power_t=0.5, average=False)),
        # perceptron
        ("SGD_perceptron",
         SGDClassifier(loss='perceptron', alpha=0.1, class_weight=None, fit_intercept=False, penalty='l2',
                       warm_start=True, l1_ratio=0.15, max_iter=5, tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                       n_jobs=5, random_state=42, learning_rate='optimal', eta0=0.0, power_t=0.5, average=False)),
        # RF
        ("Random_forest_300", RandomForestClassifier(n_estimators=300, n_jobs=5))
    ]
    return models