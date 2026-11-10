"""
focal_loss.py - Clasificador lineal entrenado con focal loss.

Es una regresión logística multiclase (softmax) en la que la entropía cruzada se cambia
por la focal loss (Lin et al., 2017):

    FL(p) = - alpha_c * (1 - p)^gamma * log(p)

donde p es la probabilidad que el modelo da a la clase correcta y c es esa clase.
  - (1 - p)^gamma reduce el peso de las reviews que el modelo ya clasifica bien (p alto)
    y concentra el aprendizaje en las difíciles. Con gamma = 0 es la entropía cruzada.
  - alpha_c da más peso a las clases con menos reviews (neutras y negativas).

Se usa como cualquier clasificador de scikit-learn: SentimentClassifier(estimator=FocalLossClassifier()).
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


class FocalLossClassifier(ClassifierMixin, BaseEstimator):
    """
    Clasificador lineal softmax entrenado con focal loss y regularización L2 (como
    LogisticRegression), optimizado con L-BFGS.
    """

    def __init__(self, gamma=2.0, class_weight='balanced', C=1.0, max_iter=300):
        """
        Parámetros:
            gamma (float): cuánto se rebaja el peso de las reviews fáciles. 0 = entropía
                cruzada normal; valores típicos, 1 o 2. Por defecto, 2.
            class_weight (str | None): 'balanced' da a cada clase un peso inversamente
                proporcional a su número de reviews (alpha_c = n / (n_clases * n_c));
                None, el mismo peso a todas. Por defecto, 'balanced'.
            C (float): inverso de la fuerza de la regularización L2, como en
                LogisticRegression (más pequeño = más regularización). Por defecto, 1.
            max_iter (int): iteraciones máximas de L-BFGS. Por defecto, 300.
        """
        self.gamma = gamma
        self.class_weight = class_weight
        self.C = C
        self.max_iter = max_iter

    def _perdida(self, theta, X, Y, y_idx, alpha):
        """
        Calcula la función objetivo (focal loss + regularización) y su gradiente.

        Parámetros:
            theta (numpy.ndarray): pesos W (n_características x n_clases) y sesgos b, en un vector.
            X (matriz dispersa o numpy.ndarray): características de las reviews (n x d).
            Y (numpy.ndarray): clases en one-hot (n x n_clases).
            y_idx (numpy.ndarray): índice de la clase correcta de cada review.
            alpha (numpy.ndarray): peso de la clase correcta de cada review.

        Devuelve:
            tuple: (valor del objetivo, gradiente respecto a theta).
        """
        n, d = X.shape
        k = Y.shape[1]
        W = theta[:d * k].reshape(d, k)
        b = theta[d * k:]

        # softmax estable: se resta el máximo de cada fila antes de exponenciar
        Z = np.asarray(X @ W) + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)

        p = np.clip(P[np.arange(n), y_idx], 1e-12, 1.0)   # probabilidad de la clase correcta
        q = np.clip(1.0 - p, 1e-12, 1.0)
        log_p = np.log(p)
        perdida = -np.sum(alpha * q ** self.gamma * log_p)

        # derivada de la focal loss respecto a los logits: g * (P - Y), con
        # g = alpha * ((1-p)^gamma - gamma * (1-p)^(gamma-1) * p * log p)   (g >= 0)
        g = alpha * q ** self.gamma
        if self.gamma > 0:
            g = g - alpha * self.gamma * q ** (self.gamma - 1) * p * log_p
        G = g[:, None] * (P - Y)

        objetivo = self.C * perdida + 0.5 * np.sum(W ** 2)          # los sesgos no se regularizan
        grad_W = self.C * np.asarray(X.T @ G) + W
        grad_b = self.C * G.sum(axis=0)
        return objetivo, np.concatenate([grad_W.ravel(), grad_b])

    def fit(self, X, y):
        """
        Entrena el clasificador.

        Parámetros:
            X (matriz dispersa o numpy.ndarray): características de las reviews (p. ej. TF-IDF).
            y (list de str): clase de cada review.

        Devuelve:
            FocalLossClassifier: el propio objeto (self), ya entrenado.
        """
        X = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]

        if self.class_weight == 'balanced':
            pesos = n / (k * np.bincount(y_idx, minlength=k))
        else:
            pesos = np.ones(k)
        alpha = pesos[y_idx]

        resultado = minimize(self._perdida, np.zeros(d * k + k), args=(X, Y, y_idx, alpha),
                             jac=True, method='L-BFGS-B', options={'maxiter': self.max_iter})
        self.coef_ = resultado.x[:d * k].reshape(d, k).T     # (n_clases x d), como en sklearn
        self.intercept_ = resultado.x[d * k:]
        self.n_iter_ = resultado.nit
        return self

    def decision_function(self, X):
        """
        Devuelve la puntuación (logit) de cada clase para cada review.

        Parámetros:
            X (matriz dispersa o numpy.ndarray): características de las reviews.

        Devuelve:
            numpy.ndarray: matriz n x n_clases.
        """
        return np.asarray(X @ self.coef_.T) + self.intercept_

    def predict_proba(self, X):
        """
        Devuelve la probabilidad de cada clase para cada review (softmax de los logits).

        Parámetros:
            X (matriz dispersa o numpy.ndarray): características de las reviews.

        Devuelve:
            numpy.ndarray: matriz n x n_clases; cada fila suma 1.
        """
        Z = self.decision_function(X)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predice la clase de cada review (la de mayor probabilidad).

        Parámetros:
            X (matriz dispersa o numpy.ndarray): características de las reviews.

        Devuelve:
            numpy.ndarray: la clase predicha de cada review.
        """
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
