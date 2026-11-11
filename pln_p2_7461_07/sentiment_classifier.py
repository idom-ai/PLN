"""
sentiment_classifier.py - EJERCICIO 4: clasificador de sentimiento con el Pipeline de scikit-learn.

SentimentClassifier encadena los pasos de la clasificación:
    1. Normalizar los textos           (TextNormalizer, ejercicio 3)
    2. Vectorizarlos                    (TfidfVectorizer: representación dispersa)
    3. Reducir la dimensionalidad       (opcional: el reductor que indique el usuario)
    4. Clasificar                       (el modelo que indique el usuario)

Hereda de BaseEstimator y ClassifierMixin para que GridSearchCV pueda probar
distintas configuraciones (clasificador, reducción, n-gramas, normalización...).
El mixin va PRIMERO: con scikit-learn >= 1.6, si BaseEstimator va delante, GridSearchCV
no reconoce la clase como clasificador y usa KFold en lugar de StratifiedKFold.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from text_normalizer import TextNormalizer


def _identidad(tokens):
    """
    Devuelve la entrada sin cambios. TfidfVectorizer la usa como preprocesador y como
    tokenizador, porque los textos ya llegan tokenizados por TextNormalizer.
    (Es una función con nombre y no una lambda para que se pueda serializar con pickle,
    que es lo que usa GridSearchCV al ejecutar en paralelo.)

    Parámetros:
        tokens (list de str): tokens de una review.

    Devuelve:
        list de str: los mismos tokens.
    """
    return tokens


class SentimentClassifier(ClassifierMixin, BaseEstimator):
    """
    Clasificador de polaridad de reviews: normalización -> TF-IDF -> [reducción] -> modelo.
    """

    def __init__(self, estimator=None, reductor=None, ngram_range=(1, 1), min_df=1,
                 class_weight=None, normalizacion='lemma', quitar_stopwords=False, incluir_pos=False):
        """
        Solo se guardan los parámetros, con el mismo nombre, para que GridSearchCV pueda
        leerlos y cambiarlos. El pipeline se construye en fit().

        Parámetros:
            estimator (clasificador de scikit-learn | None): modelo que clasifica, p. ej.
                LogisticRegression(), MultinomialNB() o SGDClassifier(). Por defecto (None),
                LogisticRegression(max_iter=1000).
            reductor (transformador de scikit-learn | None): reducción de dimensionalidad,
                p. ej. TruncatedSVD(n_components=300) o SelectKBest(chi2, k=5000).
                Por defecto None: sin reducción.
            ngram_range (tuple): n-gramas que usa TF-IDF, p. ej. (1, 1) solo palabras,
                (1, 2) palabras y parejas de palabras seguidas. Por defecto (1, 1).
            min_df (int | float): TF-IDF ignora los términos que aparecen en menos reviews
                que este número (o proporción). Por defecto 1: no ignora ninguno.
            class_weight (str | dict | None): peso de cada clase al entrenar, para que
                equivocarse con una clase cueste más:
                None (por defecto): todas pesan lo mismo.
                'balanced': peso inversamente proporcional al número de reviews de la clase.
                dict: pesos a mano, p. ej. {'positivas': 1, 'negativas': 20, 'neutras': 50}.
                Se aplica igual a cualquier clasificador, pasándole a cada review el peso de
                su clase (sample_weight), también a MultinomialNB, que no tiene class_weight.
            normalizacion (str | None): 'lemma', 'stem' o None (ver TextNormalizer).
            quitar_stopwords (bool): elimina las stopwords (ver TextNormalizer). Por defecto False.
            incluir_pos (bool): añade la etiqueta PoS a cada token (ver TextNormalizer).
                Por defecto False.
        """
        self.estimator = estimator
        self.reductor = reductor
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.class_weight = class_weight
        self.normalizacion = normalizacion
        self.quitar_stopwords = quitar_stopwords
        self.incluir_pos = incluir_pos

    def _crear_pipeline(self):
        """
        Construye el Pipeline de scikit-learn con la configuración actual.

        Parámetros:
            (ninguno: usa los parámetros guardados en el constructor)

        Devuelve:
            sklearn.pipeline.Pipeline: los pasos normalizer -> vectorizer -> [reductor] -> classifier.
        """
        pasos = [
            ('normalizer', TextNormalizer(normalizacion=self.normalizacion,
                                          quitar_stopwords=self.quitar_stopwords,
                                          incluir_pos=self.incluir_pos)),
            # Los textos ya vienen como listas de tokens: TF-IDF no debe volver a
            # tokenizar ni pasar a minúsculas, solo formar los n-gramas y contarlos.
            ('vectorizer', TfidfVectorizer(preprocessor=_identidad, tokenizer=_identidad,
                                           token_pattern=None, lowercase=False,
                                           ngram_range=self.ngram_range, min_df=self.min_df)),
        ]
        if self.reductor is not None:
            # clone(): copia sin entrenar, para no modificar el objeto que pasó el usuario
            pasos.append(('reductor', clone(self.reductor)))

        if self.estimator is None:
            clasificador = LogisticRegression(max_iter=1000)
        else:
            clasificador = clone(self.estimator)
        pasos.append(('classifier', clasificador))

        return Pipeline(pasos)

    def fit(self, X, y):
        """
        Construye el pipeline y lo entrena. Si hay class_weight, cada review se entrena con el
        peso de su clase.

        Parámetros:
            X (list): reviews etiquetadas, como las devuelve CorpusLoader.documents().
            y (list de str): clase de cada review ('positivas', 'neutras', 'negativas').

        Devuelve:
            SentimentClassifier: el propio objeto (self), ya entrenado.
        """
        self.pipeline_ = self._crear_pipeline()
        if self.class_weight is None:
            self.pipeline_.fit(X, y)
        else:
            # peso de cada review = peso de su clase; el Pipeline se lo pasa solo al clasificador
            pesos = compute_sample_weight(self.class_weight, y)
            self.pipeline_.fit(X, y, classifier__sample_weight=pesos)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, X):
        """
        Predice la clase de cada review.

        Parámetros:
            X (list): reviews etiquetadas, como las devuelve CorpusLoader.documents().

        Devuelve:
            numpy.ndarray de str: la clase predicha para cada review.
        """
        return self.pipeline_.predict(X)
