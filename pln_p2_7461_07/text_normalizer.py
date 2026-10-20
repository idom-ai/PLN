from sklearn.base import BaseEstimator, TransformerMixin

"""
EJERCICIO 3
"""
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def is_punct(self, token):
        pass

    def normalize(self, document):
        pass

    def lemmatize(self, token, pos_tag):
        pass

    """
    INCLUIR MÉTODOS HEREDADOS
    """
