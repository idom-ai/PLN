"""
text_normalizer.py - EJERCICIO 3: normalización de los textos.

TextNormalizer es un transformador de scikit-learn (se puede meter en un Pipeline y en
GridSearchCV). Recibe las reviews etiquetadas que devuelve CorpusLoader.documents()
y devuelve, por cada review, la lista de sus tokens normalizados:

    [[[('She', 'PRP'), ('loves', 'VBZ'), ('the', 'DT'), ('card', 'NN'), ('!', '.')]]]
        ->  ['she', 'love', 'the', 'card']
"""
import unicodedata
from functools import lru_cache

from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Se crean una sola vez para todo el módulo (no dependen de los parámetros)
_LEMATIZADOR = WordNetLemmatizer()
_STEMMER = SnowballStemmer('english')

# Primera letra de la etiqueta PoS (Penn Treebank) -> categoría de WordNet
_POS_WORDNET = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}


@lru_cache(maxsize=1)
def _stopwords():
    """
    Devuelve las stopwords del inglés de NLTK sin las negaciones 'no', 'nor' y 'not'
    (igual que en el reader). Se cargan una sola vez.

    Parámetros:
        (ninguno)

    Devuelve:
        frozenset: conjunto de stopwords en minúsculas.
    """
    return frozenset(stopwords.words('english')) - {'no', 'nor', 'not'}


class TextNormalizer(TransformerMixin, BaseEstimator):
    """
    Transformador que normaliza las reviews etiquetadas: quita la puntuación, pasa a
    minúsculas y lematiza (o hace stemming de) cada token.
    (El mixin va antes que BaseEstimator, como pide scikit-learn >= 1.6.)
    """

    def __init__(self, normalizacion='lemma', quitar_stopwords=False, incluir_pos=False):
        """
        Los parámetros solo se guardan, con el mismo nombre, para que GridSearchCV pueda
        leerlos y cambiarlos (get_params / set_params). Cada uno es una opción de
        preprocesamiento que se puede activar o desactivar en cada ejecución.

        Parámetros:
            normalizacion (str | None): cómo se reduce cada palabra a su forma base:
                'lemma' (por defecto): lematización con WordNet usando la etiqueta PoS
                    (loves -> love, better -> good).
                'stem': stemming con Snowball (loves -> love, lovely -> love).
                None: solo minúsculas.
            quitar_stopwords (bool): si es True, elimina las stopwords (salvo no/nor/not).
                Por defecto, False.
            incluir_pos (bool): si es True, añade a cada token su etiqueta PoS
                ('love' -> 'love_VBZ'), de modo que la misma palabra con distinta categoría
                gramatical cuenta como una característica distinta. Por defecto, False.
        """
        self.normalizacion = normalizacion
        self.quitar_stopwords = quitar_stopwords
        self.incluir_pos = incluir_pos

    def _is_punct(self, token):
        """
        Comprueba si el token está formado únicamente por signos de puntuación.

        Parámetros:
            token (str): palabra o símbolo.

        Devuelve:
            bool: True si todos sus caracteres son de puntuación según Unicode (categorías
                que empiezan por 'P'), p. ej. '.', '!?', '...', ':)'. False si tiene alguna
                letra, número o símbolo, p. ej. 'good', "n't", '5', '$'.
        """
        return all(unicodedata.category(caracter).startswith('P') for caracter in token)

    def lemmatize(self, token, pos_tag):
        """
        Lematiza el token usando su etiqueta gramatical. Si la etiqueta no es de
        sustantivo, verbo, adjetivo o adverbio, se trata como sustantivo.

        Parámetros:
            token (str): palabra, ya en minúsculas.
            pos_tag (str): etiqueta PoS del token en el conjunto Penn Treebank
                (p. ej. 'VBZ', 'NNS', 'JJR').

        Devuelve:
            str: el lema (p. ej. ('loves', 'VBZ') -> 'love', ('cards', 'NNS') -> 'card').
        """
        categoria = _POS_WORDNET.get(pos_tag[0], wn.NOUN)
        return _LEMATIZADOR.lemmatize(token, categoria)

    def normalize(self, document):
        """
        Normaliza una review etiquetada: omite la puntuación (y las stopwords si
        quitar_stopwords está activado) y devuelve todos sus tokens en minúsculas y
        lematizados (o con stemming, según normalizacion).

        Parámetros:
            document (list): una review como la devuelve JsonCorpusReader.tokenized():
                párrafos -> oraciones -> tuplas (palabra, etiqueta PoS).

        Devuelve:
            list de str: los tokens normalizados de toda la review, en orden.
        """
        tokens = []
        for paragraph in document:
            for sentence in paragraph:
                for token, tag in sentence:
                    if self._is_punct(token):
                        continue
                    token = token.lower()
                    if self.quitar_stopwords and token in _stopwords():
                        continue
                    if self.normalizacion == 'lemma':
                        token = self.lemmatize(token, tag)
                    elif self.normalizacion == 'stem':
                        token = _STEMMER.stem(token)
                    if self.incluir_pos:
                        token = f'{token}_{tag}'
                    tokens.append(token)
        return tokens

    # ------------------------------------------------------------------
    # Métodos de la interfaz de scikit-learn (TransformerMixin aporta fit_transform)
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        No aprende nada: la normalización no depende de los datos de entrenamiento.
        Existe porque el Pipeline de scikit-learn llama a fit() en cada paso.

        Parámetros:
            X (list): reviews etiquetadas (ver normalize()).
            y (list | None): clases de las reviews; no se usan.

        Devuelve:
            TextNormalizer: el propio objeto (self), como exige scikit-learn.
        """
        return self

    def transform(self, X):
        """
        Normaliza todas las reviews.

        Parámetros:
            X (list): reviews etiquetadas, como las devuelve CorpusLoader.documents().

        Devuelve:
            list de list de str: por cada review, su lista de tokens normalizados.
        """
        return [self.normalize(document) for document in X]
