"""
corpus_loader.py - EJERCICIO 2: carga del corpus para validación cruzada.

CorpusLoader guarda los fileids de las reviews (y su categoría) y, al iterar sobre él,
genera las particiones de entrenamiento y test de la validación cruzada.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CorpusLoader(object):
    """
    Prepara el corpus para validación cruzada: cada vuelta del bucle `for` devuelve
    los documentos y etiquetas de entrenamiento y de test de una partición (fold).
    """

    def __init__(self, reader, splitter=StratifiedKFold, folds=3, shuffle=True, categories=None,
                 n_por_categoria=None, semilla=42):
        """
        Parámetros:
            reader (JsonCorpusReader): lector del corpus del que se sacan las reviews.
            splitter (clase de sklearn.model_selection): clase que genera las particiones.
                Por defecto StratifiedKFold, que mantiene en cada fold la misma proporción
                de cada categoría que en el corpus (importante porque está desbalanceado).
            folds (int): número de particiones (N). Por defecto, 3.
            shuffle (bool): si es True, se barajan las reviews antes de partir. Por defecto, True.
            categories (str | list | None): categorías que se usan. Por defecto, todas.
            n_por_categoria (int | None): si se indica, se toman como mucho ese número de
                reviews al azar de cada categoría (submuestreo, para equilibrar las clases y
                reducir el tiempo de cómputo). Por defecto None: todas las reviews.
            semilla (int): semilla aleatoria del submuestreo y del barajado, para que los
                resultados sean reproducibles. Por defecto, 42.
        """
        self.reader = reader

        # Validación cruzada elegida (random_state solo se admite si se baraja)
        self.splitter = splitter(n_splits=folds, shuffle=shuffle,
                                 random_state=semilla if shuffle else None)

        # Identificadores de las reviews
        if n_por_categoria is None:
            fileids = reader.fileids(categories=categories)
        else:
            if categories is None:
                categories = reader.categories()
            elif isinstance(categories, str):
                categories = [categories]
            rng = np.random.RandomState(semilla)
            fileids = []
            for categoria in categories:
                ids = reader.fileids(categories=categoria)
                n = min(n_por_categoria, len(ids))
                fileids.extend(rng.choice(ids, n, replace=False))
            fileids = sorted(fileids)

        self.files = np.asarray(fileids)
        # La categoría de cada review se calcula una sola vez: StratifiedKFold la necesita
        # para partir y labels() la devuelve sin volver a consultar el reader.
        self._labels = np.asarray([reader.categories(fileids=[f])[0] for f in self.files])

    def fileids(self, idx=None):
        """
        Devuelve los identificadores de las reviews.

        Parámetros:
            idx (array de int | None): posiciones de las reviews que se quieren
                (p. ej. los índices de entrenamiento de un fold). Por defecto, todas.

        Devuelve:
            numpy.ndarray de str: los fileids seleccionados.
        """
        if idx is None:
            return self.files
        return self.files[idx]

    def documents(self, idx=None):
        """
        Devuelve el texto de las reviews seleccionadas, ya segmentado y etiquetado
        gramaticalmente con reader.tokenized() (ejercicio 1).

        Parámetros:
            idx (array de int | None): posiciones de las reviews que se quieren. Por defecto, todas.

        Devuelve:
            list: un documento por review, en el mismo orden que fileids(idx); cada documento
                es una lista de párrafos -> oraciones -> tuplas (palabra, etiqueta PoS).
                Es una lista (no un generador) porque GridSearchCV necesita indexarla.
        """
        return list(self.reader.tokenized(fileids=list(self.fileids(idx))))

    def labels(self, idx=None):
        """
        Devuelve la categoría de cada review seleccionada.

        Parámetros:
            idx (array de int | None): posiciones de las reviews que se quieren. Por defecto, todas.

        Devuelve:
            list de str: una categoría ('positivas', 'neutras' o 'negativas') por review,
                en el mismo orden que fileids(idx).
        """
        if idx is None:
            return list(self._labels)
        return list(self._labels[idx])

    def __iter__(self):
        """
        Recorre las particiones de la validación cruzada.

        Parámetros:
            (ninguno: usa el splitter y los fileids guardados en el constructor)

        Devuelve:
            generator de tuplas (X_train, X_test, y_train, y_test), una por fold:
                X_train, X_test: documentos de entrenamiento y de test (ver documents()).
                y_train, y_test: sus categorías (ver labels()).
        """
        for train_idx, test_idx in self.splitter.split(self.files, self._labels):
            X_train = self.documents(train_idx)
            X_test = self.documents(test_idx)
            y_train = self.labels(train_idx)
            y_test = self.labels(test_idx)
            yield X_train, X_test, y_train, y_test
