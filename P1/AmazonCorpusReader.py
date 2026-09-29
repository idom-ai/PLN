import os
import json

from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader
from nltk.corpus.reader.util import concat
from nltk.tokenize import sent_tokenize, word_tokenize, BlanklineTokenizer

class AmazonCorpusReader(CategorizedCorpusReader, CorpusReader):

    """
    Lector de un corpus de reseñas de Amazon organizado en subcarpetas
    por categoría (positivas/neutras/negativas), donde cada fichero
    contiene un JSON (o varios JSON, uno por línea) con 'rating', 'title',
    'text' y 'parent_asin'.
    """

    def __init__(self, root, fileids=r'.*\.json', **kwargs):
        
        # CategorizedCorpusReader necesita saber cómo deducir la
        # categoría de cada fichero. Como tus categorías son las
        # carpetas, usamos cat_pattern con el nombre de carpeta.
        
        """ Deduce de qué categoría es cada fichero """
        
        CategorizedCorpusReader.__init__(
            self, {'cat_pattern': r'(positivas|neutras|negativas)/.*'}
        )
        CorpusReader.__init__(self, root, fileids, **kwargs)

    def _resolve(self, fileids, categories):

        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """

        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Devuelve un generador de dicts {'title', 'text', 'parent_asin'}
        (+ 'category' añadida) para todos los ficheros seleccionados.
        """
        fileids = self._resolve(fileids, categories)
        if fileids is None:
            fileids = self.fileids()
        elif isinstance(fileids, str):
            fileids = [fileids]

        for fileid, category in self._resolve_with_category(fileids):
            with open(os.path.join(self.root, fileid), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj['category'] = category
                    yield obj

    def _resolve_with_category(self, fileids):
        for fileid in fileids:
            cats = self.categories(fileids=[fileid])
            cat = cats[0] if cats else None
            yield fileid, cat

    def titles(self, fileids=None, categories=None):
        return (doc['title'] for doc in self.docs(fileids, categories))

    def ratings(self, fileids=None, categories=None):
        """Devuelve el rating (1-5) de cada review seleccionada."""
        return (doc['rating'] for doc in self.docs(fileids, categories))

    def texts(self, fileids=None, categories=None):
        return (doc['text'] for doc in self.docs(fileids, categories))

    def raw(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        if fileids is None:
            fileids = self.fileids()
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([
            self.open(fileid).read() for fileid in fileids
        ])

    def parags(self, fileids=None, categories=None):
        """
        Para cada review, devuelve sus párrafos como lista de oraciones,
        y cada oración como lista de palabras.
        """
        for text in self.texts(fileids, categories):
            yield [
                [word_tokenize(sent) for sent in sent_tokenize(para)]
                for para in BlanklineTokenizer().tokenize(text)
            ]

    def sents(self, fileids=None, categories=None):
        """Devuelve cada oración de cada review como lista de palabras."""
        for text in self.texts(fileids, categories):
            for sent in sent_tokenize(text):
                yield word_tokenize(sent)

    def words(self, fileids=None, categories=None):
        """Devuelve cada palabra individual de todas las reviews seleccionadas."""
        for sent in self.sents(fileids, categories):
            for word in sent:
                yield word