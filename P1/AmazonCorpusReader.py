import os
import json

from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader
from nltk.corpus.reader.util import concat

class AmazonCorpusReader(CorpusReader, CategorizedCorpusReader):
    
    """
    Lector de un corpus de reseñas de Amazon organizado en subcarpetas
    por categoría (positivas/neutras/negativas), donde cada fichero
    contiene un JSON (o varios JSON, uno por línea) con 'title', 'text'
    y 'parent_asin'.
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
        
        """Traduce (fileids, categories) al conjunto final de fileids.
        
        La lógica es:
        
        no se puede pedir un fichero y una categoría a la vez
        si te piden una categoría concreta, devuelve los ficheros de esa categoría
        si no te piden un fichero concreto, devuelve todos """
        
        if fileids is not None and categories is not None:
            raise ValueError('No puedes especificar fileids y categories a la vez')
        if categories is not None:
            return self.fileids(categories)
        if fileids is None:
            return self.fileids()
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Devuelve un generador de dicts {'title', 'text', 'parent_asin'}
        (+ 'category' añadida) para todos los ficheros seleccionados.
        """
        fileids = self._resolve(fileids, categories)

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

    def texts(self, fileids=None, categories=None):
        return (doc['text'] for doc in self.docs(fileids, categories))

    def raw(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        return concat([
            self.open(fileid).read() for fileid in fileids
        ])