from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader

class JsonCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=DOC_PATTERN, categories=CAT_PATTERN, encoding='utf-8', **kwargs):
        pass

    """
    INCLUYE AQUÍ, O PEGA EN ESTE LUGAR EL CORPUS READER QUE 
    SE HA IMPLEMENTADO EN LA PRÁCTICA 1.
    """

    # Práctica 2
    def tokenized(self, fileids=None, categories=None):
        """
        EJERCICIO 1
        """
        pass