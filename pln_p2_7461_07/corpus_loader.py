from sklearn.model_selection import StratifiedKFold

"""
EJERCICIO 2
"""
class CorpusLoader(object):

    def __init__(self, reader, splitter=StratifiedKFold, folds=3, shuffle=True, categories=None):
        pass

    def fileids(self, idx=None):
        pass

    def documents(self, idx=None):
        pass

    def labels(self, idx=None):
        pass

    def __iter__(self):
        pass
