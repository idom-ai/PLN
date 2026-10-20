import os
import json

import nltk
from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader
from nltk.corpus.reader.util import concat
from nltk.tokenize import sent_tokenize, word_tokenize, BlanklineTokenizer, sent_tokenize
from nltk import pos_tag, wordpunct_tokenize
import time


import re
import html

class AmazonCorpusReader(CategorizedCorpusReader, CorpusReader):

    """
    Lector de un corpus de reseñas de Amazon organizado en subcarpetas
    por categoría (positivas/neutras/negativas), donde cada fichero
    contiene un JSON (o varios JSON, uno por línea) con 'rating', 'title',
    'text' y 'parent_asin'.
    """


    def __init__(self, root, fileids=r'(positivas|neutras|negativas)/.*\.json', **kwargs):
        
        # CategorizedCorpusReader necesita saber cómo deducir la
        # categoría de cada fichero. Como tus categorías son las
        # carpetas, usamos cat_pattern con el nombre de carpeta.
        
        """ Deduce de qué categoría es cada fichero """
        
        CategorizedCorpusReader.__init__(
            self, {'cat_pattern': r'(positivas|neutras|negativas)/.*'}
        )
        nltk.data.path.append(os.path.abspath(root))
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
                    # cambio para la sección 3: ahora limpiamos etiquedas HTML de existir y emojis
                    obj['text'] = self._limpiar(obj['text'])
                    obj['category'] = category
                    yield obj

    def _resolve_with_category(self, fileids):
        for fileid in fileids:
            cats = self.categories(fileids=[fileid])
            cat = cats[0] if cats else None
            yield fileid, cat

    def _limpiar(self, text):
        """Limpia el texto de una review. No cambia el fichero en disco."""
        # 1. etiquetas HTML (<br> pasa a salto de línea, el resto a espacio)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        # 2. entidades HTML (&amp; -> &)
        text = html.unescape(text)
        # 3. emojis y símbolos especiales
        text = re.sub(r'[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]', ' ', text)
        # 4. puntuación repetida: !!! -> !   ??? -> ?   ... -> .
        text = re.sub(r'([!?.])\1+', r'\1', text)
        # 5. letras repetidas 3+ veces: goooood -> good (solo letras, para no tocar "1000")
        text = re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)
        # 6. espacios (sin tocar los saltos de línea de párrafo)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _get_stopwords(self):
        """Lista de stopwords en inglés, cargada una sola vez."""
        if getattr(self, '_stop', None) is None:
            from nltk.corpus import stopwords
            # se conservan las negaciones: "not good" no debe quedar como "good"
            self._stop = set(stopwords.words('english')) - {'no', 'nor', 'not'}
        return self._stop

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

    def paras(self, fileids=None, categories=None, sin_stopwords=False):
        """
        Para cada review, devuelve sus párrafos como lista de oraciones,
        y cada oración como lista de palabras. Descarta lo que esté en blanco.
        Con sin_stopwords=True elimina además las stopwords.
        """
        stop = self._get_stopwords() if sin_stopwords else None
        for text in self.texts(fileids, categories):
            doc = []
            for para in BlanklineTokenizer().tokenize(text):
                if not para.strip():
                    continue
                sents = []
                for sent in sent_tokenize(para):
                    if not sent.strip():
                        continue
                    words = [w for w in word_tokenize(sent) if w.strip()]
                    if stop is not None:
                        words = [w for w in words if w.lower() not in stop]
                    if words:
                        sents.append(words)
                if sents:
                    doc.append(sents)
            yield doc

    def sents(self, fileids=None, categories=None, sin_stopwords=False):
        """Devuelve cada oración de cada review como lista de palabras."""
        for doc_paras in self.paras(fileids, categories, sin_stopwords):
            for para in doc_paras:
                for sent in para:
                    yield sent

    def words(self, fileids=None, categories=None, sin_stopwords=False):
        """Devuelve cada palabra individual de todas las reviews seleccionadas."""
        for sent in self.sents(fileids, categories, sin_stopwords):
            for word in sent:
                yield word

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    # la IA propone este cambio: 

    # def tokenize(self, fileids=None, categories=None):
    #     """Devuelve cada review con sus palabras etiquetadas (PoS)."""
    #     for doc in self.paras(fileids, categories):
    #         yield [[pos_tag(sent) for sent in para] for para in doc]

    def describe(self, fileids=None, categories=None, sin_stopwords=False):
        """Performs a single pass of the corpus and
            returns a dictionary with a variety of metrics
            concerning the state of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts  = nltk.FreqDist()
        tokens  = nltk.FreqDist()

        # tamaño total en caracteres (sobre el texto ya limpio)
        for text in self.texts(fileids, categories):
            counts['chars'] += len(text)

        # Perform single pass over paragraphs, tokenize and count
        for doc in self.paras(fileids, categories, sin_stopwords):
            for para in doc:
                counts['paras'] += 1

                for sent in para:
                    counts['sents'] += 1

                    for word in sent:
                        counts['words'] += 1
                        tokens[word.lower()] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self._resolve(fileids, categories) or self.fileids())
        n_topics  = len(self.categories(self._resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'chars':  counts['chars'], # tamaño total
            'paras':  counts['paras'],
            'sents':  counts['sents'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'ppdoc':  float(counts['paras']) / float(n_fileids),
            'sppar':  float(counts['sents']) / float(counts['paras']),
            'secs':   time.time() - started,
        }