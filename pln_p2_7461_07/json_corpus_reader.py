"""
json_corpus_reader.py - Lector (corpus reader) del corpus de reseñas de Amazon.

Es el AmazonCorpusReader de la Práctica 1, incluido en la plantilla de la Práctica 2
con las correcciones de la P1:
  - E2: los patrones de los fileids (DOC_PATTERN) y de las categorías (CAT_PATTERN)
    se definen como constantes del módulo. DOC_PATTERN reconoce el id de cada review.
  - E3: cada tarea de preprocesamiento tiene su propio flag, que se activa o
    desactiva al crear el reader.
  - M:  todas las funciones documentan sus parámetros de entrada.

Estructura del corpus (la genera pln_p1_7461_07_e1.py):
    <root>/positivas/positivas_<id>.json
    <root>/neutras/neutras_<id>.json
    <root>/negativas/negativas_<id>.json
Cada fichero contiene un JSON con los campos 'rating', 'title', 'text' y 'parent_asin'.
"""
import html
import json
import os
import re
import time

import nltk
from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader
from nltk.corpus.reader.util import concat
from nltk import pos_tag_sents
from nltk.tokenize import BlanklineTokenizer, sent_tokenize, word_tokenize

# ----------------------------------------------------------------------
# Patrones (E2)
# ----------------------------------------------------------------------
# Patrón de los fileids: '<categoria>/<categoria>_<id>.json', p. ej. 'positivas/positivas_17.json'.
#   (negativas|neutras|positivas)  -> carpeta de la categoría (grupo 1)
#   \1                             -> el nombre del fichero empieza por esa misma categoría
#   (\d+)                          -> id numérico de la review (grupo 2)
#   \.json                         -> extensión
# Así solo se aceptan reviews: se descartan ficheros como .DS_Store o nombres mal formados.
DOC_PATTERN = r'(negativas|neutras|positivas)/\1_(\d+)\.json'

# Patrón de las categorías: el primer grupo de captura (el nombre de la carpeta) es la categoría.
CAT_PATTERN = r'(negativas|neutras|positivas)/.*'


class JsonCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Lector de un corpus de reseñas de Amazon organizado en una subcarpeta por
    categoría (positivas/, neutras/, negativas/), con un JSON por review.

    El texto de cada review se preprocesa al leerlo (el fichero en disco no se
    modifica). Cada tarea de preprocesamiento tiene su propio flag (E3).
    """

    def __init__(self, root, fileids=DOC_PATTERN, categories=CAT_PATTERN, encoding='utf-8',
                 limpiar_html=True, limpiar_emojis=True, limpiar_puntuacion=True,
                 limpiar_letras=True, limpiar_espacios=True, quitar_stopwords=False,
                 **kwargs):
        """
        Parámetros:
            root (str): carpeta raíz del corpus (la que contiene positivas/, neutras/ y negativas/).
            fileids (str | list): expresión regular que deben cumplir los ficheros del corpus,
                o lista explícita de fileids. Por defecto, DOC_PATTERN.
            categories (str): expresión regular cuyo primer grupo de captura es la categoría
                de cada fichero. Por defecto, CAT_PATTERN.
            encoding (str): codificación de los ficheros. Por defecto, 'utf-8'.

            Flags de preprocesamiento (True = la tarea se aplica):
            limpiar_html (bool): elimina etiquetas HTML (<br /> pasa a salto de línea)
                y convierte entidades HTML (&amp; -> &). Por defecto, True.
            limpiar_emojis (bool): elimina emojis y símbolos gráficos. Por defecto, True.
            limpiar_puntuacion (bool): reduce la puntuación repetida (!!! -> !). Por defecto, True.
            limpiar_letras (bool): reduce las letras repetidas 3 o más veces
                (goooood -> good). Por defecto, True.
            limpiar_espacios (bool): une espacios y tabuladores repetidos y deja como mucho
                una línea en blanco entre párrafos. Por defecto, True.
            quitar_stopwords (bool): elimina las stopwords del inglés (salvo las negaciones
                'no', 'nor' y 'not') en paras(), sents() y words(). Por defecto, False.

            **kwargs: otros argumentos que se pasan a CorpusReader (p. ej. tagset).
        """
        CategorizedCorpusReader.__init__(self, {'cat_pattern': categories})

        # Las versiones recientes de NLTK solo permiten leer corpus que estén dentro de
        # alguna de las rutas de nltk.data.path, así que se añade la raíz del corpus.
        nltk.data.path.append(os.path.abspath(root))
        CorpusReader.__init__(self, root, fileids, encoding, **kwargs)

        # Un flag por tarea de preprocesamiento (E3)
        self.limpiar_html = limpiar_html
        self.limpiar_emojis = limpiar_emojis
        self.limpiar_puntuacion = limpiar_puntuacion
        self.limpiar_letras = limpiar_letras
        self.limpiar_espacios = limpiar_espacios
        self.quitar_stopwords = quitar_stopwords

        self._stopwords = None  # se cargan la primera vez que se necesitan

    # ------------------------------------------------------------------
    # Selección de ficheros
    # ------------------------------------------------------------------
    def _resolve(self, fileids, categories):
        """
        Traduce la selección del usuario a fileids. Si se pasan categorías, devuelve sus
        ficheros; si no, devuelve los fileids tal cual (implementado de forma parecida a
        CategorizedPlaintextCorpusReader de NLTK).

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros seleccionados.
            categories (str | list | None): categoría o lista de categorías seleccionadas.

        Devuelve:
            list | str | None: los fileids seleccionados (None si no se ha seleccionado nada).

        Lanza:
            ValueError: si se pasan a la vez fileids y categories.
        """
        if fileids is not None and categories is not None:
            raise ValueError('Specify fileids or categories, not both')

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def _lista_fileids(self, fileids, categories):
        """
        Igual que _resolve(), pero devuelve siempre una lista: todos los ficheros si no se
        ha seleccionado nada y una lista de un elemento si se ha pasado un único fileid.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros seleccionados.
            categories (str | list | None): categoría o lista de categorías seleccionadas.

        Devuelve:
            list: lista de fileids.
        """
        fileids = self._resolve(fileids, categories)
        if fileids is None:
            return self.fileids()
        if isinstance(fileids, str):
            return [fileids]
        return list(fileids)

    # ------------------------------------------------------------------
    # Preprocesamiento: una función por tarea (E3)
    # ------------------------------------------------------------------
    def _eliminar_html(self, texto):
        """
        Elimina las etiquetas HTML (<br /> pasa a salto de línea; el resto, a espacio)
        y convierte las entidades HTML en su carácter (&amp; -> &).

        Parámetros:
            texto (str): texto de una review.

        Devuelve:
            str: el texto sin HTML.
        """
        texto = re.sub(r'<br\s*/?>', '\n', texto, flags=re.IGNORECASE)
        texto = re.sub(r'<[^>]+>', ' ', texto)
        return html.unescape(texto)

    def _eliminar_emojis(self, texto):
        """
        Elimina emojis y símbolos gráficos (rangos Unicode de emojis, símbolos varios y
        el selector de variación U+FE0F).

        Parámetros:
            texto (str): texto de una review.

        Devuelve:
            str: el texto sin emojis.
        """
        return re.sub(r'[\U0001F000-\U0001FAFF☀-➿️]', ' ', texto)

    def _reducir_puntuacion(self, texto):
        """
        Reduce los signos de puntuación repetidos a uno solo: !!! -> !, ??? -> ?, ... -> .

        Parámetros:
            texto (str): texto de una review.

        Devuelve:
            str: el texto con la puntuación reducida.
        """
        return re.sub(r'([!?.])\1+', r'\1', texto)

    def _reducir_letras(self, texto):
        """
        Reduce a dos las letras repetidas 3 o más veces: goooood -> good. Solo afecta a
        letras, para no modificar números como 1000.

        Parámetros:
            texto (str): texto de una review.

        Devuelve:
            str: el texto con las letras repetidas reducidas.
        """
        return re.sub(r'([A-Za-z])\1{2,}', r'\1\1', texto)

    def _normalizar_espacios(self, texto):
        """
        Une espacios y tabuladores repetidos y deja como mucho una línea en blanco entre
        párrafos (sin eliminar los saltos de párrafo).

        Parámetros:
            texto (str): texto de una review.

        Devuelve:
            str: el texto con los espacios normalizados.
        """
        texto = re.sub(r'[ \t]+', ' ', texto)
        return re.sub(r'\n\s*\n+', '\n\n', texto)

    def _limpiar(self, texto):
        """
        Aplica al texto, en orden, las tareas de preprocesamiento cuyo flag está activado.
        No modifica el fichero en disco.

        Parámetros:
            texto (str): texto original de una review.

        Devuelve:
            str: el texto preprocesado, sin espacios al principio ni al final.
        """
        if self.limpiar_html:
            texto = self._eliminar_html(texto)
        if self.limpiar_emojis:
            texto = self._eliminar_emojis(texto)
        if self.limpiar_puntuacion:
            texto = self._reducir_puntuacion(texto)
        if self.limpiar_letras:
            texto = self._reducir_letras(texto)
        if self.limpiar_espacios:
            texto = self._normalizar_espacios(texto)
        return texto.strip()

    def _get_stopwords(self):
        """
        Devuelve las stopwords del inglés de NLTK, sin las negaciones 'no', 'nor' y 'not'
        (para que 'not good' no se quede en 'good'). Se cargan una sola vez.

        Parámetros:
            (ninguno)

        Devuelve:
            set: conjunto de stopwords.
        """
        if self._stopwords is None:
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words('english')) - {'no', 'nor', 'not'}
        return self._stopwords

    # ------------------------------------------------------------------
    # Acceso a las reviews
    # ------------------------------------------------------------------
    def docs(self, fileids=None, categories=None):
        """
        Devuelve, una a una, las reviews seleccionadas como diccionarios con sus campos
        ('rating', 'title', 'text', 'parent_asin') más 'category'. El texto ya viene
        preprocesado según los flags del reader.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros a leer. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías a leer.
                No se puede usar a la vez que fileids.

        Devuelve:
            generator de dict: una review por fichero (o por línea, si un fichero tuviera varias).
        """
        for fileid in self._lista_fileids(fileids, categories):
            categoria = self.categories(fileids=[fileid])[0]
            with self.open(fileid) as f:
                contenido = f.read()
            for linea in contenido.splitlines():
                if not linea.strip():
                    continue
                review = json.loads(linea)
                review['text'] = self._limpiar(review['text'])
                review['category'] = categoria
                yield review

    def titles(self, fileids=None, categories=None):
        """
        Devuelve el título de cada review seleccionada.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de str: un título por review.
        """
        return (doc['title'] for doc in self.docs(fileids, categories))

    def ratings(self, fileids=None, categories=None):
        """
        Devuelve la puntuación (de 1 a 5) de cada review seleccionada.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de float: un rating por review.
        """
        return (doc['rating'] for doc in self.docs(fileids, categories))

    def texts(self, fileids=None, categories=None):
        """
        Devuelve el texto preprocesado de cada review seleccionada.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de str: un texto por review.
        """
        return (doc['text'] for doc in self.docs(fileids, categories))

    def raw(self, fileids=None, categories=None):
        """
        Devuelve el contenido original (sin preprocesar) de los ficheros seleccionados,
        concatenado.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            str: contenido en crudo de los ficheros.
        """
        return concat([self.open(fileid).read()
                       for fileid in self._lista_fileids(fileids, categories)])

    # ------------------------------------------------------------------
    # Segmentación: review -> párrafos -> oraciones -> palabras
    # ------------------------------------------------------------------
    def _segmentar(self, texto):
        """
        Divide el texto de una review en párrafos, cada párrafo en oraciones y cada
        oración en palabras. Descarta párrafos, oraciones y palabras en blanco.

        Parámetros:
            texto (str): texto (ya preprocesado) de una review.

        Devuelve:
            list: lista de párrafos -> oraciones -> palabras (str).
        """
        doc = []
        for para in BlanklineTokenizer().tokenize(texto):
            sents = []
            for sent in sent_tokenize(para):
                words = [w for w in word_tokenize(sent) if w.strip()]
                if words:
                    sents.append(words)
            if sents:
                doc.append(sents)
        return doc

    def _filtrar_stopwords(self, doc):
        """
        Elimina las stopwords de una review ya segmentada. Si una oración o un párrafo
        se queda vacío, también se elimina.

        Parámetros:
            doc (list): párrafos -> oraciones -> tokens, donde cada token es una palabra
                (str) o una tupla (palabra, etiqueta PoS).

        Devuelve:
            list: la misma estructura sin las stopwords.
        """
        stop = self._get_stopwords()
        filtrado = []
        for para in doc:
            sents = []
            for sent in para:
                sent = [t for t in sent
                        if (t[0] if isinstance(t, tuple) else t).lower() not in stop]
                if sent:
                    sents.append(sent)
            if sents:
                filtrado.append(sents)
        return filtrado

    def paras(self, fileids=None, categories=None):
        """
        Para cada review, devuelve sus párrafos como listas de oraciones y cada oración
        como lista de palabras. Descarta párrafos, oraciones y palabras en blanco.
        Si quitar_stopwords está activado, elimina además las stopwords.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de list: por cada review, lista de párrafos -> oraciones -> palabras.
        """
        for texto in self.texts(fileids, categories):
            doc = self._segmentar(texto)
            if self.quitar_stopwords:
                doc = self._filtrar_stopwords(doc)
            yield doc

    def sents(self, fileids=None, categories=None):
        """
        Devuelve cada oración de las reviews seleccionadas como lista de palabras.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de list: una lista de palabras por oración.
        """
        for doc in self.paras(fileids, categories):
            for para in doc:
                for sent in para:
                    yield sent

    def words(self, fileids=None, categories=None):
        """
        Devuelve cada palabra de las reviews seleccionadas.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            generator de str: una palabra cada vez.
        """
        for sent in self.sents(fileids, categories):
            for word in sent:
                yield word

    # Práctica 2
    def tokenized(self, fileids=None, categories=None):
        """
        EJERCICIO 1 (Práctica 2): devuelve cada review segmentada y con la etiqueta
        gramatical (PoS, conjunto de etiquetas Penn Treebank) de cada palabra.

        El etiquetado se hace sobre las oraciones completas y, si quitar_stopwords está
        activado, las stopwords se eliminan después: así el etiquetador ve el contexto
        real de cada palabra.

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.
                No se puede usar a la vez que fileids.

        Devuelve:
            generator de list: por cada review, lista de párrafos -> oraciones ->
                tuplas (palabra, etiqueta), p. ej. [[[('Great', 'JJ'), ('gift', 'NN')]]].
        """
        for texto in self.texts(fileids, categories):
            # pos_tag_sents etiqueta todas las oraciones del párrafo en una sola llamada
            # (llamar a pos_tag oración a oración es unas 10 veces más lento)
            doc = [pos_tag_sents(para) for para in self._segmentar(texto)]
            if self.quitar_stopwords:
                doc = self._filtrar_stopwords(doc)
            yield doc

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------
    def describe(self, fileids=None, categories=None):
        """
        Recorre el corpus seleccionado y devuelve un diccionario con sus métricas.
        Las palabras se cuentan después del preprocesamiento (y sin stopwords si
        quitar_stopwords está activado).

        Parámetros:
            fileids (str | list | None): fichero o lista de ficheros. Por defecto, todos.
            categories (str | list | None): categoría o lista de categorías.

        Devuelve:
            dict: con las claves
                files  - número de ficheros
                topics - número de categorías
                chars  - número de caracteres (del texto ya preprocesado)
                paras, sents, words - número de párrafos, oraciones y palabras
                vocab  - tamaño del vocabulario (en minúsculas)
                lexdiv - diversidad léxica (palabras / vocabulario)
                ppdoc  - párrafos por documento
                sppar  - oraciones por párrafo
                secs   - segundos que ha tardado
        """
        started = time.time()
        lista = self._lista_fileids(fileids, categories)

        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        for texto in self.texts(lista):
            counts['chars'] += len(texto)

        for doc in self.paras(lista):
            for para in doc:
                counts['paras'] += 1
                for sent in para:
                    counts['sents'] += 1
                    for word in sent:
                        counts['words'] += 1
                        tokens[word.lower()] += 1

        n_fileids = len(lista)
        n_topics = len(self.categories(lista))

        return {
            'files':  n_fileids,
            'topics': n_topics,
            'chars':  counts['chars'],
            'paras':  counts['paras'],
            'sents':  counts['sents'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': counts['words'] / len(tokens) if tokens else 0.0,
            'ppdoc':  counts['paras'] / n_fileids if n_fileids else 0.0,
            'sppar':  counts['sents'] / counts['paras'] if counts['paras'] else 0.0,
            'secs':   time.time() - started,
        }
