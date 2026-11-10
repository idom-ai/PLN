"""
pln_p1_7461_07_e1.py - Ejercicio 1 de la Práctica 1: descarga y preparación del corpus.

Pasos:
  1. Descarga el fichero de reviews de una categoría del dataset Amazon Reviews 2023
     (por defecto, Gift_Cards) y lo descomprime. Si ya está descargado, no lo repite.
  2. Recorre el fichero y descarta las reviews duplicadas (mismo usuario, título y texto).
  3. Guarda cada review en un JSON propio, dentro de la carpeta de su categoría:
         <salida>/positivas/positivas_<id>.json
         <salida>/neutras/neutras_<id>.json
         <salida>/negativas/negativas_<id>.json
     donde <id> es un número correlativo (1, 2, 3...) dentro de cada categoría.
     Clases: positivas = 4 y 5 estrellas, neutras = 2 y 3, negativas = 1.

Uso:
  python3 pln_p1_7461_07_e1.py                          # Gift_Cards -> carpeta corpus/
  python3 pln_p1_7461_07_e1.py --salida corpus --categoria Gift_Cards
  python3 pln_p1_7461_07_e1.py --forzar                 # vuelve a descargar aunque exista
"""
import argparse
import gzip
import json
import os
import shutil
import urllib.request
from typing import List

# Fuentes de descarga, en orden de preferencia. {categoria} se sustituye por el
# nombre de la categoría (p. ej. 'Gift_Cards'). Si la primera falla, se prueba la segunda.
URLS_DESCARGA = [
    # Web oficial del dataset (fichero comprimido con gzip)
    'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{categoria}.jsonl.gz',
    # Copia en Hugging Face (fichero sin comprimir)
    'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{categoria}.jsonl',
]

# Campos de cada review que se guardan en el corpus
CAMPOS_POR_DEFECTO = ['rating', 'title', 'text', 'parent_asin']


def descargar_reviews(categoria: str = 'Gift_Cards', carpeta: str = '.',
                      urls: List[str] = URLS_DESCARGA, forzar: bool = False) -> str:
    """
    Descarga el fichero .jsonl con las reviews de una categoría de Amazon Reviews 2023.

    Parámetros:
        categoria (str): nombre de la categoría de productos tal y como aparece en el
            dataset (p. ej. 'Gift_Cards', 'All_Beauty'). Por defecto, 'Gift_Cards'.
        carpeta (str): carpeta donde se guarda el fichero descargado. Por defecto, la actual.
        urls (List[str]): plantillas de URL a probar en orden; '{categoria}' se sustituye por
            el parámetro categoria. Si la URL acaba en '.gz', se descomprime al descargar.
        forzar (bool): si es True, descarga el fichero aunque ya exista. Por defecto, False.

    Devuelve:
        str: ruta del fichero .jsonl descargado (y ya descomprimido).

    Lanza:
        RuntimeError: si no se ha podido descargar desde ninguna de las URLs.
    """
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f'{categoria}.jsonl')

    if os.path.exists(ruta) and not forzar:
        print(f'{ruta} ya existe: no se vuelve a descargar (usa --forzar para repetirlo).')
        return ruta

    # Se descarga primero a un fichero temporal: si la descarga se corta a medias,
    # no queda un .jsonl incompleto que la siguiente ejecución daría por bueno.
    temporal = ruta + '.part'
    for plantilla in urls:
        url = plantilla.format(categoria=categoria)
        print(f'Descargando {url} ...')
        try:
            peticion = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                # Si viene comprimido, se descomprime mientras se descarga
                origen = gzip.GzipFile(fileobj=respuesta) if url.endswith('.gz') else respuesta
                with open(temporal, 'wb') as destino:
                    shutil.copyfileobj(origen, destino)
            os.replace(temporal, ruta)
            print(f'Guardado en {ruta}')
            return ruta
        except OSError as error:  # incluye errores de red, HTTP y de descompresión
            print(f'  No se ha podido descargar ({error}). Se prueba la siguiente fuente.')
            if os.path.exists(temporal):
                os.remove(temporal)

    raise RuntimeError(f'No se ha podido descargar {categoria} desde ninguna fuente.')


class ProcesadorDeTexto:
    """
    Construye el corpus a partir del fichero .jsonl de Amazon: elimina las reviews
    duplicadas y guarda cada una en un JSON propio dentro de la carpeta de su categoría.
    """

    def __init__(self, fichero: str, campos: List[str] = CAMPOS_POR_DEFECTO,
                 carpeta_salida: str = 'corpus',
                 umbral_positiva: float = 3.5, umbral_negativa: float = 1.5):
        """
        Parámetros:
            fichero (str): ruta del fichero .jsonl descargado (una review por línea).
            campos (List[str]): campos de cada review que se guardan en el corpus.
                Por defecto ['rating', 'title', 'text', 'parent_asin'].
            carpeta_salida (str): carpeta raíz del corpus; dentro se crean positivas/,
                neutras/ y negativas/. Por defecto, 'corpus'.
            umbral_positiva (float): rating mínimo para que una review sea positiva.
                Por defecto 3.5 (es decir, las de 4 y 5 estrellas).
            umbral_negativa (float): rating máximo para que una review sea negativa.
                Por defecto 1.5 (es decir, solo las de 1 estrella). El resto (2 y 3
                estrellas) son neutras.
        """
        self.__fichero = fichero
        self.__campos = campos
        self.__carpeta_salida = carpeta_salida
        self.__umbral_positiva = umbral_positiva
        self.__umbral_negativa = umbral_negativa

    def _categoria(self, rating: float) -> str:
        """
        Decide la categoría de una review a partir de su puntuación.

        Parámetros:
            rating (float): puntuación de la review (de 1 a 5 estrellas).

        Devuelve:
            str: 'positivas', 'neutras' o 'negativas'.
        """
        if rating >= self.__umbral_positiva:
            return 'positivas'
        if rating <= self.__umbral_negativa:
            return 'negativas'
        return 'neutras'

    def preprocesar_texto(self) -> dict:
        """
        Recorre el fichero .jsonl, descarta duplicados y escribe un JSON por review.

        Se considera duplicada una review con el mismo usuario, título y texto que otra
        anterior (sin distinguir mayúsculas). Así se evita que las repeticiones
        aumenten todavía más el desbalanceo entre clases.

        Parámetros:
            (ninguno: usa los valores recibidos en el constructor)

        Devuelve:
            dict: número de reviews guardadas por categoría,
                p. ej. {'positivas': 130991, 'neutras': 5064, 'negativas': 12124}.
        """
        contadores = {'positivas': 0, 'neutras': 0, 'negativas': 0}
        for categoria in contadores:
            carpeta = os.path.join(self.__carpeta_salida, categoria)
            os.makedirs(carpeta, exist_ok=True)
            # Si ya hay reviews de otra ejecución (p. ej. con otros umbrales), al numerar de
            # nuevo quedarían ficheros viejos mezclados con los nuevos: se pide borrarlas antes.
            if any(nombre.endswith('.json') for nombre in os.listdir(carpeta)):
                raise RuntimeError(f'{carpeta} ya tiene reviews de otra ejecución. '
                                   f'Borra la carpeta {self.__carpeta_salida} y vuelve a ejecutar.')

        vistos = set()
        with open(self.__fichero, 'r', encoding='utf-8') as fp:
            for linea in fp:
                if not linea.strip():
                    continue
                original = json.loads(linea)

                # Detección de duplicados (user_id solo se usa aquí, no se guarda)
                clave = (original['user_id'], original['title'].lower(), original['text'].lower())
                if clave in vistos:
                    continue
                vistos.add(clave)

                # Nos quedamos solo con los campos pedidos
                review = {campo: original[campo] for campo in self.__campos}

                # Fichero <categoria>/<categoria>_<id>.json, con id correlativo por categoría
                categoria = self._categoria(original['rating'])
                contadores[categoria] += 1
                nombre = f'{categoria}_{contadores[categoria]}.json'
                ruta = os.path.join(self.__carpeta_salida, categoria, nombre)
                with open(ruta, 'w', encoding='utf-8') as f:
                    json.dump(review, f)

        return contadores


def main():
    """
    Punto de entrada del script: lee los argumentos de la línea de comandos,
    descarga las reviews y construye el corpus.

    Parámetros (de línea de comandos):
        --categoria: categoría de Amazon Reviews 2023 a descargar. Por defecto, Gift_Cards.
        --descargas: carpeta donde se guarda el .jsonl descargado. Por defecto, la actual.
        --salida: carpeta raíz del corpus que se genera. Por defecto, corpus.
        --forzar: vuelve a descargar el fichero aunque ya exista.
    """
    parser = argparse.ArgumentParser(description='Descarga reviews de Amazon y construye el corpus.')
    parser.add_argument('--categoria', default='Gift_Cards',
                        help='categoría de Amazon Reviews 2023 (por defecto, Gift_Cards)')
    parser.add_argument('--descargas', default='.',
                        help='carpeta donde se guarda el .jsonl descargado (por defecto, la actual)')
    parser.add_argument('--salida', default='corpus',
                        help='carpeta raíz del corpus (por defecto, corpus)')
    parser.add_argument('--forzar', action='store_true',
                        help='volver a descargar aunque el fichero ya exista')
    args = parser.parse_args()

    fichero = descargar_reviews(args.categoria, args.descargas, forzar=args.forzar)

    procesador = ProcesadorDeTexto(fichero, carpeta_salida=args.salida)
    contadores = procesador.preprocesar_texto()

    print(f'Corpus creado en {args.salida}/:')
    for categoria, n in contadores.items():
        print(f'  {categoria:<10} {n:>7} reviews')
    print(f'  {"total":<10} {sum(contadores.values()):>7} reviews')


if __name__ == '__main__':
    main()
