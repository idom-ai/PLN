"""
prueba.py - Demostración de los requisitos del Ejercicio 3 (AmazonCorpusReader).

Qué hace:
  1. Construye un corpus de juguete (en una carpeta temporal) con reseñas
     pensadas para provocar cada caso: HTML, emojis, signos y letras repetidos,
     párrafos en blanco, negaciones, una reseña que queda vacía...
  2. Comprueba con asserts que cada requisito se cumple y lo imprime.
  3. Muestra casos de uso (consultas típicas sobre el corpus).
  4. Opcional: `python3 prueba.py --real` repite lo básico sobre tu corpus
     real (carpetas positivas/, neutras/, negativas/ en la carpeta actual).

Uso:  python3 prueba.py
"""
import json
import os
import sys
import tempfile

import nltk

from AmazonCorpusReader import AmazonCorpusReader

FALLOS = []


def check(condicion, mensaje):
    """Imprime [OK] o [FALLO] y recuerda los fallos para el resumen final."""
    estado = "OK   " if condicion else "FALLO"
    print(f"  [{estado}] {mensaje}")
    if not condicion:
        FALLOS.append(mensaje)


def titulo(texto):
    print(f"\n=== {texto} ===")


# ----------------------------------------------------------------------
# Corpus de juguete
# ----------------------------------------------------------------------
RESENAS = {
    "positivas": [
        {"rating": 5.0, "title": "Great gift", "parent_asin": "A1",
         "text": "This is a gooooood gift!!! <br />The card arrived fast "
                 "&amp; I love it \U0001F600\U0001F600"},
        {"rating": 5.0, "title": "Perfect", "parent_asin": "A2",
         "text": "Perfect for my mom.\n\n\n\n   \n\nThe price was fine."},
    ],
    "neutras": [
        {"rating": 3.0, "title": "Okay", "parent_asin": "A3",
         "text": "It is <b>okay</b>, but the delivery was slow..."},
    ],
    "negativas": [
        {"rating": 1.0, "title": "Bad", "parent_asin": "A4",
         "text": "This is not good at all??? I don't like it. "
                 "The code did not work."},
        {"rating": 1.0, "title": "Empty", "parent_asin": "A5",
         "text": "<br /><br />   \U0001F621\U0001F621"},
    ],
}


def crear_corpus_juguete(raiz):
    for categoria, resenas in RESENAS.items():
        os.makedirs(os.path.join(raiz, categoria), exist_ok=True)
        for i, resena in enumerate(resenas, start=1):
            ruta = os.path.join(raiz, categoria, f"{categoria}_{i}.json")
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(resena, f)


# ----------------------------------------------------------------------
# Pruebas
# ----------------------------------------------------------------------
def prueba_estructura(r):
    titulo("1. Estructura del corpus (ficheros y categorías)")
    check(r.categories() == ["negativas", "neutras", "positivas"],
          f"categorías detectadas a partir de las carpetas: {r.categories()}")
    check(len(r.fileids()) == 5,
          f"ficheros del corpus: {len(r.fileids())} (esperados 5)")
    check(len(r.fileids(categories="negativas")) == 2,
          "fileids(categories='negativas') devuelve 2 ficheros")
    cats = [d["category"] for d in r.docs()]
    check(sorted(set(cats)) == ["negativas", "neutras", "positivas"],
          "cada documento lleva su categoría en el campo 'category'")


def prueba_limpieza(r):
    titulo("2. Limpieza de texto (_limpiar)")
    textos = {d["title"]: d["text"] for d in r.docs()}
    t = textos["Great gift"]
    print(f"  Texto original : {RESENAS['positivas'][0]['text']!r}")
    print(f"  Texto limpio   : {t!r}")
    check("<br" not in t and "<" not in t, "etiquetas HTML eliminadas")
    check("&amp;" not in t and "&" in t, "entidad HTML '&amp;' convertida en '&'")
    check("\U0001F600" not in t, "emojis eliminados")
    check("!!!" not in t and "!" in t, "signos repetidos '!!!' reducidos a '!'")
    check("gooooood" not in t and "good" in t,
          "letras repetidas 'gooooood' reducidas a 'good'")
    check("<b>" not in textos["Okay"] and "okay" in textos["Okay"],
          "etiqueta <b> eliminada conservando el contenido")
    check("..." not in textos["Okay"], "puntos suspensivos '...' reducidos")
    check("???" not in textos["Bad"] and "?" in textos["Bad"],
          "signos repetidos '???' reducidos a '?'")
    # Los números no se tocan
    check(r._limpiar("I paid 1000 for it") == "I paid 1000 for it",
          "los números no se alteran (1000 sigue siendo 1000)")
    # El fichero original NO se modifica
    crudo = r.raw(categories="positivas")
    check("<br />" in crudo and "gooooood" in crudo,
          "raw() sigue devolviendo el texto original: el disco no se modifica")


def prueba_blancos(r):
    titulo("3. Eliminación de párrafos, oraciones y palabras en blanco")
    hay_parrafo_vacio = hay_oracion_vacia = hay_palabra_vacia = False
    for doc in r.paras():
        for para in doc:
            if not para:
                hay_parrafo_vacio = True
            for sent in para:
                if not sent:
                    hay_oracion_vacia = True
                for w in sent:
                    if not w.strip():
                        hay_palabra_vacia = True
    check(not hay_parrafo_vacio, "ningún párrafo vacío en paras()")
    check(not hay_oracion_vacia, "ninguna oración vacía en paras()")
    check(not hay_palabra_vacia, "ninguna palabra vacía o de solo espacios")
    # La reseña 'Perfect' tenía párrafos en blanco entre medias
    docs = {d["title"]: p for d, p in zip(r.docs(), r.paras())}
    check(len(docs["Perfect"]) == 2,
          "'Perfect' tenía párrafos en blanco entre medias: quedan 2 párrafos reales")
    # La reseña 'Empty' solo tenía HTML y emojis
    check(docs["Empty"] == [],
          "reseña con solo HTML/emojis queda vacía y no genera elementos fantasma")
    check(len(list(r.paras())) == len(list(r.texts())),
          "paras() devuelve una entrada por reseña (queda alineado con texts())")


def prueba_stopwords(r):
    titulo("4. Stopwords: versión completa y versión filtrada")
    con = list(r.words(categories="negativas"))
    sin = list(r.words(categories="negativas", sin_stopwords=True))
    print(f"  Con stopwords    : {con}")
    print(f"  Sin stopwords    : {sin}")
    check(len(sin) < len(con), f"filtrar reduce las palabras ({len(con)} -> {len(sin)})")
    check("This" in con and "This" not in sin and "this" not in sin,
          "'This' es stopword: está en la versión completa y no en la filtrada")
    check("not" in sin, "la negación 'not' se conserva al filtrar")
    check("good" in sin, "'good' se conserva (no es stopword)")
    check("n't" in sin, "la negación \"n't\" de \"don't\" se conserva")
    # Por defecto NO se filtra
    check(list(r.words(categories="negativas")) == con,
          "por defecto words() devuelve todo (sin_stopwords=False)")
    # sents() y paras() también aceptan el parámetro
    s_con = list(r.sents(categories="negativas"))
    s_sin = list(r.sents(categories="negativas", sin_stopwords=True))
    check(sum(map(len, s_sin)) == len(sin) and sum(map(len, s_con)) == len(con),
          "sents() y words() son coherentes entre sí")


def prueba_describe(r):
    titulo("5. describe(): estadísticas del corpus")
    d = r.describe()
    claves = ["files", "topics", "chars", "paras", "sents", "words",
              "vocab", "lexdiv", "ppdoc", "sppar", "secs"]
    check(all(k in d for k in claves), f"contiene las claves: {claves}")
    check(d["files"] == 5 and d["topics"] == 3,
          f"5 ficheros y 3 categorías (files={d['files']}, topics={d['topics']})")
    check(d["chars"] > 0, f"tamaño total en caracteres: {d['chars']}")
    check(d["vocab"] <= d["words"], "el vocabulario no supera al número de palabras")
    # El vocabulario cuenta en minúsculas: 'The' y 'the' son la misma palabra
    esperado = len({w.lower() for w in r.words()})
    check(d["vocab"] == esperado, f"vocabulario en minúsculas: {d['vocab']} palabras distintas")
    ds = r.describe(sin_stopwords=True)
    check(ds["words"] < d["words"], f"sin stopwords hay menos palabras ({d['words']} -> {ds['words']})")
    # lexdiv = palabras / vocabulario = veces que se repite cada palabra de media.
    # Las stopwords son las más repetidas, así que al quitarlas baja.
    check(ds["lexdiv"] < d["lexdiv"],
          f"sin stopwords baja la repetición media de palabras, lexdiv ({d['lexdiv']:.2f} -> {ds['lexdiv']:.2f})")
    dn = r.describe(categories="negativas")
    check(dn["topics"] == 1 and dn["files"] == 2,
          "describe(categories='negativas') solo cuenta esa categoría")


def casos_de_uso(r):
    titulo("6. Casos de uso")

    print("\n  a) Reseñas y rating por categoría:")
    for cat in r.categories():
        ratings = list(r.ratings(categories=cat))
        print(f"     {cat:<10} -> {len(ratings)} reseñas, ratings {ratings}")

    print("\n  b) Títulos de las reseñas negativas:")
    print(f"     {list(r.titles(categories='negativas'))}")

    print("\n  c) Primera reseña positiva, troceada (párrafos > oraciones > palabras):")
    print(f"     {next(r.paras(categories='positivas'))}")

    print("\n  d) Palabras más frecuentes de las negativas, con y sin stopwords:")
    for flag in (False, True):
        fd = nltk.FreqDist(w.lower() for w in
                           r.words(categories="negativas", sin_stopwords=flag))
        etiqueta = "sin stopwords" if flag else "con stopwords"
        print(f"     {etiqueta}: {fd.most_common(5)}")

    print("\n  e) Resumen comparativo por categoría (con stopwords):")
    print(f"     {'categoría':<10} {'reseñas':>7} {'palabras':>9} {'vocab':>6} {'lexdiv':>7}")
    for cat in r.categories():
        d = r.describe(categories=cat)
        n = len(list(r.texts(categories=cat)))
        print(f"     {cat:<10} {n:>7} {d['words']:>9} {d['vocab']:>6} {d['lexdiv']:>7.2f}")


def prueba_corpus_real():
    titulo("7. Corpus real (carpeta actual)")
    carpetas = ["positivas", "neutras", "negativas"]
    if not all(os.path.isdir(c) for c in carpetas):
        print("  No se encuentran las carpetas positivas/, neutras/ y negativas/ "
              "en la carpeta actual. Ejecuta este script desde la carpeta del corpus.")
        return
    r = AmazonCorpusReader(".")
    print(f"  Categorías: {r.categories()}")
    for cat in r.categories():
        print(f"  {cat}: {len(r.fileids(categories=cat))} ficheros")
    cat = r.categories()[0]
    print(f"\n  describe() sobre '{cat}' (puede tardar):")
    for k, v in r.describe(categories=cat).items():
        print(f"     {k:<7} {v}")


def main():
    with tempfile.TemporaryDirectory() as raiz:
        crear_corpus_juguete(raiz)
        r = AmazonCorpusReader(raiz)

        prueba_estructura(r)
        prueba_limpieza(r)
        prueba_blancos(r)
        prueba_stopwords(r)
        prueba_describe(r)
        casos_de_uso(r)

    if "--real" in sys.argv:
        prueba_corpus_real()

    titulo("RESUMEN")
    if FALLOS:
        print(f"  {len(FALLOS)} comprobación(es) fallida(s):")
        for f in FALLOS:
            print(f"   - {f}")
        sys.exit(1)
    print("  Todas las comprobaciones han pasado.")


if __name__ == "__main__":
    main()