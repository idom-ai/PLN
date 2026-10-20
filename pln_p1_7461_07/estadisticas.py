"""
estadisticas.py - Estadísticas del corpus de reseñas de Amazon (Gift_Cards).

Usa los métodos de AmazonCorpusReader para analizar el corpus:
  - categories() / fileids()  -> estructura y balance de clases
  - describe()                -> métricas globales (con y sin stopwords)
  - ratings()                 -> distribución de estrellas
  - titles()                  -> longitud de los títulos
  - paras()                   -> longitud de cada reseña (palabras, oraciones)
  - words(sin_stopwords=True) -> palabras más frecuentes por categoría

Uso (desde la carpeta que contiene positivas/, neutras/ y negativas/):
  python3 estadisticas.py                   # corpus completo (tarda varios minutos)
  python3 estadisticas.py --muestra 2000    # máx. 2000 reseñas por categoría (rápido)
  python3 estadisticas.py --top 15          # nº de palabras frecuentes a mostrar

Genera además:
  estadisticas.json  -> todos los resultados
  estadisticas.tex   -> tablas listas para \\input{} en la memoria
"""
import argparse
import json
import random
import statistics
import time
from collections import Counter

import nltk

from AmazonCorpusReader import AmazonCorpusReader


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def preparar_nltk():
    """Descarga los recursos de NLTK que usa la clase si no están instalados."""
    for recurso, ruta in [("punkt_tab", "tokenizers/punkt_tab"),
                          ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(recurso, quiet=True)


def seleccionar_fileids(reader, categoria, muestra, semilla=42):
    """Ficheros de una categoría; si hay `muestra`, un subconjunto aleatorio fijo."""
    fileids = reader.fileids(categories=categoria)
    if muestra and muestra < len(fileids):
        fileids = sorted(random.Random(semilla).sample(fileids, muestra))
    return fileids


def es_palabra(token):
    """
    True si el token es una palabra: tiene al menos una letra (descarta '.', '!',
    '5'...) y no es un clítico como 's o 're. Se conserva n't por ser negación.
    """
    if token.startswith("'"):
        return False
    return any(c.isalpha() for c in token)


def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# ----------------------------------------------------------------------
# Cálculo de estadísticas
# ----------------------------------------------------------------------
def estadisticas_categoria(reader, fileids, top):
    """Calcula todas las estadísticas para un conjunto de ficheros."""
    res = {}

    # 1. describe(): métricas de corpus, con y sin stopwords
    res["describe"] = reader.describe(fileids=fileids)
    res["describe_sin_stopwords"] = reader.describe(fileids=fileids, sin_stopwords=True)

    # 2. ratings(): distribución de estrellas
    ratings = list(reader.ratings(fileids=fileids))
    res["ratings"] = {
        "media": statistics.mean(ratings),
        "distribucion": {str(k): v for k, v in sorted(Counter(ratings).items())},
    }

    # 3. titles(): longitud media del título en palabras
    long_titulos = [len(t.split()) for t in reader.titles(fileids=fileids)]
    res["titulo_palabras_media"] = statistics.mean(long_titulos)

    # 4. paras(): longitud de cada reseña (palabras y oraciones)
    palabras_por_resena, oraciones_por_resena = [], []
    for doc in reader.paras(fileids=fileids):
        oraciones = [sent for para in doc for sent in para]
        oraciones_por_resena.append(len(oraciones))
        palabras_por_resena.append(sum(len(s) for s in oraciones))
    res["longitud_resena"] = {
        "palabras_media": statistics.mean(palabras_por_resena),
        "palabras_mediana": statistics.median(palabras_por_resena),
        "palabras_max": max(palabras_por_resena),
        "oraciones_media": statistics.mean(oraciones_por_resena),
        "resenas_vacias": palabras_por_resena.count(0),
        "resenas_1_oracion": oraciones_por_resena.count(1),
    }

    # 5. words(sin_stopwords=True): palabras más frecuentes (sin puntuación)
    fd = nltk.FreqDist(w.lower() for w in reader.words(fileids=fileids, sin_stopwords=True)
                       if es_palabra(w))
    res["top_palabras"] = fd.most_common(top)
    res["_freqdist"] = fd  # se usa para las palabras características (no va al JSON)
    return res


def palabras_caracteristicas(resultados, top, min_frec=20):
    """
    Palabras más 'propias' de cada categoría: las que tienen mayor frecuencia
    relativa en esa categoría frente al resto (ratio con suavizado +1).
    """
    salida = {}
    for cat, res in resultados.items():
        fd = res["_freqdist"]
        resto = nltk.FreqDist()
        for otra, res_otra in resultados.items():
            if otra != cat:
                resto.update(res_otra["_freqdist"])
        n_cat, n_resto = fd.N() or 1, resto.N() or 1
        ratios = [(w, (c / n_cat) / ((resto[w] + 1) / n_resto))
                  for w, c in fd.items() if c >= min_frec]
        salida[cat] = [w for w, _ in sorted(ratios, key=lambda x: -x[1])[:top]]
    return salida


# ----------------------------------------------------------------------
# Salida por pantalla
# ----------------------------------------------------------------------
def imprimir(reader, resultados, caracteristicas, muestra, segundos):
    cats = list(resultados)

    titulo("1. ESTRUCTURA DEL CORPUS")
    total_ficheros = sum(len(reader.fileids(categories=c)) for c in cats)
    print(f"Categorías: {cats}")
    print(f"{'categoría':<11}{'reseñas':>10}{'% corpus':>10}{'analizadas':>12}")
    for c in cats:
        n = len(reader.fileids(categories=c))
        print(f"{c:<11}{n:>10}{100 * n / total_ficheros:>9.1f}%"
              f"{resultados[c]['describe']['files']:>12}")
    print(f"{'TOTAL':<11}{total_ficheros:>10}{100.0:>9.1f}%")
    if muestra:
        print(f"(Análisis sobre una muestra de hasta {muestra} reseñas por categoría)")

    titulo("2. DESCRIBE() POR CATEGORÍA")
    claves = ["files", "chars", "paras", "sents", "words", "vocab",
              "lexdiv", "ppdoc", "sppar", "secs"]
    for etiqueta, clave in [("Con stopwords", "describe"),
                            ("Sin stopwords (se conservan no/nor/not)", "describe_sin_stopwords")]:
        print(f"\n{etiqueta}:")
        print(f"{'métrica':<9}" + "".join(f"{c:>14}" for c in cats))
        for k in claves:
            fila = [resultados[c][clave][k] for c in cats]
            print(f"{k:<9}" + "".join(f"{v:>14.2f}" if isinstance(v, float) else f"{v:>14}"
                                      for v in fila))

    print("\nEfecto de quitar stopwords sobre el nº de palabras:")
    for c in cats:
        con = resultados[c]["describe"]["words"]
        sin = resultados[c]["describe_sin_stopwords"]["words"]
        print(f"  {c:<10} {con:>10} -> {sin:>10}  ({100 * (con - sin) / con:.1f}% eran stopwords)")

    titulo("3. RATINGS (ratings())")
    for c in cats:
        r = resultados[c]["ratings"]
        print(f"  {c:<10} media {r['media']:.2f}  distribución {r['distribucion']}")

    titulo("4. LONGITUD DE LAS RESEÑAS (paras() y titles())")
    print(f"{'categoría':<11}{'pal/reseña':>11}{'mediana':>9}{'máx':>7}"
          f"{'orac/reseña':>13}{'1 oración':>11}{'vacías':>8}{'pal/título':>12}")
    for c in cats:
        l = resultados[c]["longitud_resena"]
        n = resultados[c]["describe"]["files"]
        print(f"{c:<11}{l['palabras_media']:>11.1f}{l['palabras_mediana']:>9.0f}"
              f"{l['palabras_max']:>7}{l['oraciones_media']:>13.2f}"
              f"{100 * l['resenas_1_oracion'] / n:>10.1f}%{l['resenas_vacias']:>8}"
              f"{resultados[c]['titulo_palabras_media']:>12.2f}")

    titulo("5. PALABRAS MÁS FRECUENTES SIN STOPWORDS (words())")
    for c in cats:
        print(f"\n  {c}:")
        print("   " + ", ".join(f"{w} ({n})" for w, n in resultados[c]["top_palabras"]))

    titulo("6. PALABRAS CARACTERÍSTICAS DE CADA CATEGORÍA")
    print("  (mayor frecuencia relativa en la categoría que en el resto)")
    for c in cats:
        print(f"  {c:<10} {', '.join(caracteristicas[c])}")

    print(f"\nTiempo total: {segundos:.1f} s")


# ----------------------------------------------------------------------
# Exportación
# ----------------------------------------------------------------------
def guardar_json(resultados, caracteristicas, ruta):
    datos = {c: {k: v for k, v in r.items() if not k.startswith("_")}
             for c, r in resultados.items()}
    for c in datos:
        datos[c]["palabras_caracteristicas"] = caracteristicas[c]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def guardar_latex(reader, resultados, ruta):
    cats = list(resultados)
    total = sum(len(reader.fileids(categories=c)) for c in cats)
    def fmt(v):
        """Formato español: punto para miles y coma para decimales (12.100 / 8,17)."""
        texto = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    filas = [
        ("Reseñas en el corpus", [len(reader.fileids(categories=c)) for c in cats]),
        ("\\% del corpus", [100 * len(reader.fileids(categories=c)) / total for c in cats]),
        ("Reseñas analizadas", [resultados[c]["describe"]["files"] for c in cats]),
        ("Caracteres", [resultados[c]["describe"]["chars"] for c in cats]),
        ("Párrafos", [resultados[c]["describe"]["paras"] for c in cats]),
        ("Oraciones", [resultados[c]["describe"]["sents"] for c in cats]),
        ("Palabras", [resultados[c]["describe"]["words"] for c in cats]),
        ("Palabras (sin stopwords)", [resultados[c]["describe_sin_stopwords"]["words"] for c in cats]),
        ("Vocabulario", [resultados[c]["describe"]["vocab"] for c in cats]),
        ("\\texttt{lexdiv}", [resultados[c]["describe"]["lexdiv"] for c in cats]),
        ("Párrafos por reseña", [resultados[c]["describe"]["ppdoc"] for c in cats]),
        ("Oraciones por párrafo", [resultados[c]["describe"]["sppar"] for c in cats]),
        ("Palabras por reseña", [resultados[c]["longitud_resena"]["palabras_media"] for c in cats]),
        ("Rating medio", [resultados[c]["ratings"]["media"] for c in cats]),
    ]
    lineas = [
        "\\begin{table}[h]", "\\centering",
        "\\begin{tabular}{l" + "r" * len(cats) + "}", "\\toprule",
        "Métrica & " + " & ".join(cats) + " \\\\", "\\midrule",
    ]
    lineas += [f"{nombre} & " + " & ".join(fmt(v) for v in vals) + " \\\\" for nombre, vals in filas]
    lineas += ["\\bottomrule", "\\end{tabular}",
               "\\caption{Estadísticas del corpus por categoría (\\texttt{AmazonCorpusReader}).}",
               "\\label{tab:estadisticas}", "\\end{table}", ""]
    lineas += ["\\begin{table}[h]", "\\centering", "\\begin{tabularx}{\\textwidth}{lX}",
               "\\toprule", "Categoría & Palabras más frecuentes (sin stopwords) \\\\", "\\midrule"]
    lineas += [f"{c} & " + ", ".join(w for w, _ in resultados[c]["top_palabras"]) + " \\\\"
               for c in cats]
    lineas += ["\\bottomrule", "\\end{tabularx}",
               "\\caption{Palabras más frecuentes por categoría.}",
               "\\label{tab:top-palabras}", "\\end{table}"]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Estadísticas del corpus de reseñas de Amazon")
    parser.add_argument("--root", default=".", help="carpeta raíz del corpus (por defecto, la actual)")
    parser.add_argument("--muestra", type=int, default=None,
                        help="máximo de reseñas por categoría (por defecto, todas)")
    parser.add_argument("--top", type=int, default=10, help="nº de palabras frecuentes a mostrar")
    args = parser.parse_args()

    preparar_nltk()
    inicio = time.time()
    reader = AmazonCorpusReader(args.root)

    resultados = {}
    for cat in reader.categories():
        fileids = seleccionar_fileids(reader, cat, args.muestra)
        print(f"Analizando '{cat}' ({len(fileids)} reseñas)...", flush=True)
        resultados[cat] = estadisticas_categoria(reader, fileids, args.top)

    caracteristicas = palabras_caracteristicas(resultados, args.top)
    imprimir(reader, resultados, caracteristicas, args.muestra, time.time() - inicio)

    guardar_json(resultados, caracteristicas, "estadisticas.json")
    guardar_latex(reader, resultados, "estadisticas.tex")
    print("\nResultados guardados en estadisticas.json y estadisticas.tex")


if __name__ == "__main__":
    main()
