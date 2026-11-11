"""
experimento_calidad_corpus.py - EXPERIMENTO 1: cómo cambian los modelos al mejorar el corpus.

Para cada tamaño de muestra (5000, 10000, 20000 y 40000 reviews), con la proporción
original de clases (~84 % positivas, ~8 % negativas, ~8 % neutras):
  1. Muestra por pantalla las métricas de calidad del corpus (describe(), total y por clase).
  2. Entrena LogisticRegression, MultinomialNB y SGDClassifier con su configuración por
     defecto, con validación cruzada estratificada de 3 folds (CorpusLoader).
  3. Muestra por pantalla la matriz de confusión (suma de los folds) y las métricas de cada modelo.

Al final guarda en graficas/:
  evolucion_<clase>_<corpus>.png (negativas, neutras y positivas)
      accuracy de la clase (recall) de cada modelo frente al vocabulario de esa clase
  matrices_confusion_<corpus>.png
      matrices de confusión (filas: tamaños; columnas: modelos)

Los folds de cada tamaño se ejecutan en paralelo (un proceso por fold). Lo que sale por
pantalla se guarda también en logs/calidad_<corpus>_<fecha>/salida.log.

Uso (desde la carpeta de la práctica):
  python3 experimento_calidad_corpus.py                              # corpus/
  python3 experimento_calidad_corpus.py --corpus corpus_pos5_neu234  # otro reparto de clases
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')  # dibuja en fichero, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from corpus_loader import CorpusLoader
from json_corpus_reader import JsonCorpusReader
from registro import Registro
from sentiment_classifier import SentimentClassifier

CORPUS = 'corpus'
GRAFICAS = 'graficas'
TAMANOS = [20000, 40000, 80000, 160000]
FOLDS = 3
CLASES = ['negativas', 'neutras', 'positivas']

# Los 3 modelos, con su configuración por defecto (la misma en todos los tamaños)
MODELOS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'SGDClassifier': SGDClassifier(random_state=42),
}

# Color (paleta categórica validada) y marcador de cada modelo: así se distinguen
# aunque no se vea bien el color
ESTILO = {
    'LogisticRegression': {'color': '#2a78d6', 'marker': 'o'},
    'MultinomialNB': {'color': '#eb6834', 'marker': 's'},
    'SGDClassifier': {'color': '#1baf7a', 'marker': '^'},
}


# ----------------------------------------------------------------------
# Cálculo
# ----------------------------------------------------------------------
def calidad_corpus(reader, loader):
    """
    Calcula las métricas de describe() de la muestra, en total y para cada clase.

    Parámetros:
        reader (JsonCorpusReader): lector del corpus.
        loader (CorpusLoader): cargador con la muestra de reviews.

    Devuelve:
        dict: {'total': {...}, 'negativas': {...}, 'neutras': {...}, 'positivas': {...}},
            cada uno con las métricas de describe().
    """
    fileids = list(loader.fileids())
    etiquetas = loader.labels()
    calidad = {'total': reader.describe(fileids=fileids)}
    for clase in CLASES:
        calidad[clase] = reader.describe(fileids=[f for f, e in zip(fileids, etiquetas) if e == clase])
    return calidad


def evaluar_fold(corpus, tamano, train_idx, test_idx):
    """
    Trabajo de un fold, que se ejecuta en su propio proceso: entrena los 3 modelos con el
    train del fold y calcula su matriz de confusión en el test del fold.

    Parámetros:
        corpus (str): carpeta del corpus.
        tamano (int): tamaño de la muestra (el mismo que en el proceso principal, para que el
            loader tenga las mismas reviews en el mismo orden).
        train_idx (numpy.ndarray): posiciones de las reviews de entrenamiento en el CorpusLoader.
        test_idx (numpy.ndarray): posiciones de las reviews de test.

    Devuelve:
        dict: {nombre del modelo: matriz de confusión 3x3 (numpy.ndarray)} de este fold.
    """
    loader = CorpusLoader(JsonCorpusReader(corpus), folds=FOLDS, n_total=tamano)
    X_train, X_test = loader.documents(train_idx), loader.documents(test_idx)
    y_train, y_test = loader.labels(train_idx), loader.labels(test_idx)
    matrices = {}
    for nombre, estimador in MODELOS.items():
        modelo = SentimentClassifier(estimator=estimador).fit(X_train, y_train)
        matrices[nombre] = confusion_matrix(y_test, modelo.predict(X_test), labels=CLASES)
    return matrices


def evaluar_modelos(loader, corpus, tamano):
    """
    Entrena y evalúa los 3 modelos con validación cruzada (un proceso por fold) y suma sus
    matrices de confusión.

    Parámetros:
        loader (CorpusLoader): cargador con la muestra; su splitter da los folds.
        corpus (str): carpeta del corpus (cada proceso crea su propio reader).
        tamano (int): tamaño de la muestra con la que se creó el loader.

    Devuelve:
        dict: {nombre del modelo: matriz de confusión 3x3 (numpy.ndarray)}, con
            filas = clase real y columnas = clase predicha, en el orden de CLASES.
    """
    matrices = {nombre: np.zeros((len(CLASES), len(CLASES)), dtype=int) for nombre in MODELOS}
    particiones = list(loader.splitter.split(loader.fileids(), loader.labels()))
    with ProcessPoolExecutor(max_workers=FOLDS) as procesos:
        tareas = [procesos.submit(evaluar_fold, corpus, tamano, tr, te) for tr, te in particiones]
        for tarea in tareas:
            for nombre, m in tarea.result().items():
                matrices[nombre] += m
    return matrices


def metricas_matriz(matriz):
    """
    Calcula las métricas de un modelo a partir de su matriz de confusión.

    Parámetros:
        matriz (numpy.ndarray): matriz de confusión (filas: clase real; columnas: predicha).

    Devuelve:
        dict: 'accuracy' (aciertos / total), 'f1_macro' (media del F1 de las clases) y, por
            cada clase, su accuracy (= recall: aciertos de esa clase / reviews de esa clase).
    """
    m = matriz.astype(float)
    aciertos = np.diag(m)
    recall = np.divide(aciertos, m.sum(axis=1), out=np.zeros(3), where=m.sum(axis=1) > 0)
    precision = np.divide(aciertos, m.sum(axis=0), out=np.zeros(3), where=m.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3),
                   where=(precision + recall) > 0)
    resultado = {'accuracy': aciertos.sum() / m.sum(), 'f1_macro': f1.mean()}
    resultado.update(dict(zip(CLASES, recall)))
    return resultado


# ----------------------------------------------------------------------
# Salida por pantalla
# ----------------------------------------------------------------------
def imprimir_calidad(calidad):
    """
    Muestra una tabla con las métricas de describe() (filas) para el total y cada clase (columnas).

    Parámetros:
        calidad (dict): resultado de calidad_corpus().
    """
    columnas = ['total'] + CLASES
    print('  Calidad del corpus (describe()):')
    print(f'    {"métrica":<8}' + ''.join(f'{c:>12}' for c in columnas))
    for clave in ['files', 'chars', 'paras', 'sents', 'words', 'vocab', 'lexdiv', 'ppdoc', 'sppar']:
        valores = [calidad[c][clave] for c in columnas]
        print(f'    {clave:<8}' + ''.join(f'{v:>12.2f}' if isinstance(v, float) else f'{v:>12}'
                                       for v in valores))


def imprimir_modelos(matrices, metricas):
    """
    Muestra, para cada modelo, su matriz de confusión y sus métricas.

    Parámetros:
        matrices (dict): matriz de confusión de cada modelo (resultado de evaluar_modelos()).
        metricas (dict): métricas de cada modelo (resultado de metricas_matriz()).
    """
    for nombre in MODELOS:
        m, r = matrices[nombre], metricas[nombre]
        print(f'\n  {nombre}: accuracy {r["accuracy"]:.3f} | F1 macro {r["f1_macro"]:.3f} | '
              + ' | '.join(f'accuracy {c} {r[c]:.3f}' for c in CLASES))
        print('    ' + 'real / predicha'.ljust(16) + ''.join(f'{c:>11}' for c in CLASES))
        for clase, fila in zip(CLASES, m):
            print(f'    {clase:<16}' + ''.join(f'{v:>11}' for v in fila))


def imprimir_resumen(resultados):
    """
    Muestra una tabla final: accuracy de cada clase por modelo y tamaño, junto al vocabulario
    de esa clase (los datos de las gráficas).

    Parámetros:
        resultados (dict): {tamaño: {'calidad', 'matrices', 'metricas'}} (ver main()).
    """
    print('\n=== RESUMEN (datos de las gráficas) ===')
    print(f'{"tamaño":>7} {"modelo":<19}{"accuracy":>9}{"F1 macro":>9}'
          + ''.join(f'{"acc " + c[:3]:>9}{"vocab " + c[:3]:>11}' for c in CLASES))
    for tamano, r in resultados.items():
        for nombre in MODELOS:
            m = r['metricas'][nombre]
            print(f'{tamano:>7} {nombre:<19}{m["accuracy"]:>9.3f}{m["f1_macro"]:>9.3f}'
                  + ''.join(f'{m[c]:>9.3f}{r["calidad"][c]["vocab"]:>11}' for c in CLASES))


# ----------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------
def grafica_clase(resultados, clase, ruta):
    """
    Dibuja la accuracy de la clase (recall) de cada modelo frente al vocabulario de esa clase
    en la muestra. Cada punto es un tamaño de muestra.

    Parámetros:
        resultados (dict): {tamaño: {'calidad', 'matrices', 'metricas'}} (ver main()).
        clase (str): 'negativas', 'neutras' o 'positivas'.
        ruta (str): fichero .png donde se guarda.
    """
    tamanos = list(resultados)
    x = [resultados[t]['calidad'][clase]['vocab'] for t in tamanos]

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    finales = []
    for nombre, estilo in ESTILO.items():
        y = [resultados[t]['metricas'][nombre][clase] for t in tamanos]
        ax.plot(x, y, color=estilo['color'], marker=estilo['marker'], linewidth=2, markersize=7,
                markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=nombre)
        finales.append(y[-1])

    # valor final de cada línea a su derecha; si dos están muy juntos, se separan un poco
    posiciones = {}
    for i in sorted(range(len(finales)), key=lambda k: finales[k]):
        cercanos = [p for p in posiciones.values() if finales[i] - p < 0.05]
        posiciones[i] = max(cercanos) + 0.05 if cercanos else finales[i]
    for i, valor in enumerate(finales):
        ax.annotate(f'{valor:.2f}', (x[-1], posiciones[i]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=8, color='#52514e', annotation_clip=False)

    # una marca en el eje X por tamaño: vocabulario y, debajo, tamaño de la muestra
    ax.set_xticks(x, [f'{v:,}'.replace(',', '.') for v in x])
    for xi, t in zip(x, tamanos):
        ax.annotate(f'({t // 1000}k reviews)', (xi, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -26), textcoords='offset points', ha='center', fontsize=7,
                    color='#898781')

    ax.set_xlabel(f'Vocabulario de las reviews {clase} de la muestra', color='#52514e', labelpad=16)
    ax.set_ylabel('Accuracy en la clase (recall)', color='#52514e')
    ax.set_title(f'Reviews {clase}: accuracy según la calidad del corpus',
                 color='#0b0b0b', loc='left', fontsize=11, pad=24)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color='#e1e0d9', linewidth=0.8)
    ax.tick_params(colors='#898781', labelsize=8)
    for lado in ['top', 'right']:
        ax.spines[lado].set_visible(False)
    for lado in ['left', 'bottom']:
        ax.spines[lado].set_color('#c3c2b7')
    # leyenda en una fila entre el título y la gráfica, para que no tape ninguna línea
    ax.legend(frameon=False, fontsize=8, ncol=len(ESTILO), loc='lower left',
              bbox_to_anchor=(0, 1.0), borderaxespad=0.3, handlelength=2.5)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def grafica_matrices(resultados, ruta):
    """
    Dibuja las matrices de confusión: una fila por tamaño y una columna por modelo.
    El color es el porcentaje de cada fila (clase real); el texto, el número de reviews.

    Parámetros:
        resultados (dict): {tamaño: {'calidad', 'matrices', 'metricas'}} (ver main()).
        ruta (str): fichero .png donde se guarda.
    """
    tamanos = list(resultados)
    fig, ejes = plt.subplots(len(tamanos), len(MODELOS), squeeze=False, dpi=130,
                             figsize=(3.2 * len(MODELOS), 2.9 * len(tamanos)))
    fig.patch.set_facecolor('#fcfcfb')
    cortas = ['neg', 'neu', 'pos']
    for i, t in enumerate(tamanos):
        for j, nombre in enumerate(MODELOS):
            ax = ejes[i][j]
            m = resultados[t]['matrices'][nombre]
            proporcion = m / m.sum(axis=1, keepdims=True)
            ax.imshow(proporcion, cmap='Blues', vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, m[a, b], ha='center', va='center', fontsize=8,
                            color='#ffffff' if proporcion[a, b] > 0.6 else '#0b0b0b')
            ax.set_xticks(range(3), cortas, fontsize=7)
            ax.set_yticks(range(3), cortas, fontsize=7)
            if i == 0:
                ax.set_title(nombre, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{t // 1000}k\nreal', fontsize=8)
            if i == len(tamanos) - 1:
                ax.set_xlabel('predicha', fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main():
    """
    Ejecuta el experimento 1 completo: muestra los resultados por pantalla (y en el log) y
    guarda las gráficas en graficas/.

    Parámetros (de línea de comandos):
        --corpus: carpeta del corpus. Por defecto, corpus.
    """
    parser = argparse.ArgumentParser(description='Modelos por defecto con muestras de distinto tamaño.')
    parser.add_argument('--corpus', default=CORPUS, help=f'carpeta del corpus (por defecto, {CORPUS})')
    args = parser.parse_args()
    nombre_corpus = os.path.basename(os.path.normpath(args.corpus))

    with Registro(f'calidad_{nombre_corpus}'):
        reader = JsonCorpusReader(args.corpus)
        resultados = {}
        for tamano in TAMANOS:
            inicio = time.time()
            print(f'\n=== {tamano} reviews ===')
            loader = CorpusLoader(reader, folds=FOLDS, n_total=tamano)
            calidad = calidad_corpus(reader, loader)
            imprimir_calidad(calidad)
            matrices = evaluar_modelos(loader, args.corpus, tamano)
            metricas = {nombre: metricas_matriz(m) for nombre, m in matrices.items()}
            imprimir_modelos(matrices, metricas)
            resultados[tamano] = {'calidad': calidad, 'matrices': matrices, 'metricas': metricas}
            print(f'\n  ({time.time() - inicio:.0f} s)', flush=True)

        imprimir_resumen(resultados)

        os.makedirs(GRAFICAS, exist_ok=True)
        for clase in CLASES:
            grafica_clase(resultados, clase,
                          os.path.join(GRAFICAS, f'evolucion_{clase}_{nombre_corpus}.png'))
        grafica_matrices(resultados, os.path.join(GRAFICAS, f'matrices_confusion_{nombre_corpus}.png'))
        print(f'\nGráficas guardadas en {GRAFICAS}/ (sufijo _{nombre_corpus})')


if __name__ == '__main__':
    main()
