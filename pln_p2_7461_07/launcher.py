"""
launcher.py - EJERCICIO 5 / EXPERIMENTO 2: configuración óptima de cada modelo.

Pasos (los del enunciado):
  1. Crea el CorpusReader (JsonCorpusReader).
  2. Crea el CorpusLoader con todo el corpus y 3 folds estratificados.
  3. Por cada fold, crea un GridSearchCV con el SentimentClassifier que prueba:
       a. los modelos LogisticRegression, MultinomialNB y SGDClassifier,
       b. reducción con TruncatedSVD (300 dimensiones), con SelectKBest (chi2, 5000) y sin reducción,
       c. n-gramas (1, 1) y (1, 2),
     y además el peso de las clases (para que equivocarse con neutras y negativas cueste más
     al entrenar): sin pesos, 'balanced' y unos pesos agresivos; y alpha en NB.
  4. Ejecuta todas las pruebas, con un proceso por fold (los folds van en paralelo y, dentro
     de cada uno, GridSearchCV reparte las configuraciones entre los núcleos que quedan).
     La mejor configuración se elige por F1 macro con validación cruzada interna sobre el
     train del fold; el test del fold solo se usa para medirla.
  5. Muestra por pantalla la mejor configuración de cada modelo en cada fold, con su accuracy
     en test (global y por clase) y F1 macro, y la configuración sugerida de cada modelo.

Guarda la gráfica graficas/configuracion_optima.png (accuracy por clase de cada modelo).

Uso (desde la carpeta de la práctica, con el corpus en corpus/):
  python3 launcher.py
"""
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')  # dibuja en fichero, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from corpus_loader import CorpusLoader
from json_corpus_reader import JsonCorpusReader
from sentiment_classifier import SentimentClassifier

CORPUS = 'corpus'
GRAFICAS = 'graficas'
N_REVIEWS = 50000        # None = todo el corpus (con un número, muestra con la proporción original)
FOLDS = 3
CLASES = ['negativas', 'neutras', 'positivas']
MODELOS = ['LogisticRegression', 'MultinomialNB', 'SGDClassifier']

NGRAMAS = [(1, 1), (1, 2)]
REDUCTORES = [None,                                              # sin reducción
              TruncatedSVD(n_components=300, random_state=42),   # 300 dimensiones (LSA)
              SelectKBest(chi2, k=5000)]                         # las 5000 características más útiles

# Peso de cada clase al entrenar (equivocarse con una review de esa clase cuesta ese peso):
#   None: todas igual.
#   'balanced': inversamente proporcional al número de reviews de la clase; con todo el corpus
#               sale aprox. positivas 1 : negativas 11 : neutras 26.
#   PESOS_AGRESIVOS: aproximadamente el doble que 'balanced' para neutras y negativas.
PESOS_AGRESIVOS = {'positivas': 1, 'negativas': 20, 'neutras': 50}
PESOS = [None, 'balanced', PESOS_AGRESIVOS]

# Lista de rejillas independientes. MultinomialNB va aparte porque no admite valores
# negativos, y TruncatedSVD los produce.
PARAM_GRID = [
    {'estimator': [LogisticRegression(max_iter=1000)],
     'reductor': REDUCTORES, 'ngram_range': NGRAMAS, 'class_weight': PESOS},
    {'estimator': [SGDClassifier(random_state=42)],
     'reductor': REDUCTORES, 'ngram_range': NGRAMAS, 'class_weight': PESOS},
    {'estimator': [MultinomialNB(), MultinomialNB(alpha=0.1)],
     'reductor': [None, SelectKBest(chi2, k=5000)], 'ngram_range': NGRAMAS, 'class_weight': PESOS},
]

# Color de cada clase en la gráfica (paleta categórica validada)
COLORES = {'negativas': '#2a78d6', 'neutras': '#eb6834', 'positivas': '#1baf7a'}


def describir(params):
    """
    Escribe una configuración de forma legible.

    Parámetros:
        params (dict): parámetros del SentimentClassifier ('estimator', 'reductor',
            'ngram_range' y 'class_weight').

    Devuelve:
        str: p. ej. "LogisticRegression | (1, 2)-gramas | SVD 300 | pesos balanced".
    """
    est = params['estimator']
    por_defecto = type(est)().get_params()
    cambiados = [f'{k}={v}' for k, v in est.get_params().items()
                 if k not in ('max_iter', 'random_state') and v != por_defecto[k]]
    modelo = type(est).__name__ + (f'({", ".join(cambiados)})' if cambiados else '')
    red = params['reductor']
    if red is None:
        reduccion = 'sin reducción'
    elif isinstance(red, TruncatedSVD):
        reduccion = f'SVD {red.n_components}'
    else:
        reduccion = f'chi2 k={red.k}'
    pesos = params.get('class_weight')
    if pesos is None:
        texto_pesos = 'sin pesos'
    elif pesos == 'balanced':
        texto_pesos = 'pesos balanced'
    else:
        texto_pesos = 'pesos agresivos'
    return f'{modelo} | {params["ngram_range"]}-gramas | {reduccion} | {texto_pesos}'


def evaluar(modelo, X_test, y_test):
    """
    Mide un modelo entrenado sobre el test de un fold.

    Parámetros:
        modelo (SentimentClassifier): modelo ya entrenado.
        X_test (list): reviews etiquetadas de test.
        y_test (list de str): clases reales.

    Devuelve:
        dict: 'accuracy', 'f1_macro' y la accuracy (recall) de cada clase.
    """
    pred = modelo.predict(X_test)
    resultado = {'accuracy': accuracy_score(y_test, pred),
                 'f1_macro': f1_score(y_test, pred, labels=CLASES, average='macro', zero_division=0)}
    resultado.update(zip(CLASES, recall_score(y_test, pred, labels=CLASES, average=None,
                                              zero_division=0)))
    return resultado


def procesar_fold(i, train_idx, test_idx, n_jobs):
    """
    Trabajo de un fold, que se ejecuta en su propio proceso: carga las reviews del fold,
    hace el GridSearchCV y, para cada modelo, reentrena su mejor configuración con todo el
    train del fold y la evalúa en el test del fold.

    Parámetros:
        i (int): número del fold (1, 2, 3...).
        train_idx (numpy.ndarray): posiciones de las reviews de entrenamiento en el CorpusLoader.
        test_idx (numpy.ndarray): posiciones de las reviews de test.
        n_jobs (int): núcleos que puede usar el GridSearchCV de este fold.

    Devuelve:
        dict: 'fold', 'segundos', 'n_configuraciones', 'top5' (las 5 mejores configuraciones
            con su F1 interno) y 'mejores' ({modelo: {'configuracion', 'f1_interno', 'test'}}).
    """
    inicio = time.time()
    # Cada proceso crea su propio reader y loader; con la misma semilla, el loader tiene las
    # mismas reviews en el mismo orden que en el proceso principal.
    loader = CorpusLoader(JsonCorpusReader(CORPUS), folds=FOLDS, n_total=N_REVIEWS)
    X_train, X_test = loader.documents(train_idx), loader.documents(test_idx)
    y_train, y_test = loader.labels(train_idx), loader.labels(test_idx)

    busqueda = GridSearchCV(SentimentClassifier(), PARAM_GRID, scoring='f1_macro', refit=False,
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                            n_jobs=n_jobs)
    busqueda.fit(X_train, y_train)
    cv = busqueda.cv_results_

    mejores = {}
    for modelo in MODELOS:
        candidatos = [k for k, p in enumerate(cv['params']) if type(p['estimator']).__name__ == modelo]
        k = max(candidatos, key=lambda c: cv['mean_test_score'][c])
        final = SentimentClassifier(**cv['params'][k]).fit(X_train, y_train)
        mejores[modelo] = {'configuracion': describir(cv['params'][k]),
                           'f1_interno': cv['mean_test_score'][k],
                           'test': evaluar(final, X_test, y_test)}
    top5 = [(cv['mean_test_score'][k], describir(cv['params'][k]))
            for k in np.argsort(-cv['mean_test_score'])[:5]]
    return {'fold': i, 'segundos': time.time() - inicio, 'n_configuraciones': len(cv['params']),
            'top5': top5, 'mejores': mejores}


def imprimir_fold(r):
    """
    Muestra por pantalla el resultado de un fold.

    Parámetros:
        r (dict): resultado de procesar_fold().
    """
    print(f'\n=== Fold {r["fold"]} ({r["n_configuraciones"]} configuraciones, {r["segundos"]:.0f} s) ===')
    print('  Las 5 mejores configuraciones (F1 macro interno):')
    for f1, configuracion in r['top5']:
        print(f'    {f1:.3f}  {configuracion}')
    print('  Mejor configuración de cada modelo, medida en el test del fold:')
    for modelo, m in r['mejores'].items():
        t = m['test']
        print(f'    {modelo:<20} accuracy {t["accuracy"]:.3f} | F1 macro {t["f1_macro"]:.3f} | '
              + ' | '.join(f'{c[:3]} {t[c]:.2f}' for c in CLASES)
              + f'\n      <- {m["configuracion"]}', flush=True)


def resumir(folds):
    """
    Para cada modelo: la configuración sugerida (la elegida en más folds; si hay empate, la de
    mejor F1 interno) y la media y desviación en test de la mejor configuración de cada fold.

    Parámetros:
        folds (list de dict): {modelo: {'configuracion', 'f1_interno', 'test'}} de cada fold.

    Devuelve:
        dict: por modelo, 'configuracion', 'veces' y, para accuracy, f1_macro y cada clase,
            su media y su desviación ('<métrica>_std').
    """
    resumen = {}
    for modelo in MODELOS:
        elegidas = Counter(f[modelo]['configuracion'] for f in folds)
        f1_medio = {c: np.mean([f[modelo]['f1_interno'] for f in folds
                                if f[modelo]['configuracion'] == c]) for c in elegidas}
        configuracion = max(elegidas, key=lambda c: (elegidas[c], f1_medio[c]))
        resumen[modelo] = {'configuracion': configuracion, 'veces': elegidas[configuracion]}
        for metrica in ['accuracy', 'f1_macro'] + CLASES:
            valores = [f[modelo]['test'][metrica] for f in folds]
            resumen[modelo][metrica] = np.mean(valores)
            resumen[modelo][f'{metrica}_std'] = np.std(valores)
    return resumen


def grafica(resumen, n_reviews, ruta):
    """
    Gráfica de barras: accuracy de cada clase (recall) en test, media ± desviación de los
    folds, de la configuración óptima de cada modelo. Debajo de cada modelo, su configuración
    y su accuracy global y F1 macro.

    Parámetros:
        resumen (dict): resultado de resumir().
        n_reviews (int): número de reviews usadas (para el título).
        ruta (str): fichero .png donde se guarda.
    """
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    ancho = 0.26
    posiciones = np.arange(len(MODELOS))
    for j, clase in enumerate(CLASES):
        medias = [resumen[m][clase] for m in MODELOS]
        desv = [resumen[m][f'{clase}_std'] for m in MODELOS]
        xs = posiciones + (j - 1) * (ancho + 0.015)
        ax.bar(xs, medias, ancho, yerr=desv, color=COLORES[clase], label=clase,
               error_kw={'ecolor': '#52514e', 'elinewidth': 1, 'capsize': 3},
               edgecolor='#fcfcfb', linewidth=2)
        for xi, v, d in zip(xs, medias, desv):
            ax.annotate(f'{v:.2f}', (xi, v + d), xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=7.5, color='#0b0b0b')

    etiquetas = []
    for m in MODELOS:
        partes = resumen[m]['configuracion'].split(' | ')
        extra = partes[0][len(m):]   # parámetros cambiados del modelo, si los hay
        r = resumen[m]
        etiquetas.append('\n'.join([m] + partes[1:] + ([extra] if extra else [])
                                   + [f'accuracy {r["accuracy"]:.3f} | F1 macro {r["f1_macro"]:.3f}',
                                      f'(elegida en {r["veces"]} de {FOLDS} folds)']))
    ax.set_xticks(posiciones, etiquetas, fontsize=7, color='#52514e')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Accuracy de la clase en test (recall)', color='#52514e')
    ax.set_title(f'Configuración óptima de cada modelo: accuracy por clase ({n_reviews:,} reviews)'
                 .replace(',', '.'), color='#0b0b0b', loc='left', fontsize=11, pad=24)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0),
              borderaxespad=0.3)
    ax.grid(True, axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='#898781', labelsize=8)
    for lado in ['top', 'right']:
        ax.spines[lado].set_visible(False)
    for lado in ['left', 'bottom']:
        ax.spines[lado].set_color('#c3c2b7')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main():
    """
    Ejecuta el experimento 2 completo: lanza un proceso por fold, muestra los resultados por
    pantalla y guarda la gráfica en graficas/.

    Parámetros:
        (ninguno: usa las constantes CORPUS, GRAFICAS, N_REVIEWS, FOLDS y PARAM_GRID)
    """
    inicio = time.time()
    reader = JsonCorpusReader(CORPUS)                               # paso 1
    loader = CorpusLoader(reader, folds=FOLDS, n_total=N_REVIEWS)   # paso 2
    particiones = list(loader.splitter.split(loader.fileids(), loader.labels()))
    n_jobs = max(1, (os.cpu_count() or 1) // FOLDS)                 # núcleos para cada fold
    print(f'{len(loader.fileids())} reviews, {FOLDS} folds en paralelo, {n_jobs} núcleos por fold')

    resultados = []
    with ProcessPoolExecutor(max_workers=FOLDS) as procesos:        # pasos 3 y 4
        tareas = [procesos.submit(procesar_fold, i, tr, te, n_jobs)
                  for i, (tr, te) in enumerate(particiones, 1)]
        for tarea in as_completed(tareas):
            resultados.append(tarea.result())
            imprimir_fold(resultados[-1])

    resumen = resumir([r['mejores'] for r in sorted(resultados, key=lambda r: r['fold'])])
    print(f'\n=== CONFIGURACIÓN SUGERIDA DE CADA MODELO (test, media ± desviación de {FOLDS} folds) ===')
    for modelo, r in resumen.items():                               # paso 5
        print(f'  {modelo:<20} accuracy {r["accuracy"]:.3f} ± {r["accuracy_std"]:.3f} | '
              f'F1 macro {r["f1_macro"]:.3f} ± {r["f1_macro_std"]:.3f} | '
              + ' | '.join(f'{c[:3]} {r[c]:.2f} ± {r[c + "_std"]:.2f}' for c in CLASES)
              + f'\n    <- {r["configuracion"]} (elegida en {r["veces"]} de {FOLDS} folds)')

    os.makedirs(GRAFICAS, exist_ok=True)
    grafica(resumen, len(loader.fileids()), os.path.join(GRAFICAS, 'configuracion_optima.png'))
    print(f'\nGráfica guardada en {GRAFICAS}/configuracion_optima.png ({time.time() - inicio:.0f} s)')


if __name__ == '__main__':
    main()
