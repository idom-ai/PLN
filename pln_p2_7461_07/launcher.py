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
     La mejor configuración se elige (por defecto por F1 macro; con --metrica accuracy, por
     accuracy) con validación cruzada interna sobre el train del fold; el test del fold solo
     se usa para medirla.
  5. Muestra por pantalla la mejor configuración de cada modelo en cada fold, con su accuracy
     en test (global y por clase) y F1 macro, y la configuración sugerida de cada modelo.

Todo se guarda en logs/launcher_<corpus>_<fecha>/:
  salida.log                  lo mismo que sale por pantalla
  grid_fold<i>.csv            TODAS las configuraciones probadas en el fold i, con su accuracy,
                              F1 macro y recall de cada clase (media y desviación de la CV interna)
  todas_configuraciones.csv   cada configuración con la media de esas métricas en los folds,
                              ordenada por la métrica de selección (para elegir el mejor modelo)
  mejores_test.csv            mejor configuración de cada modelo en cada fold, medida en test
y la gráfica en graficas/configuracion_optima_<corpus>.png (accuracy por clase de cada modelo).

Uso (desde la carpeta de la práctica):
  python3 launcher.py                                    # corpus/, selección por F1 macro
  python3 launcher.py --corpus corpus_pos5_neu234        # otro reparto de clases
  python3 launcher.py --metrica accuracy --n-reviews 0   # selección por accuracy, todo el corpus
"""
import argparse
import csv
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
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from corpus_loader import CorpusLoader
from json_corpus_reader import JsonCorpusReader
from registro import Registro
from sentiment_classifier import SentimentClassifier

CORPUS = 'corpus'
GRAFICAS = 'graficas'
N_REVIEWS = 50000        # None = todo el corpus (con un número, muestra con la proporción original)
FOLDS = 3
CLASES = ['negativas', 'neutras', 'positivas']
MODELOS = ['LogisticRegression', 'MultinomialNB', 'SGDClassifier']

# Métricas que GridSearchCV calcula para TODAS las configuraciones (se guardan en los CSV).
# Calcularlas cuesta casi nada: los modelos se entrenan una vez y se miden con todas.
METRICAS = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
METRICAS.update({f'recall_{c}': make_scorer(recall_score, labels=[c], average='macro',
                                            zero_division=0) for c in CLASES})

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


def procesar_fold(i, train_idx, test_idx, n_jobs, corpus, n_reviews, metrica):
    """
    Trabajo de un fold, que se ejecuta en su propio proceso: carga las reviews del fold,
    hace el GridSearchCV y, para cada modelo, reentrena su mejor configuración con todo el
    train del fold y la evalúa en el test del fold.

    Parámetros:
        i (int): número del fold (1, 2, 3...).
        train_idx (numpy.ndarray): posiciones de las reviews de entrenamiento en el CorpusLoader.
        test_idx (numpy.ndarray): posiciones de las reviews de test.
        n_jobs (int): núcleos que puede usar el GridSearchCV de este fold.
        corpus (str): carpeta del corpus.
        n_reviews (int | None): tamaño de la muestra (None: todo el corpus).
        metrica (str): métrica con la que se elige la mejor configuración ('f1_macro' o 'accuracy').

    Devuelve:
        dict: 'fold', 'segundos', 'n_configuraciones', 'top5' (las 5 mejores configuraciones
            con su métrica interna), 'mejores' ({modelo: {'configuracion', 'f1_interno', 'test'}})
            y 'filas' (una fila por configuración con todas las métricas, para los CSV).
    """
    inicio = time.time()
    # Cada proceso crea su propio reader y loader; con la misma semilla, el loader tiene las
    # mismas reviews en el mismo orden que en el proceso principal.
    loader = CorpusLoader(JsonCorpusReader(corpus), folds=FOLDS, n_total=n_reviews)
    X_train, X_test = loader.documents(train_idx), loader.documents(test_idx)
    y_train, y_test = loader.labels(train_idx), loader.labels(test_idx)

    busqueda = GridSearchCV(SentimentClassifier(), PARAM_GRID, scoring=METRICAS, refit=False,
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                            n_jobs=n_jobs)
    busqueda.fit(X_train, y_train)
    cv = busqueda.cv_results_
    puntuacion = cv[f'mean_test_{metrica}']

    mejores = {}
    for modelo in MODELOS:
        candidatos = [k for k, p in enumerate(cv['params']) if type(p['estimator']).__name__ == modelo]
        k = max(candidatos, key=lambda c: puntuacion[c])
        final = SentimentClassifier(**cv['params'][k]).fit(X_train, y_train)
        mejores[modelo] = {'configuracion': describir(cv['params'][k]),
                           'f1_interno': puntuacion[k],
                           'test': evaluar(final, X_test, y_test)}
    top5 = [(puntuacion[k], describir(cv['params'][k])) for k in np.argsort(-puntuacion)[:5]]

    filas = []
    for k, params in enumerate(cv['params']):
        fila = {'fold': i, 'modelo': type(params['estimator']).__name__,
                'configuracion': describir(params)}
        for nombre in METRICAS:
            fila[f'{nombre}_media'] = round(float(cv[f'mean_test_{nombre}'][k]), 4)
            fila[f'{nombre}_std'] = round(float(cv[f'std_test_{nombre}'][k]), 4)
        fila['segundos_entrenar'] = round(float(cv['mean_fit_time'][k]), 1)
        filas.append(fila)
    return {'fold': i, 'segundos': time.time() - inicio, 'n_configuraciones': len(cv['params']),
            'top5': top5, 'mejores': mejores, 'filas': filas}


def guardar_csv(filas, ruta):
    """
    Guarda una lista de diccionarios (todos con las mismas claves) en un fichero CSV.

    Parámetros:
        filas (list de dict): una fila por diccionario; las claves son las columnas.
        ruta (str): fichero .csv donde se guarda.
    """
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def agregar_configuraciones(resultados, metrica):
    """
    Junta las configuraciones de todos los folds: para cada una, la media en los folds de
    cada métrica de la validación cruzada interna.

    Parámetros:
        resultados (list de dict): resultados de procesar_fold().
        metrica (str): métrica por la que se ordenan (de mayor a menor).

    Devuelve:
        list de dict: una fila por configuración, con 'modelo', 'configuracion' y la media de
            cada métrica ('<métrica>'), ordenadas por la métrica indicada.
    """
    por_configuracion = {}
    for r in resultados:
        for fila in r['filas']:
            por_configuracion.setdefault((fila['modelo'], fila['configuracion']), []).append(fila)
    tabla = []
    for (modelo, configuracion), filas in por_configuracion.items():
        fila = {'modelo': modelo, 'configuracion': configuracion, 'folds': len(filas)}
        for nombre in METRICAS:
            fila[nombre] = round(float(np.mean([f[f'{nombre}_media'] for f in filas])), 4)
        fila['segundos_entrenar'] = round(float(np.mean([f['segundos_entrenar'] for f in filas])), 1)
        tabla.append(fila)
    return sorted(tabla, key=lambda f: -f[metrica])


def imprimir_fold(r, metrica):
    """
    Muestra por pantalla el resultado de un fold.

    Parámetros:
        r (dict): resultado de procesar_fold().
        metrica (str): métrica con la que se ha elegido la mejor configuración.
    """
    print(f'\n=== Fold {r["fold"]} ({r["n_configuraciones"]} configuraciones, {r["segundos"]:.0f} s) ===')
    print(f'  Las 5 mejores configuraciones ({metrica} interno):')
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
    pantalla, guarda el log y los CSV en logs/ y la gráfica en graficas/.

    Parámetros (de línea de comandos):
        --corpus: carpeta del corpus. Por defecto, corpus.
        --n-reviews: tamaño de la muestra (con la proporción original de clases); 0 = todo
            el corpus. Por defecto, N_REVIEWS.
        --metrica: métrica con la que se elige la mejor configuración: f1_macro (por defecto)
            o accuracy.
    """
    parser = argparse.ArgumentParser(description='Grid search de los 3 modelos con validación cruzada.')
    parser.add_argument('--corpus', default=CORPUS, help=f'carpeta del corpus (por defecto, {CORPUS})')
    parser.add_argument('--n-reviews', type=int, default=N_REVIEWS,
                        help=f'reviews de la muestra; 0 = todas (por defecto, {N_REVIEWS})')
    parser.add_argument('--metrica', choices=['f1_macro', 'accuracy'], default='f1_macro',
                        help='métrica para elegir la mejor configuración (por defecto, f1_macro)')
    args = parser.parse_args()
    n_reviews = args.n_reviews or None
    nombre = os.path.basename(os.path.normpath(args.corpus))

    with Registro(f'launcher_{nombre}_{args.metrica}') as registro:
        inicio = time.time()
        reader = JsonCorpusReader(args.corpus)                          # paso 1
        loader = CorpusLoader(reader, folds=FOLDS, n_total=n_reviews)   # paso 2
        particiones = list(loader.splitter.split(loader.fileids(), loader.labels()))
        n_jobs = max(1, (os.cpu_count() or 1) // FOLDS)                 # núcleos para cada fold
        print(f'Corpus {args.corpus}: {len(loader.fileids())} reviews '
              f'({", ".join(f"{c} {loader.labels().count(c)}" for c in CLASES)}), '
              f'selección por {args.metrica}, {FOLDS} folds en paralelo, {n_jobs} núcleos por fold',
              flush=True)

        resultados = []
        with ProcessPoolExecutor(max_workers=FOLDS) as procesos:        # pasos 3 y 4
            tareas = [procesos.submit(procesar_fold, i, tr, te, n_jobs, args.corpus, n_reviews,
                                      args.metrica)
                      for i, (tr, te) in enumerate(particiones, 1)]
            for tarea in as_completed(tareas):
                resultados.append(tarea.result())
                imprimir_fold(resultados[-1], args.metrica)
                guardar_csv(resultados[-1]['filas'],
                            os.path.join(registro.carpeta, f'grid_fold{resultados[-1]["fold"]}.csv'))

        resultados.sort(key=lambda r: r['fold'])
        tabla = agregar_configuraciones(resultados, args.metrica)
        guardar_csv(tabla, os.path.join(registro.carpeta, 'todas_configuraciones.csv'))
        print(f'\n=== LAS 10 MEJORES CONFIGURACIONES ({args.metrica} interno, media de {FOLDS} folds;'
              f' el resto en todas_configuraciones.csv) ===')
        for fila in tabla[:10]:
            print(f'  acc {fila["accuracy"]:.3f} | F1 {fila["f1_macro"]:.3f} | '
                  + ' | '.join(f'{c[:3]} {fila["recall_" + c]:.2f}' for c in CLASES)
                  + f'  {fila["configuracion"]}')

        mejores_test = [{'fold': r['fold'], 'modelo': modelo, 'configuracion': m['configuracion'],
                         f'{args.metrica}_interno': round(float(m['f1_interno']), 4),
                         **{k: round(float(v), 4) for k, v in m['test'].items()}}
                        for r in resultados for modelo, m in r['mejores'].items()]
        guardar_csv(mejores_test, os.path.join(registro.carpeta, 'mejores_test.csv'))

        resumen = resumir([r['mejores'] for r in resultados])
        print(f'\n=== CONFIGURACIÓN SUGERIDA DE CADA MODELO (test, media ± desviación de {FOLDS} folds) ===')
        for modelo, r in resumen.items():                               # paso 5
            print(f'  {modelo:<20} accuracy {r["accuracy"]:.3f} ± {r["accuracy_std"]:.3f} | '
                  f'F1 macro {r["f1_macro"]:.3f} ± {r["f1_macro_std"]:.3f} | '
                  + ' | '.join(f'{c[:3]} {r[c]:.2f} ± {r[c + "_std"]:.2f}' for c in CLASES)
                  + f'\n    <- {r["configuracion"]} (elegida en {r["veces"]} de {FOLDS} folds)')

        os.makedirs(GRAFICAS, exist_ok=True)
        ruta_grafica = os.path.join(GRAFICAS, f'configuracion_optima_{nombre}_{args.metrica}.png')
        grafica(resumen, len(loader.fileids()), ruta_grafica)
        print(f'\nGráfica guardada en {ruta_grafica}; CSV en {registro.carpeta}/ '
              f'({time.time() - inicio:.0f} s)')


if __name__ == '__main__':
    main()
