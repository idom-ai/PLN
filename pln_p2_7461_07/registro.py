"""
registro.py - Guarda en un fichero de log todo lo que un experimento muestra por pantalla.

Uso:
    with Registro('launcher_corpus') as reg:
        print('...')          # sale por pantalla y en logs/launcher_corpus_<fecha>/salida.log
        reg.carpeta           # carpeta del log, para guardar ahí otros ficheros (CSV...)
"""
import os
import sys
import time


class _Tee(object):
    """
    Flujo de salida que escribe a la vez en varios flujos (la pantalla y el fichero de log).
    """

    def __init__(self, *flujos):
        """
        Parámetros:
            *flujos: flujos de texto abiertos (p. ej. sys.stdout y un fichero).
        """
        self.flujos = flujos

    def write(self, texto):
        """
        Escribe el texto en todos los flujos.

        Parámetros:
            texto (str): texto a escribir.

        Devuelve:
            int: número de caracteres escritos.
        """
        for flujo in self.flujos:
            flujo.write(texto)
        return len(texto)

    def flush(self):
        """
        Vacía el búfer de todos los flujos (para que el log esté al día aunque el
        experimento se interrumpa).
        """
        for flujo in self.flujos:
            flujo.flush()


class Registro(object):
    """
    Context manager que, mientras está activo, copia la salida estándar a
    <carpeta_logs>/<nombre>_<AAAAMMDD_HHMMSS>/salida.log.
    """

    def __init__(self, nombre, carpeta_logs='logs'):
        """
        Parámetros:
            nombre (str): nombre del experimento (p. ej. 'launcher_corpus').
            carpeta_logs (str): carpeta donde se crean las carpetas de log. Por defecto, 'logs'.
        """
        self.carpeta = os.path.join(carpeta_logs, f'{nombre}_{time.strftime("%Y%m%d_%H%M%S")}')

    def __enter__(self):
        """
        Crea la carpeta del log y empieza a copiar la salida estándar al fichero.

        Devuelve:
            Registro: el propio objeto (su atributo carpeta es la carpeta del log).
        """
        os.makedirs(self.carpeta, exist_ok=True)
        self._fichero = open(os.path.join(self.carpeta, 'salida.log'), 'w', encoding='utf-8')
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._fichero)
        print(f'[log] {os.path.join(self.carpeta, "salida.log")} '
              f'(python {" ".join(sys.argv)})', flush=True)
        return self

    def __exit__(self, tipo, valor, traza):
        """
        Deja de copiar la salida y cierra el fichero. Si el experimento ha fallado, el
        error también queda en el log.

        Parámetros:
            tipo, valor, traza: excepción que ha terminado el bloque with (o None).

        Devuelve:
            bool: False, para que la excepción (si la hay) se siga propagando.
        """
        if tipo is not None:
            import traceback
            traceback.print_exception(tipo, valor, traza, file=self._fichero)
        sys.stdout = self._stdout
        self._fichero.close()
        return False
