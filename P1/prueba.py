from AmazonCorpusReader import AmazonCorpusReader

# prueba 1: ver si se eliminan las etiquetas HTML y los emojis
# también se añade tratamiento de signos de interrogación

r = AmazonCorpusReader('.')
print(r._limpiar("Greaaaat<br />product!!! &amp; I paid 1000 😀😀"))

# prueba 2: ver si los 
print(next(r.paras()))
print(next(r.sents()))
print(next(r.words()))

# prueba 3: ver si se filtran las stopwords
print(next(r.sents()))
print(next(r.sents(sin_stopwords=True)))

print(r.describe(categories='negativas'))