from nltk import *

user_corpus = input("Ingrese el corpus de texto: ")
tokenized = sent_tokenize(user_corpus)
i=1
for j in tokenized:
    print(f"{i}: "+j)
    i+=1

words_token = [word_tokenize(i) for i in tokenized ]
i=1
total=0
for j in words_token:
    total+=len(j)
    print(f"{i}: {j} número de tokens:{len(j)}")

print(f"Número total de oraciones: {len(tokenized)}")
print(f"Número total de tokens: {total}")
print(f"Tokens por oracion medio: {total/len(tokenized)}")