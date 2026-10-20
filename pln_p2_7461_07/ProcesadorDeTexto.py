from typing import List
import json
import os

class ProcesadorDeTexto(): 
    
    def __init__(self, file_name: str, categories: List[str]):
        
        self.__file_name = file_name
        self.__categories = categories
        self.__vistos = set()
        self.__devolver = []
        self.__num_positivas, self.__num_neutras, self.__num_negativas = 0, 0, 0
        
    def preprocesar_texto(self):
        
        with open(self.__file_name + '.jsonl', 'r', encoding='utf-8') as fp:
            for line in fp:
                # un diccionario para cada nueva review
                review = dict()
                # pasar cada linea de str a diccionario
                line = json.loads(line)
                
                # para cada parámetro de la lista recibida, quedárnoslo
                for parametro in self.__categories: 
                    review[parametro] = line[parametro]  
                
                # guardar los que ya hemos visto sin tener en cuenta mayúsculas
                # esto elimina comentarios repetidos para evitar el desbalance de datos     
                clave = (review['user_id'], review['title'].lower(), review['text'].lower())
                if clave in self.__vistos:
                    continue
                self.__vistos.add(clave)
                
                # eliminar los ids una vez ya los hemos usado para detectar duplicados
                del review['user_id']
                
                self.__devolver.append(review)
        
        for elemento in self.__devolver:
            
            line = elemento
            
            if line['rating'] > 4.5:
                with open(f'positivas/positivas_{self.__num_positivas + 1}.json', 'w', encoding='utf-8') as f:
                    json.dump(line, f)
                    self.__num_positivas +=1

            elif (line['rating'] < 4.5) and (line['rating'] > 1.5):
                with open(f'neutras/neutras_{self.__num_neutras + 1}.json', 'w', encoding='utf-8') as f:
                    json.dump(line, f)
                    self.__num_neutras +=1

            else:

                with open(f'negativas/negativas_{self.__num_negativas + 1}.json', 'w', encoding='utf-8') as f:
                    json.dump(line, f)
                    self.__num_negativas +=1
                        
                        
def main():
    
    os.makedirs("positivas", exist_ok=True)
    os.makedirs("neutras", exist_ok=True)
    os.makedirs("negativas", exist_ok=True)
    
    procesador = ProcesadorDeTexto(
        file_name='Gift_Cards',
        categories=['rating', 'title', 'text', 'parent_asin', 'user_id']
        )
    
    procesador.preprocesar_texto()

if __name__ == '__main__':
    main()