# 2: Iris Dataset
## Clasificadores Binarios
Se utilizó el dataframe que contiene $150$ entradas en total. Este fue dividido en tres partes:
* ***iris_setosa***: Valores de Iris Setosa fueron marcados como $1$. Para Iris Versicolor e Iris Virginica fueron marcados como $0$.
* ***iris_versicolor***:  Valores de Iris Versicolor fueron marcados como $1$. Para Iris Setosa e Iris Virginica fueron marcados como $0$.
* ***iris_virginica***:  Valores de Iris Virginica fueron marcados como $1$. Para Iris Versicolor e Iris Setosa fueron marcados como $0$.

Luego, cada una de ellas fue a su vez separado en conjuntos de entrenamiento y prueba con una división de $80%$ y $20%$ respectivamente.

Así, para cada especie, tenemos un conjunto de entrenamiento con $120$ valores y uno de prueba con $30$ valores.

Se utilizó un umbral de $0.5$ para determinar el valor arrojado por la capa de salida de la red neuronal calificaba como $0$ o $1$. Es decir, si el valor arrojado estaba en el rango $[0, 0.5)$, se asume como $0$, mientras que si estaba entre $[0.5, 1]$ se asume como $1$.

### Única Neurona:
Se utilizaron los siguientes valores:
* Tasas de aprendizaje: 1e-06, 1e-05, 0.001, 0.01
* Cantidad máxima de iteraciones: 100000, 10000, 5000, 1000

En total, se crearon siete distintos casos por cada una de las especies del dataframe.

Para calcular el valor mínimo, máximo y promedio de errores, se evaluó cada uno de los casos cinco veces. Es decir, se corrió el algoritmo cinco veces para cada uno de ellos y luego se vió la iteración donde más errores se produjeron, donde menos hubo y el promedio de estas cinco.

En el caso de la tasa de aprendizaje $0.01$, se decidió hacerla con tres distintas cantidades de iteraciones ya que al momento de ejecutarla con $10000$ y ver los datos, nos pudimos dar cuenta de la diferencia de la cantidad de errores entre los casos evaluados previamente y este, por lo que se decidió ver si era posible llegar a mejores resultados con menos iteraciones.

|Especie|Learning  Rate|Max.  Iter|Total  Errores|Min.E.|Max. E.|AVG E.|Total  Falsos  -| Total  Falsos  +
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**Iris Setosa**|**1e-06**|**100000**|78|9|27|15.6|42|36|
|**I.Versicolor**|**1e-06**|**100000**|83|13|27|16.6|26|57|
|**I.Virginica**|**1e-06**|**100000**|81|10|19|16.2|20|61|
||
|**Iris Setosa**|**1e-05**|**100000**|63|10|19|12.6|44|19|
|**I.Versicolor**|**1e-05**|**100000**|70|10|20|14|43|27|
|**I.Virginica**|**1e-05**|**100000**|73|12|19|14.6|24|59|
||
|**Iris Setosa**|**0.001**|**10000**|55|1|20|11|12|43|
|**I.Versicolor**|**0.001**|**10000**|65|10|19|13|21|44|
|**I.Virginica**|**0.001**|**10000**|63|4|20|10.75|4|59|
||
|**Iris Setosa**|**0.01**|**10000**|4|0|2|0.8|2|2|
|**I.Versicolor**|**0.01**|**10000**|63|10|20|12.6|43|20|
|**I.Virginica**|**0.01**|**10000**|30|2|11|6|22|8|
||
|**Iris Setosa**|**0.1**|**10000**|4|4|0|0.8|4|0|
|**I.Versicolor**|**0.1**|**10000**|24|7|10|4.8|5|19|
|**I.Virginica**|**0.1**|**10000**|11|1|10|2.2|10|1|
||
|**Iris Setosa**|**0.1**|**5000**|22|0|15|4.4|22|0|
|**I.Versicolor**|**0.1**|**5000**|40|0|8|8|31|9|
|**I.Virginica**|**0.1**|**5000**|15|1|5|3|10|5|
||
|**Iris Setosa**|**0.1**|**1000**|6|0|3|1.2|5|1|
|**I.Versicolor**|**0.1**|**1000**|57|10|13|11.4|51|6|
|**I.Virginica**|**0.1**|**1000**|8|1|3|1.6|2|5|

Es claro por estos resultados que con la tasa de aprendizaje entre $0.01$ y $0.1$ se logra obtener menos cantidad de errores en cada una de las categorías.

(hablar hablar hablar)

### Una Capa Oculta
Al ver los resultados obtenidos previamente, se decidió optar por sólo dos posibles tasas de aprendizaje: $0.01$ y $0.1$, y conocer los valores resultantes al usar tres cantidades máximas de iteraciones: $1000$, $5000$ y $10000$.
Debido al tiempo que toma experimentar con mayor cantidad de neuronas en las capas ocultas y los límites que se tenían, se decidió seguir los consejos dados en clase de experimentar hasta con un máximo de cinco neuronas en la capa oculta. Así, se varió la cantidad en el intervalo [1, 5], resultando finalmente $30$ distintos casos evaluados que se pueden observar a continuación:

|Especie |Learning  Rate|Neuronas  en  Capa  Oculta|Max.  Iter|Total  Errores|Min.  E.|Máx  E.|AVG  E.|Total  Falsos  - |Total  Falsos  +|Max.  Iter|Total  Errores|Min.  E.|Máx  E.|AVG  E.|Total  Falsos  - |Total  Falsos  +|Max.  Iter|Total  Errores|Min.  E.|Máx  E.|AVG  E.|Total  Falsos  - |Total  Falsos  +|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**Iris Setosa**|**0.01** |**1** |**1000** |63 |10 |16 |12.6 |33 |30 |**5000** |50 |1 |13 |10 |50 |0 |**10000** |20 |0 |10 |4 |20 |0 |
|**I.Versicolor**|**0.01** |**1** |**1000** |65 |10 |19 |13 |30 |35 |**5000** |57 |10 |11 |11.4 |57 |0 |**10000** |59 |11 |13 |11.8 |54 |5 |
|**I.Virginica**|**0.01** |**1** |**1000** |58 |10 |17 |11.6 |41 |17 |**5000** |53 |10 |12 |10.6 |53 |0 |**10000** |57 |10 |13 |11.4 |57 |0 |
||
|**Iris Setosa**|**0.01** |**2** |**1000** |71 |10 |20 |74.2 |32 |39 |**5000** |47 |0 |13 |9.4 |47 |0 |**10000** |34 |1 |11 |6.8 |34 |0 |
|**I.Versicolor**|**0.01** |**2** |**1000** |80 |12 |19 |16 |41 |39 |**5000** |56|10 |14 |11.2 |56 |0 |**10000** |59 |10 |13 |11.8 |59 |0 |
|**I.Virginica**|**0.01** |**2** |**1000** |65 |12 |16 |13 |49 |16 |**5000** |47 |3 |11 |9.4 |47 |0 |**10000** |30 |2 |15 |6 |24 |6 |
||
|**Iris Setosa**|**0.01** |**3** |**1000** |44 |0 |12 |8.8 |44 |0 |**5000** |25 |0 |12 |5 |25 |0 |**10000** |12 |0 |12 |2.4 |12 |0 |
|**I.Versicolor**|**0.01** |**3** |**1000** |72 |10 |23 |14.4 |42 |30 |**5000** |52 |10 |12 |10.4 |52|0 |**10000** |58 |10 |13 |11.6 |58 |0 |
|**I.Virginica**|**0.01** |**3** |**1000** |65 |11 |17 |13 |48 |17 |**5000** |59 |10 |15 |11.8 |59 |0 |**10000** |25 |1 |11 |5 |22 |3 |
||
|**Iris Setosa**|**0.01** |**4** |**1000** |44 |0 |20 |8.8 |24 |20 |**5000** |22 |0 |11 |4.4 |22 |0 |**10000** |10 |0 |10 |2 |10 |0 |
|**I.Versicolor**|**0.01** |**4** |**1000** |61 |11 |14 |12.2 |61 |0 |**5000** |52 |7 |14 |10.4 |52 |0 |**10000** |64 |10 |15 |12.8 |64 |0 |
|**I.Virginica**|**0.01** |**4** |**1000** |55 |8 |12 |11 |55 |0 |**5000** |57 |10 |13 |11.4 |57 |0 |**10000** |39 |0 |15 |7.8 |39 |0 |
||
|**Iris Setosa**|**0.01** |**5** |**1000** |27 |0 |15 |5.4 |27 |0 |**5000** |17 |0 |15 |3.4 |17 |0 |**10000** |11 |0 |11 |2.2 |11 |0 |
|**I.Versicolor**|**0.01** |**5** |**1000** |66 |10 |15 |13.2 |50 |16 |**5000** |54|10 |11 |10.8 |54 |0 |**10000** |58 |10 |15 |11.6 |58 |0 |
|**I.Virginica**|**0.01** |**5** |**1000** |39 |3 |10 |7.8 |38 |1 |**5000** |37 |0 |12 |7.4 |37 |0 |**10000** |14 |0 |5 |2.8 |8 |6 |
||
|**Iris Setosa**|**0.1** |**1** |**1000** |16 |0 |13 |3.2 |16 |0 |**5000** |48 |10 |14 |9.6 |48 |0 |**10000** |22 |0 |12 |4.4 |22 |0 |
|**I.Versicolor**|**0.1** |**1** |**1000** |56 |10 |13 |11.2 |56 |0 |**5000** |57|10 |13 |11.4 |57 |0 |**10000** |62 |11 |14 |12.4 |62 |0 |
|**I.Virginica**|**0.1** |**1** |**1000** |47 |10 |14 |9.4 |47 |0 |**5000** |43 |1 |15 |8.6 |43 |0 |**10000** |51 |1 |14 |10.2 |50 |1 |
||
|**Iris Setosa**|**0.1** |**2** |**1000** |33 |10 |12 |6.6 |33 |0 |**5000** |3 |0 |3 |0.6 |3 |0 |**10000** |0 |0 |0 |0 |0 |0 |
|**I.Versicolor**|**0.1** |**2** |**1000** |60 |10 |16 |12 |60 |0 |**5000** |53 |10.6 |8 |13 |52 |1 |**10000** |47 |4 |13 |9.4 |46 |1 |
|**I.Virginica**|**0.1** |**2** |**1000** |41 |2 |13 |8.2 |40 |1 |**5000** |27 |1 |11 |5.4 |24 |3 |**10000** |27 |0 |13 |5.4 |26 |1 |
||
|**Iris Setosa**|**0.1** |**3** |**1000** |36 |0 |14 |7.2 |14 |0 |**5000** |15 |0 |13 |3 |15 |0 |**10000** |0 |0 |0 |0 |0 |0 |
|**I.Versicolor**|**0.1** |**3** |**1000** |55 |10 |12 |11 |55 |0 |**5000** |44 |0 |16 |8.8 |39 |5 |**10000** |20 |0 |11 |4 |20 |0 |
|**I.Virginica**|**0.1** |**3** |**1000** |46 |4 |14 |9.2 |46 |0 |**5000** |24 |0 |12 |4.8 |23 |1 |**10000** |8 |1 |2 |1.6 |5 |3 |
||
|**Iris Setosa**|**0.1** |**4** |**1000** |14 |0 |13 |2.8 |14 |0 |**5000** |0 |0 |0 |0 |0 |0 |**10000** |0 |0 |0 |0 |0 |0 |
|**I.Versicolor**|**0.1** |**4** |**1000** |53 |10 |11 |10.6 |53 |0 |**5000** |52 |1 |16 |10.4 |52 |0 |**10000** |30 |0 |13 |6 |29 |1 |
|**I.Virginica**|**0.1** |**4** |**1000** |27 |1 |11 |5.4 |25 |2 |**5000** |6 |0 |3 |5.2 |3 |3 |**10000** |8 |0 |4 |1.6 |7 |1 |
||
|**Iris Setosa**|**0.1** |**5** |**1000** |11 |0 |10 |2.2 |11 |0 |**5000** |0 |0 |0 |0 |0 |0 |**10000** |20 |0 |11 |4 |20 |0 |
|**I.Versicolor**|**0.1** |**5** |**1000** |60 |11 |13 |12 |56 |4 |**5000** |17 |0 |10 |3.4 |17 |0 |**10000** |28 |0 |12 |5.6 |28 |0 |
|**I.Virginica**|**0.1** |**5** |**1000** |33 |2 |13 |6.6 |31 |2 |**5000** |21 |1 |14 |4.2 |17 |4 |**10000** |6 |0 |3 |1.2 |3 |3 |

(hablar sobre qué representan estos valores y blablabla)

### Dos Capas Ocultas
Se utilizaron los siguientes valores:
* Cantidad máxima de iteraciones: 2000, 5000, 7000
* Tasas de aprendizaje: 0.01, 0.1
* Cantidad de neuronas en las capas ocultas: (1, 3), (3, 5), (5, 3), (3, 3), (5, 5)
