# Prediccion Sobrevivientes Titanic con Machine Learning (Proyecto)

## Para comenzar con este proyecto, importamos las librerias que utilizaremos para hacer la predicción.

![Alt Text](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/1.JPG)

## Despues cargamos nuestros archivos de entrenamiento y de prueba
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/2.JPG)

### Al imprimir el contenido de los archivos obtenemos lo siguiente, si prestamos atención, notaremos que existen filas que no tienen datos o columnas que ni siquiera son numericas, ademas de que hay algunas que no nos son de utilidad.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/3.JPG)

### Si vemos de que tipo son nuestros datos, notaremos que algunos son objetos, pero para poder hacer la predicción necesitamos valores númericos, más adelante acomodaremos esos datos.

![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/4.JPG)

### Revisamos cuantos datos hay vacios y los sumamos, podemos ver que hay muchos en Age y Cabin, la edad parece ser importante, por lo que los llenaremos, por otro lado Cabin no parece ser de utilidad, ademas que le faltan demasiados datos, por lo que lo eliminaremos.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/5.JPG)


### En el caso de Embarked y Sex, cambiaremos sus valores por unos numericos, para poder manipularlos.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/6.JPG)

![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/7.JPG)

### Una vez cambiado lo anterior, rellenaremos los datos faltantes de edad, con el promedio de edad, que en este caso es 30 años, en la primera imagen se muestra como calcular el promedio y en la segunda aplicamos ese valor promedio a las edades que faltan.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/8.JPG)

![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/9.JPG)

### Ahora, si relacionamos las columnas de Age y Survived prodremos ver la relacion entre estos y determinar si la edad influye en la supervivencia. Se puede apreciar que muchos bebes o niños, sobrevivieron por lo que podria ser relevante este dato.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/10.JPG)

### En esta grafica, relacionamos Age, Survived y Pclass que es en que clase se encontraban, se puede observar que habia más personas en tercera clase, pero la mayoria murieron, por otro lado la primera clase se salvaron la mayoria de personas.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/11.JPG)

### Tenemos dos columnas que son: sibsp y parch que son cuantos niños, padres, hermanos, esposos, etc. tenian las personas, por lo que para reducir la cantidad de columnas y datos, podemos juntar estás dos en una que se llama FamilySize y al lado mostramos su porcentaje de supervivencia.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/13.JPG)

### Se puede simplificar aun más, puesto que en FamilySize tenemos alrededor de seis o siete opciones, si ponemos una columna llamada IsAlone, nos dira si iban solos o acompañados de esta forma se reduce a solo dos opciones, ahora con esto podemos eliminar FamilySize
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/14.JPG)

### Nuestra tabla de datos quedaria asi de momento
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/15.JPG)

### Tenemos una columna llamada Fare, esto es el costo del pasaje, podria estar relacionado con Pclass y ser determinante, pero son muchos valores por lo que se crean rangos de precio y vemos cuanto porcentaje sobrevivió, despues les asignamos un valor a cada rango.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/16.JPG)

### La tabla quedaria de esta forma de momento.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/17.JPG)

### Podriamos relacionar los nombres con su supervivencia, asi que obtenemos los titulos de los nombres, es decir, la palabra antes de un punto
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/18.JPG)

### Podemos limitar los nombres a cinco tipos, los que no esten entre los primeros cuatro estaran se consideraran como raro.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/20.JPG)

![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/19.JPG)


### Ahora tambien crearemos grupos por edades, para tenerlo más delimitado y sea más sencillo. 
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/21.JPG)

### Si aun hay algun dato faltante se eliminara
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/22.JPG)

### Verificamos si todo esta en orden, que no falte nada y que este todo lo más optimizado posible.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/23.JPG)

### Colocamos nuestro x_train, x_test y y_train, eliminando lo que no nos sirve de cada una de ellas.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/24.JPG)


### Sleccionamos los modelos que probaremos, en este caso son: regresion logistica, svc, kneighbors, decision tree y random forest
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/26.JPG)

### Estos son los resultados de los modelos, podemos observar que el desicion tree y random forest fueron los mejores, asi que seleccionare el desicion tree para predecir la supervivencia
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/25.JPG)

### Cree un archivo con los resultados usando el desicion tree, en el cual me dice que id de pasajero y su sobrevivio o no.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/27.JPG)



