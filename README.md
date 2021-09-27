# Prediccion Sobrevivientes Titanic con Machine Learning (Proyecto)

Este es un proyecto en el cual mediante machine learning podremos predecir los supervivientes del Titanic, en este ejemplo se hace la prueba con cinco modelos diferentes, todo esto utilizando python, a continuación cada parte de este proyecto.

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


### Seleccionamos los modelos que probaremos, en este caso son: regresion logistica, svc, kneighbors, decision tree y random forest
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/26.JPG)

### Estos son los resultados de los modelos, pero esto es sin hacer cross validation.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/25.JPG)

### Agregue esta linea para hacer cross-validation utilizando diez resultados, despues sacando su media y por ultimo redondeando a solo dos decimales
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/29.JPG)

### Como podemos observar la mayoria obtuvieron resultados similares, por el redondeo, hice más pruebas y en algunas ocaciones salian con puntaje más bajo excepto el Random Forest, por lo cual utilizare este modelo para hacer la predicción.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/28.JPG)

### Cree un archivo con los resultados usando el Random Forest, en el cual me dice que id de pasajero y si sobrevivio o no.
![](https://github.com/JuanWong02/proyecto1-titanic/blob/master/images/27.JPG)

##### Como conclusión podemos llegar a que este dataset si tenia datos importantes como la edad, sexo, si tenia familia e incluso la tarifa o sus nombres, pues aunque no lo parezca desde mi punto de vista si son relevantes, ya que con la tarifa se puede determinar que si pagaron más, estuvieron en primera clase, donde fueron los que más se salvaron
asi como sus nombres, no como tal sino como titulo para determinar si pudieran tener algun cargo importante y estar en alguna clase o relacionarlo por su sexo, estos datos junto con los otros de una forma optimizada y lo más simple posible se puede hacer la predicción de los sobrevivientes del titanic.



