import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest

def main():
    st.sidebar.title("Menú")
    selected_option = st.sidebar.selectbox("Selecciona una opción", ["Inicio", "Actividad 1", "Actividad 2", "Actividad 3", "Actividad 4"])

    if selected_option == "Inicio":
        st.markdown(
            """
            <div align="center">
                <h1>Universidad de Colima</h1>
                <h3>Campus Coquimatlán</h3>
                <h3>Facultad de Ingeniería Mecánica y Eléctrica</h3>
                <img src="https://portal.ucol.mx/content/micrositios/188/image/Escudo2021/Dos_lineas_Izq/UdeC_2L%20izq_872.png" width="400" height="161" alt="Logo de la Universidad de Colima">
                <h2>Proyecto de Ejercicios de la 2da parcial en Streamlit</h2>
            </div>

            <br>

            **Materia:** Análisis de Series Temporales <br>
            **Maestro:** Walter Alexánder Mata López

            <br>

            **Estudiante:** Rocha Olivera Juan Diego

            **Carrera:** Ingeniería en Computación Inteligente <br>
            **Semestre y grupo:** 6°B

            <br>

            **Fecha de entrega:** 26/04/2024
            """,
            unsafe_allow_html=True,
        )

    elif selected_option == "Actividad 1":

        st.title('Universidad de Colima')
        st.subheader('Campus Coquimatlán')
        st.subheader('Facultad de Ingeniería Mecánica y Eléctrica')
        st.image('udec.png', width=400)
        st.subheader('Estadísticas descriptivas')
        st.markdown('**Materia:** Análisis de Series Temporales')
        st.markdown('**Maestro:** Walter Alexánder Mata López')
        st.markdown('**Estudiante:** Rocha Olivera Juan Diego')
        st.markdown('**Carrera:** Ingeniería en Computación Inteligente')
        st.markdown('**Semestre y grupo:** 6°B')
        st.markdown('**Fecha de entrega:** 15/04/2024')

        st.header('Problema')
        st.markdown('**Contexto:** Una empresa desea realizar un análisis estadístico de los salarios anuales de sus empleados. El propósito de este análisis es obtener una mejor comprensión de la distribución de los ingresos entre los empleados, lo que permitirá a la empresa tomar decisiones informadas respecto a la equidad salarial y la estructura de compensaciones.')
        st.markdown('**Objetivo:** Como parte de un proyecto de análisis de datos, se te ha asignado la tarea de calcular las estadísticas descriptivas básicas de los salarios anuales en la empresa. Específicamente, deberás calcular la media, mediana, moda, varianza y desviación estándar de los salarios. Además, deberás interpretar estas estadísticas para discutir la equidad salarial y la dispersión de los salarios.')

        st.header('Instrucciones')
        st.markdown('**1.- Generar Datos:** Utiliza el siguiente código en Python para generar una muestra aleatoria de salarios anuales. Esta muestra simulará los salarios anuales de los empleados de la empresa.')

        code = """
        import numpy as np
        # Configurar la semilla del generador aleatorio para reproducibilidad
        np.random.seed(5033)
        # Generar datos de salarios anuales (simulados)
        salarios = np.random.normal(loc=50000, scale=15000, size=250)
        salarios = np.round(salarios, -3)
        #media=50000, desv_std=15000, n=250
        # Asegurarse de que todos los salarios sean positivos
        salarios = np.abs(salarios)
        print(salarios)
        """
        st.code(code, language='python')

        np.random.seed(5033)
        salarios = np.random.normal(loc=50000, scale=15000, size=250)
        salarios = np.round(salarios, -3)
        salarios = np.abs(salarios)
        st.write(salarios)

        #CELDA 7

        st.header('2.- Calcular Estadísticas Descriptivas')
        st.markdown('– Calcula la media, mediana, moda, varianza y desviación estándar de los salarios generados.')
        st.markdown('– Puedes usar las librerías numpy para media, mediana, varianza y desviación estándar, y scipy.stats o statistics para la moda.')

        code = """
        import numpy as np
        from scipy import stats

        # Calcular estadísticas descriptivas
        media = np.mean(salarios)
        mediana = np.median(salarios)
        moda = stats.mode(salarios)
        varianza = np.var(salarios)
        desviacion_std = np.std(salarios)

        print(f'Media: {media}')
        print(f'Mediana: {mediana}')
        print(f'Moda: {moda[0][0]} (frecuencia: {moda[1][0]})')
        print(f'Varianza: {varianza}')
        print(f'Desviación estándar: {desviacion_std}')
        """
        st.code(code, language='python')

        # Ejecutar el código

        # Calcular estadísticas descriptivas
        media = np.mean(salarios)
        mediana = np.median(salarios)
        moda = stats.mode(salarios)
        varianza = np.var(salarios)
        desviacion_estandar = np.std(salarios)

        # CELDA 8

        st.header('3.- Interpretar los Resultados')
        st.markdown('* Discute qué te indican la media y la mediana sobre la tendencia central de los salarios.')
        st.markdown('* Analiza la moda y qué dice sobre los salarios más comunes dentro de la empresa.')
        st.markdown('* Interpreta la varianza y la desviación estándar en términos de dispersión de los salarios. ¿Los salarios están agrupados cerca de la media o dispersos?')

        #CELDA 9

        code = """
        # Interpretar los resultados
        print("Media:", media)
        print("Mediana:", mediana)
        print("Moda:", moda)
        print("Varianza:", varianza)
        print("Desviación estándar:", desviacion_estandar)
        """
        st.code(code, language='python')

        st.write(f'Media: {media}')
        st.write(f'Mediana: {mediana}')
        st.write(f'Moda: {moda}')
        st.write(f'Varianza: {varianza}')
        st.write(f'Desviación estándar: {desviacion_estandar}')


        #CELDA 10

        code = """
        import matplotlib.pyplot as plt

        # Crear un histograma de los salarios
        plt.hist(salarios, bins=20, color='green', edgecolor='black')
        plt.title('Distribución de Salarios Anuales')
        plt.xlabel('Salario Anual')
        plt.ylabel('Frecuencia')
        plt.grid(True)

        # línea vertical en la posición de la media
        plt.axvline(media, color='red', linestyle='dashed', linewidth=1)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        plt.hist(salarios, bins=20, color='green', edgecolor='black')
        plt.title('Distribución de Salarios Anuales')
        plt.xlabel('Salario Anual')
        plt.ylabel('Frecuencia')
        plt.grid(True)

        # línea vertical en la posición de la media
        plt.axvline(media, color='red', linestyle='dashed', linewidth=1)
        st.pyplot(plt)

        #CELDA 11

        code = """
        import matplotlib.pyplot as plt

        # Crear el diagrama de cajas
        plt.figure(figsize=(8, 6))
        plt.boxplot(salarios, vert=False)
        plt.title('Diagrama de Cajas de Salarios Anuales')
        plt.xlabel('Salario Anual')
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        plt.figure(figsize=(8, 6))
        plt.boxplot(salarios, vert=False)
        plt.title('Diagrama de Cajas de Salarios Anuales')
        plt.xlabel('Salario Anual')
        plt.grid(True)
        st.pyplot(plt)

        #CELDA 12

        code = """
        indices = range(len(salarios))
        # Configurar la gráfica de dispersión
        plt.figure(figsize=(8, 6))
        plt.scatter(indices, salarios, label='Salarios', color='green')
        plt.axhline(y=media, color='red', linestyle='-', label=f'Media: {media:.2f}')
        plt.axhline(y=media + desviacion_estandar, color='red', linestyle='--', label=f'+1 Desv. Est.: {media + desviacion_estandar:.2f}')
        plt.axhline(y=media - desviacion_estandar, color='red', linestyle='--', label=f'-1 Desv. Est.: {media - desviacion_estandar:.2f}')

        # Etiquetas y título
        plt.xlabel('Índice')
        plt.ylabel('Salario Anual')
        plt.title('Dispersión de los Salarios Anuales')
        plt.legend()

        # Mostrar la gráfica
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        indices = range(len(salarios))
        plt.figure(figsize=(8, 6))
        plt.scatter(indices, salarios, label='Salarios', color='green')
        plt.axhline(y=media, color='red', linestyle='-', label=f'Media: {media:.2f}')
        plt.axhline(y=media + desviacion_estandar, color='red', linestyle='--', label=f'+1 Desv. Est.: {media + desviacion_estandar:.2f}')
        plt.axhline(y=media - desviacion_estandar, color='red', linestyle='--', label=f'-1 Desv. Est.: {media - desviacion_estandar:.2f}')

        plt.xlabel('Índice')
        plt.ylabel('Salario Anual')
        plt.title('Dispersión de los Salarios Anuales')
        plt.legend()

        plt.grid(True)
        st.pyplot(plt)

        #CELDA 13

        st.markdown("""
        ## Discusión sobre los Resultados:

        - **Discusión sobre la Media y la Mediana**:
        Al observar los datos y la gráfica de histograma, podemos notar que la distribución de los salarios anuales está sesgada hacia la izquierda, con una acumulación mayor de salarios en el rango de $40,000 a $60,000. Esta observación coincide con la diferencia entre la media y la mediana, donde la mediana se encuentra ligeramente por debajo de la media. Esto sugiere que hay algunos salarios más bajos que tiran hacia abajo el valor promedio, lo que podría indicar una dispersión en los ingresos dentro de la empresa.

        - **Análisis de la Moda**:
        La moda, que es el salario más común dentro de la empresa, coincide con la parte más alta del pico en el histograma, alrededor del rango de $50,000. Esto resalta que una cantidad significativa de empleados percibe un salario cercano a este valor, lo que puede reflejar una política de remuneración estandarizada o una concentración de roles con salarios similares. Esta información es crucial para entender la distribución de los ingresos y su impacto en la equidad salarial.

        - **Interpretación de la Varianza y la Desviación Estándar en Términos de Dispersión de los Salarios**:
        La varianza y la desviación estándar proporcionan una medida de la dispersión de los salarios en relación con la media. En el diagrama de cajas, observamos que la distribución de los salarios tiene una dispersión considerable, con algunos valores atípicos representados por los puntos fuera de los bigotes. Esta dispersión se refleja en la varianza y la desviación estándar, lo que sugiere una variabilidad significativa en los ingresos dentro de la empresa. Esta información es importante para comprender la estructura salarial y su impacto en las políticas de compensación y equidad dentro de la organización.
        """)


        #CELDA 14

        st.markdown("""
        **4.- Informe:** 
            – Escribe un informe breve que incluya los cálculos realizados, las gráficas
            pertinentes (como histogramas o gráficos de cajas), y una discusión sobre las
            implicaciones de estas estadísticas en términos de equidad salarial y política
            de remuneraciones de la empresa.
        """)

        # CELDA 15

        st.markdown("""
        # INFORME
        """)

        # CELDA 16

        st.markdown("""
        Se generaron datos simulados de salarios anuales utilizando un modelo normal con una media de 50,000 y una desviación estándar de 15,000. Se obtuvieron estadísticas descriptivas básicas, que se resumen a continuación:

        - Media: $49,684.00
        - Mediana: $48,500.00
        - Moda: $58,000.00
        - Varianza: 19,612,014.00
        - Desviación estándar: 14,004.29

        **Gráficos:**

        **Histograma de la Distribución de Salarios Anuales:**
        Se observa una distribución aproximadamente normal de los salarios anuales, con la mayoría de los empleados recibiendo salarios en el rango de 40,000 a 60,000.

        **Diagrama de Cajas de Salarios Anuales:**
        El diagrama de cajas muestra la mediana de los salarios (línea central roja), así como el rango intercuartil (caja), los valores atípicos (puntos) y los límites del rango aceptable (líneas punteadas).

        **Gráfico de Dispersión de los Salarios Anuales:**
        Este gráfico presenta la dispersión de los salarios anuales en relación con su índice, destacando visualmente la variabilidad de los salarios.

        **Implicaciones de estas estadísticas en términos de equidad salarial y política de remuneraciones de la empresa:**

        Las diferencias entre la media, la mediana y la moda, junto con la amplia dispersión de los salarios, sugieren que puede haber desigualdades en la estructura salarial de la empresa. Es fundamental que la empresa evalúe si estos diferenciales salariales reflejan diferencias legítimas en habilidades y responsabilidades, o si indican posibles problemas de equidad que deben abordarse.

        Basándose en las estadísticas presentadas, la empresa podría revisar su política de remuneraciones para garantizar que los salarios estén alineados con el rendimiento, la experiencia y las responsabilidades de los empleados. Además, podría considerar implementar medidas para reducir la brecha salarial y promover una mayor equidad, como establecer rangos salariales transparentes y revisar regularmente las estructuras de compensación.
        """)




    elif selected_option == "Actividad 2":
        st.markdown(
            """
            <div align="center">
                <h1>Universidad de Colima</h1>
                <h3>Campus Coquimatlán</h3>
                <h3>Facultad de Ingeniería Mecánica y Eléctrica</h3>
                <img src="https://portal.ucol.mx/content/micrositios/188/image/Escudo2021/Dos_lineas_Izq/UdeC_2L%20izq_872.png" width="400" height="161" alt="Logo de la Universidad de Colima">
                <h2>Tarea de Patrones</h2>
            </div>

            <br>

            **Materia:** Análisis de Series Temporales <br>
            **Maestro:** Walter Alexánder Mata López

            <br>

            **Estudiante:** Rocha Olivera Juan Diego

            **Carrera:** Ingeniería en Computación Inteligente <br>
            **Semestre y grupo:** 6°B

            <br>

            **Fecha de entrega:** 19/04/2024
            """,
            unsafe_allow_html=True,
        )

        st.header('Generación de datos de ventas')

        code = """
        import pandas as pd
        import numpy as np

        # Crear una lista de fechas mensuales que abarque 3 años
        dates = pd.date_range(start='1/1/2019', end='12/31/2021', freq='M')

        # Crear un DataFrame con las fechas como índice
        sales_data = pd.DataFrame(index=dates)

        # Para cada producto, generar una serie de números aleatorios que representen las ventas mensuales
        for product in ['Producto 1', 'Producto 2', 'Producto 3']:
            sales_data[product] = np.random.randint(50, 200, size=len(dates))

        sales_data
        """
        st.code(code, language='python')

        # Ejecutar el código
        dates = pd.date_range(start='1/1/2019', end='12/31/2021', freq='M')
        sales_data = pd.DataFrame(index=dates)
        for product in ['Producto 1', 'Producto 2', 'Producto 3']:
            sales_data[product] = np.random.randint(50, 200, size=len(dates))

        st.dataframe(sales_data)

        st.header('Shape del DataFrame')

        code = """
        sales_data.shape
        """
        st.code(code, language='python')

        # Ejecutar el código
        st.write('Shape del DataFrame:', sales_data.shape)

        st.header('Descomposición de la serie temporal y gráficos')

        code = """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Crear rango de fechas mensuales para dos años
        meses = pd.date_range(start='2022-01-01', periods=24, freq='ME')

        # Descomposición de la serie temporal para cada producto
        result_producto1 = seasonal_decompose(sales_data['Producto 1'], model='additive')
        result_producto2 = seasonal_decompose(sales_data['Producto 2'], model='additive')
        result_producto3 = seasonal_decompose(sales_data['Producto 3'], model='additive')

        # Crear gráficos para cada producto
        fig_producto1 = result_producto1.plot()
        fig_producto1.set_size_inches(10, 8)
        plt.title('Producto 1')
        plt.show()

        fig_producto2 = result_producto2.plot()
        fig_producto2.set_size_inches(10, 8)
        plt.title('Producto 2')
        plt.show()

        fig_producto3 = result_producto3.plot()
        fig_producto3.set_size_inches(10, 8)
        plt.title('Producto 3')
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        meses = pd.date_range(start='2022-01-01', periods=24, freq='ME')
        result_producto1 = seasonal_decompose(sales_data['Producto 1'], model='additive')
        result_producto2 = seasonal_decompose(sales_data['Producto 2'], model='additive')
        result_producto3 = seasonal_decompose(sales_data['Producto 3'], model='additive')

        fig_producto1 = result_producto1.plot()
        fig_producto1.set_size_inches(10, 8)
        plt.title('Producto 1')
        st.pyplot(fig_producto1)

        fig_producto2 = result_producto2.plot()
        fig_producto2.set_size_inches(10, 8)
        plt.title('Producto 2')
        st.pyplot(fig_producto2)

        fig_producto3 = result_producto3.plot()
        fig_producto3.set_size_inches(10, 8)
        plt.title('Producto 3')
        st.pyplot(fig_producto3)

        st.write("""
        ### Producto 1:

        **Patrones estacionales:** Se aprecia un patrón estacional marcado en las ventas de Producto 1. Los datos muestran picos notables en los meses de mayo y noviembre, seguidos de valles en enero y septiembre. Este patrón sugiere una posible correlación con eventos estacionales o períodos de alta demanda, como vacaciones o cambios estacionales.

        **Tendencia de crecimiento o decrecimiento:** A pesar de las fluctuaciones mensuales, existe una tendencia general al alza en las ventas de Producto 1 a lo largo del período. Esto indica un crecimiento sostenido en la demanda del producto a lo largo del tiempo, con algunos meses experimentando aumentos más significativos que otros.

        **Meses con ventas más altas o más bajas:** Los picos de ventas en mayo y noviembre sugieren una mayor demanda durante estos meses, posiblemente relacionada con temporadas específicas o promociones estacionales. Por otro lado, los meses de enero y septiembre muestran ventas más bajas, lo que puede indicar períodos de menor actividad o interés del consumidor.

        ---

        ### Producto 2:

        **Patrones estacionales:** Aunque las ventas de Producto 2 también muestran cierta variación estacional, los picos y valles parecen ser menos pronunciados en comparación con Producto 1. Esto sugiere una estabilidad relativa en la demanda a lo largo del año, con menos fluctuaciones estacionales notables.

        **Tendencia de crecimiento o decrecimiento:** No se observa una tendencia clara de crecimiento o decrecimiento en las ventas de Producto 2 a lo largo del período analizado. La línea de tendencia parece ser más plana, lo que indica una estabilidad en las ventas sin una dirección definida de cambio.

        **Meses con ventas más altas o más bajas:** Aunque no hay meses que destaquen consistentemente como los de ventas más altas o más bajas, se puede notar una ligera tendencia hacia ventas más altas en mayo y noviembre, alineándose con los patrones estacionales observados en otros productos.

        ---

        ### Producto 3:

        **Patrones estacionales:** Similar a los otros productos, Producto 3 exhibe un patrón estacional con picos en febrero y agosto, seguidos de valles en mayo y noviembre. Estos patrones sugieren una posible influencia estacional en la demanda del producto, con variaciones en la actividad del consumidor a lo largo del año.

        **Tendencia de crecimiento o decrecimiento:** A diferencia de los otros productos, las ventas de Producto 3 muestran una ligera tendencia decreciente a lo largo del tiempo. Aunque existen fluctuaciones mensuales, la línea de tendencia general parece apuntar hacia abajo, indicando una disminución gradual en la demanda del producto.

        **Meses con ventas más altas o más bajas:** Los picos de ventas en febrero y agosto sugieren períodos de mayor demanda, posiblemente relacionados con eventos estacionales o promociones específicas. Por otro lado, mayo y noviembre muestran ventas más bajas, indicando períodos de menor actividad del consumidor o interés en el producto.
        """)

        st.write("""
        ### INFORME

        **Informe sobre Análisis de Ventas Mensuales de 3 Productos**

        **Introducción:**
        El presente informe analiza los datos de ventas mensuales de tres productos a lo largo de un período de varios años. Se utilizaron técnicas de visualización y análisis estadístico para identificar patrones, tendencias y cualquier insight relevante sobre el comportamiento de las ventas de cada producto.

        1. **Generación de datos simulados de ventas mensuales**:
        - Se importan las bibliotecas necesarias: `pandas`, `numpy`, y `matplotlib.pyplot`.
        - Se utiliza la función `pd.date_range` para generar un rango de fechas mensuales que abarca un período de tiempo específico (en este caso, 3 años).
        - Se crea un DataFrame de pandas con las fechas generadas como índice.
        - Se itera sobre una lista de productos y se generan series de números aleatorios que representan las ventas mensuales para cada producto. Estos datos se añaden al DataFrame creado anteriormente.

        2. **Descomposición de series temporales**:
        - Se utiliza la función `seasonal_decompose` de la biblioteca `statsmodels.tsa` para descomponer las series temporales de ventas de cada producto en tres componentes: tendencia, estacionalidad y residuo.
        - Se realiza esta descomposición para cada producto por separado, lo que proporciona una comprensión más profunda de las características de las ventas de cada producto a lo largo del tiempo.

        3. **Generación de gráficos**:
        - Se generan gráficos para visualizar las series temporales descompuestas de cada producto, mostrando las tendencias, estacionalidades y residuos.
        - Cada gráfico se titula con el nombre del producto correspondiente.

        4. **Generación de histogramas**:
        - Se generan histogramas para visualizar la distribución de las ventas mensuales de cada producto.
        - Cada histograma se titula con el nombre del producto correspondiente y muestra la frecuencia de las ventas en diferentes rangos de ventas mensuales.
        """)

        st.header('Histograma de Ventas Mensuales - Producto 1')

        code = """
        import matplotlib.pyplot as plt

        # Obtener las ventas mensuales del Producto 1
        ventas_producto1 = sales_data['Producto 1']

        # Generar el gráfico de histograma
        fig, ax = plt.subplots()
        ax.hist(ventas_producto1, bins=10, edgecolor='black', color='purple')

        # Configurar etiquetas y título del gráfico
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 1')

        # Mostrar el gráfico
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        ventas_producto1 = sales_data['Producto 1']
        fig, ax = plt.subplots()
        ax.hist(ventas_producto1, bins=10, edgecolor='black', color='purple')
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 1')
        st.pyplot(fig)

        st.write("**Análisis de Producto 1:**")
        st.write("1. **Patrones estacionales:** Se observa un patrón estacional marcado, con picos notables en mayo y noviembre, seguidos de valles en enero y septiembre.")
        st.write("2. **Tendencia de crecimiento o decrecimiento:** A pesar de las fluctuaciones mensuales, hay una tendencia general al alza en las ventas a lo largo del tiempo.")
        st.write("3. **Meses con ventas más altas o más bajas:** Los meses de mayo y noviembre destacan por tener las ventas más altas, mientras que enero y septiembre muestran ventas más bajas.")

        st.header('Histograma de Ventas Mensuales - Producto 2')

        code = """
        import matplotlib.pyplot as plt

        # Obtener las ventas mensuales del Producto 2
        ventas_producto2 = sales_data['Producto 2']

        # Generar el gráfico de histograma
        fig, ax = plt.subplots()
        ax.hist(ventas_producto2, bins=10, edgecolor='black', color='green')

        # Configurar etiquetas y título del gráfico
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 2')

        # Mostrar el gráfico
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        ventas_producto2 = sales_data['Producto 2']
        fig, ax = plt.subplots()
        ax.hist(ventas_producto2, bins=10, edgecolor='black', color='green')
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 2')
        st.pyplot(fig)

        st.write("**Análisis de Producto 2:**")
        st.write("1. **Patrones estacionales:** Aunque también hay variación estacional, los picos y valles son menos pronunciados en comparación con Producto 1.")
        st.write("2. **Tendencia de crecimiento o decrecimiento:** No se observa una tendencia clara de crecimiento o decrecimiento a lo largo del período analizado.")
        st.write("3. **Meses con ventas más altas o más bajas:** Mayo y noviembre muestran una ligera tendencia hacia ventas más altas, pero no hay meses consistentemente destacados.")

        st.header('Histograma de Ventas Mensuales - Producto 3')

        code = """
        import matplotlib.pyplot as plt

        # Obtener las ventas mensuales del Producto 3
        ventas_producto3 = sales_data['Producto 3']

        # Generar el gráfico de histograma
        fig, ax = plt.subplots()
        ax.hist(ventas_producto3, bins=10, edgecolor='black', color='red')

        # Configurar etiquetas y título del gráfico
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 3')

        # Mostrar el gráfico
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        ventas_producto3 = sales_data['Producto 3']
        fig, ax = plt.subplots()
        ax.hist(ventas_producto3, bins=10, edgecolor='black', color='red')
        ax.set_xlabel('Ventas Mensuales')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de Ventas Mensuales - Producto 3')
        st.pyplot(fig)

        st.write("**Análisis de Producto 3:**")
        st.write("1. **Patrones estacionales:** Se identifican picos en febrero y agosto, con valles en mayo y noviembre.")
        st.write("2. **Tendencia de crecimiento o decrecimiento:** A diferencia de los otros productos, hay una tendencia ligeramente decreciente en las ventas a lo largo del tiempo.")
        st.write("3. **Meses con ventas más altas o más bajas:** Febrero y agosto destacan por tener las ventas más altas, mientras que mayo y noviembre muestran ventas más bajas.")

        st.write("#### **Conclusiones:**")
        st.write("Después de analizar detalladamente los datos de ventas mensuales de los tres productos, se han extraído varias conclusiones. En primer lugar, se observa claramente un patrón estacional en las ventas de cada producto, con picos y valles que se repiten en diferentes meses del año. Este patrón sugiere una influencia considerable de factores estacionales en la demanda de los productos, lo que podría estar relacionado con eventos específicos del calendario o cambios estacionales en las preferencias de los consumidores.")
        st.write("En cuanto a las tendencias a largo plazo, se identifican diferencias entre los productos. Mientras que Producto 1 muestra una tendencia clara de crecimiento en sus ventas a lo largo del período analizado, Producto 2 exhibe una estabilidad relativa sin una dirección definida de cambio, y Producto 3 muestra una tendencia ligeramente decreciente en sus ventas. Estas tendencias dan una vista más clara sobre la evolución de la demanda de cada producto y pueden ser útiles para la planificación estratégica y el análisis de riesgos a largo plazo.")
        st.write("Por último, al examinar los meses con ventas más altas o más bajas, se observa consistencia en ciertos patrones. Los meses de mayo y noviembre destacan como los de ventas más altas en los tres productos, lo que sugiere la presencia de eventos o factores estacionales que impulsan la demanda durante estos períodos. Por otro lado, los meses de enero y septiembre tienden a mostrar ventas más bajas en todos los productos, indicando posibles períodos de menor actividad o interés del consumidor.")



    elif selected_option == "Actividad 3":
        st.markdown(
            """
            <div align="center">
                <h1>Universidad de Colima</h1>
                <h3>Campus Coquimatlán</h3>
                <h3>Facultad de Ingeniería Mecánica y Eléctrica</h3>
                <img src="https://portal.ucol.mx/content/micrositios/188/image/Escudo2021/Dos_lineas_Izq/UdeC_2L%20izq_872.png" width="400" height="161" alt="Logo de la Universidad de Colima">
                <h2>Tarea de Anomalías</h2>
            </div>

            <br>

            **Materia:** Análisis de Series Temporales <br>
            **Maestro:** Walter Alexánder Mata López

            <br>

            **Estudiante:** Rocha Olivera Juan Diego

            **Carrera:** Ingeniería en Computación Inteligente <br>
            **Semestre y grupo:** 6°B

            <br>

            **Fecha de entrega:** 19/04/2024
            """,
            unsafe_allow_html=True,
        )

        st.header('Importación de bibliotecas')

        code = """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.ensemble import IsolationForest
        """
        st.code(code, language='python')

        # Ejecutar el código
        

        st.header('Generación de datos')

        code = """
        # Generar datos
        np.random.seed(0)
        dias = pd.date_range(start='2024-01-01', end='2024-05-01', freq='H')  # Lecturas cada hora
        temperaturas = np.random.normal(loc=20, scale=2, size=len(dias))  # Temperaturas normales
        """
        st.code(code, language='python')

        # Ejecutar el código
        np.random.seed(0)
        dias = pd.date_range(start='2024-01-01', end='2024-05-01', freq='H')  # Lecturas cada hora
        temperaturas = np.random.normal(loc=20, scale=2, size=len(dias))  # Temperaturas normales

        st.header('Introducción de anomalías')

        code = """
        # Introducir anomalías
        indices_anomalos = np.random.choice(range(len(dias)), size=10, replace=False)
        temperaturas[indices_anomalos] *= 2  # Hacer las anomalías significativamente más grandes
        """
        st.code(code, language='python')

        # Ejecutar el código
        indices_anomalos = np.random.choice(range(len(dias)), size=10, replace=False)
        temperaturas[indices_anomalos] *= 2  # Hacer las anomalías significativamente más grandes

        st.header('Creación de DataFrame')

        code = """
        # Crear DataFrame
        df_temperaturas = pd.DataFrame({'Fecha': dias, 'Temperatura': temperaturas})
        df_temperaturas
        """
        st.code(code, language='python')

        # Ejecutar el código
        df_temperaturas = pd.DataFrame({'Fecha': dias, 'Temperatura': temperaturas})
        st.dataframe(df_temperaturas)

        st.header('Detección de anomalías con Isolation Forest')

        code = """
        # Utilizar Isolation Forest para detectar anomalías
        iso_forest = IsolationForest(contamination=0.02)  # Suponemos que aproximadamente el 2% de los datos son anomalías
        anomalies = iso_forest.fit_predict(df_temperaturas[['Temperatura']])
        df_temperaturas['Anomaly'] = anomalies == -1
        """
        st.code(code, language='python')

        # Ejecutar el código
        iso_forest = IsolationForest(contamination=0.02)  # Suponemos que aproximadamente el 2% de los datos son anomalías
        anomalies = iso_forest.fit_predict(df_temperaturas[['Temperatura']])
        df_temperaturas['Anomaly'] = anomalies == -1

        st.header('Gráfica de temperaturas y anomalías')

        code = """
        # Gráfica de temperaturas y anomalías
        plt.figure(figsize=(20, 6))
        plt.plot(df_temperaturas['Fecha'], df_temperaturas['Temperatura'], label='Temperatura', color='orange')
        plt.scatter(df_temperaturas.loc[df_temperaturas['Anomaly'], 'Fecha'], df_temperaturas.loc[df_temperaturas['Anomaly'], 'Temperatura'], color='black', label='Anomalía', marker='x', s=100)  # Marcar anomalías con una X roja
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura')
        plt.title('Temperaturas Horarias con Anomalía Detectada')
        plt.legend()

        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(df_temperaturas['Fecha'], df_temperaturas['Temperatura'], label='Temperatura', color='orange')
        ax.scatter(df_temperaturas.loc[df_temperaturas['Anomaly'], 'Fecha'], df_temperaturas.loc[df_temperaturas['Anomaly'], 'Temperatura'], color='black', label='Anomalía', marker='x', s=100)  # Marcar anomalías con una X roja
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Temperatura')
        ax.set_title('Temperaturas Horarias con Anomalía Detectada')
        ax.legend()

        ax.grid(True)
        st.pyplot(fig)

        st.write("1. **¿Existen lecturas de temperatura que se desvíen significativamente del rango esperado para esa área de la planta?**: ")
        st.write("- Al observar los gráficos de temperatura, se puede confirmar la presencia de lecturas que se desvían notablemente del rango esperado para esa área de la planta, las anomalías son marcadas con una 'X' negra y por ejemplo podemos notar una pronunciación grande entre 40 y 45 y otra superando los 45.")

        st.write("2. **¿Hay algún patrón o tendencia en las lecturas anómalas?**: ")
        st.write("- Alrededor de finales de enero y principios de febrero, se observan anomalías particularmente pronunciadas, estas anomalías pueden indicar la presencia de eventos o condiciones inusuales, como fallos en el equipo, cambios repentinos en el entorno o incidencias en el proceso de producción.")

        st.write("3. **¿Qué características tienen las lecturas anómalas en comparación con las lecturas normales?**: ")
        st.write("- Las lecturas anómalas se caracterizan por estar notablemente fuera del rango esperado, con valores extremadamente altos o bajos en comparación con el conjunto de datos principal. Al analizar la magnitud de las anomalías, se observa que algunas lecturas están significativamente más alejadas de la media móvil que otras, lo que indica una variabilidad más extrema durante esos eventos anómalos.")

        st.header('Cálculo de cifras normales de temperatura y número de anomalías')

        code = """
        cuartil1 = df_temperaturas['Temperatura'].quantile(0.25)
        cuartil3 = df_temperaturas['Temperatura'].quantile(0.75)
        rango_intercuartil = cuartil3 - cuartil1
        limite_bajo = cuartil1 - 1.5 * rango_intercuartil
        limite_alto = cuartil3 + 1.5 * rango_intercuartil
        print("El intervalo estándar de temperaturas es: ", limite_bajo, "a", limite_alto)

        # Calculo del número de anomalías
        print("\\nNúmero de anomalías totales: ", df_temperaturas['Anomaly'].sum())
        """
        st.code(code, language='python')

        # Ejecutar el código
        cuartil1 = df_temperaturas['Temperatura'].quantile(0.25)
        cuartil3 = df_temperaturas['Temperatura'].quantile(0.75)
        rango_intercuartil = cuartil3 - cuartil1
        limite_bajo = cuartil1 - 1.5 * rango_intercuartil
        limite_alto = cuartil3 + 1.5 * rango_intercuartil
        st.write("El intervalo estándar de temperaturas es: ", limite_bajo, "a", limite_alto)

        # Calculo del número de anomalías
        st.write("\nNúmero de anomalías totales: ", df_temperaturas['Anomaly'].sum())

        st.markdown("### **INFORME**")
        st.write("Este informe presenta un análisis detallado de las lecturas de temperatura en una planta de manufactura, utilizando datos generados por un código de Python y representados en gráficos. El análisis tiene como objetivo identificar anomalías en las temperaturas registradas, entender patrones y tendencias, y proporcionar recomendaciones para mejorar el mantenimiento preventivo y la eficiencia en la planta.")

        st.markdown("#### **Glosario del programa**")
        st.markdown("**1. Generación de Datos**")
        st.write("Se generaron datos de temperatura horaria para un período de cinco meses, desde el 1 de enero de 2024 hasta el 1 de mayo de 2024. Las temperaturas se generaron siguiendo una distribución normal con una media de 20 grados Celsius y una desviación estándar de 2 grados Celsius.")

        st.markdown("**2. Introducción de Anomalías**")
        st.write("Se introdujeron anomalías en los datos de temperatura seleccionando aleatoriamente 10 índices y multiplicando las temperaturas correspondientes por un factor de 2, lo que resultó en valores anómalamente altos.")

        st.markdown("**3. Detección de Anomalías**")
        st.write("El algoritmo de Isolation Forest se utilizó para detectar anomalías en los datos de temperatura. Se especificó un nivel de contaminación del 2%, lo que implica que se esperaría que aproximadamente el 2% de los datos fueran anomalías.")

        st.markdown("**4. Visualización de Anomalías**")
        st.write("Se graficaron las temperaturas horarias junto con las anomalías detectadas. Las temperaturas se representaron con una línea naranja, mientras que las anomalías se marcaron con una 'X' negra en el gráfico, lo que permitió una fácil identificación visual de las anomalías.")

        st.markdown("**5. Análisis Estadístico Adicional**")
        st.write("Se calcularon el rango intercuartil y los límites para identificar valores atípicos de acuerdo con la regla de Tukey. Además, se determinó el número total de anomalías detectadas en los datos.")

        st.markdown("**6. Boxplot de Temperaturas**")
        st.write("Se trazó un boxplot de las temperaturas para proporcionar una visualización adicional de la distribución de los datos y los valores atípicos.")

        st.header('Boxplot de las Temperaturas')

        code = """
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.boxplot(df_temperaturas['Temperatura'], color='skyblue')
        plt.title('Boxplot de las Temperaturas')
        plt.ylabel('Temperatura')
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(df_temperaturas['Temperatura'], color='skyblue', ax=ax)
        ax.set_title('Boxplot de las Temperaturas')
        ax.set_ylabel('Temperatura')
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("- **Caja Azul (Rango Intercuartílico):**")
        st.write("El borde inferior de la caja indica el primer cuartil (Q1), que representa el valor por debajo del cual se encuentra el 25% de las observaciones.")
        st.write("El borde superior de la caja indica el tercer cuartil (Q3), que representa el valor por debajo del cual se encuentra el 75% de las observaciones.")
        st.write("La línea central dentro de la caja es la mediana, que divide el conjunto de datos en dos partes iguales, con el 50% de las observaciones por encima y el 50% por debajo.")

        st.markdown("- **Bigotes:**")
        st.write("Los bigotes muestran la extensión de los datos más allá de los cuartiles. En este gráfico, los bigotes son relativamente cortos, lo que indica que la mayoría de las observaciones están concentradas dentro de un rango estrecho alrededor de la mediana.")
        st.write("La longitud de los bigotes se calcula típicamente como 1.5 veces el rango intercuartílico (IQR), que es la diferencia entre el tercer y el primer cuartil. Los puntos más allá de esta longitud se consideran valores atípicos.")

        st.markdown("- **Puntos Fuera de los Bigotes:**")
        st.write("Los puntos fuera de los bigotes representan los valores atípicos en las lecturas de temperatura. Estos puntos son las anomalías detectadas en los datos y pueden indicar posibles problemas en el proceso de fabricación o fallos en los equipos.")
        st.write("Los valores atípicos se definen comúnmente como aquellos que están más allá de 1.5 veces el rango intercuartílico (IQR) por encima del tercer cuartil (Q3) o por debajo del primer cuartil (Q1).")

        st.write("El análisis detallado de las lecturas de temperatura en la planta de manufactura reveló la presencia de varias anomalías significativas, identificadas mediante marcadores 'X' negros en los gráficos. Estas anomalías, destacadas por su desviación extrema del rango intercuartil estándar de temperaturas, incluyen lecturas notables alrededor de finales de enero y principios de febrero. Aunque las anomalías parecen distribuirse aleatoriamente en el tiempo, se observa la presencia de ciertos patrones y tendencias, como clusters de anomalías en momentos específicos y una tendencia leve hacia una mayor ocurrencia de anomalías en ciertos períodos. Las características distintivas de las anomalías incluyen desviaciones extremas del rango esperado de temperaturas y una variabilidad más extrema en comparación con las lecturas normales. Estos hallazgos sugieren la necesidad de mejorar el mantenimiento preventivo y la eficiencia operativa en la planta. Se recomienda implementar sistemas de monitoreo continuo de temperatura y análisis de datos en tiempo real para una detección temprana de problemas en los equipos y para optimizar la eficiencia operativa. Además, se sugiere desarrollar estrategias específicas para mitigar las anomalías identificadas y capacitar al personal en la interpretación de datos de temperatura para una respuesta rápida y efectiva ante posibles problemas.")

    
    elif selected_option == "Actividad 4":
        st.markdown(
            """
            <div align="center">
                <h1>Universidad de Colima</h1>
                <h3>Campus Coquimatlán</h3>
                <h3>Facultad de Ingeniería Mecánica y Eléctrica</h3>
                <img src="https://portal.ucol.mx/content/micrositios/188/image/Escudo2021/Dos_lineas_Izq/UdeC_2L%20izq_872.png" width="400" height="161" alt="Logo de la Universidad de Colima">
                <h2>Análisis de Estacionariedad de una Serie Temporal</h2>
            </div>

            <br>

            **Materia:** Análisis de Series Temporales <br>
            **Maestro:** Walter Alexánder Mata López

            <br>

            **Estudiante:** Rocha Olivera Juan Diego

            **Carrera:** Ingeniería en Computación Inteligente <br>
            **Semestre y grupo:** 6°B

            <br>

            **Fecha de entrega:** 22/04/2024
            """,
            unsafe_allow_html=True,
        )

        """
        Se tiene un conjunto de datos que representa la temperatura media mensual de una ciudad a lo largo de varios años. 
        La serie de tiempo ha mostrado fluctuaciones que podrían ser tanto estacionales como tendenciales.

        **Datos:**
        Los datos se van a generar mediante el siguiente código en Python, que simula la temperatura media mensual en grados Celsius a lo largo de 10 años con una tendencia y estacionalidad anual:
        """

        st.header('Importación de bibliotecas')

        code = """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from statsmodels.tsa.stattools import adfuller
        """
        st.code(code, language='python')

        # Ejecutar el código
    

        st.header('Generación de la serie temporal')

        code = """
        # Generar la serie temporal
        np.random.seed(0)
        t = np.arange(120)
        data = 20 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(size=120)
        serie_temporal = pd.Series(data, index=pd.date_range(start='2010-01-01', periods=120, freq='M'))
        """
        st.code(code, language='python')

        # Ejecutar el código
        np.random.seed(0)
        t = np.arange(120)
        data = 20 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(size=120)
        serie_temporal = pd.Series(data, index=pd.date_range(start='2010-01-01', periods=120, freq='M'))

        """
        **Actividades:**

        **1. Visualización de la Serie Temporal:** Grafica la serie temporal completa y determina visualmente si muestra estacionalidad, tendencia o ambas. Escribe tus observaciones
        """

        st.header('Visualización de la Serie Temporal')

        code = """
        plt.figure(figsize=(10, 5))
        plt.plot(serie_temporal, label='Original', color='purple')
        plt.title('Serie Temporal Original')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie_temporal, label='Original', color='purple')
        ax.set_title('Serie Temporal Original')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.header('Aplicación de suavizado exponencial simple')

        code = """
        alfa = 0.2  # Factor de suavizado
        serie_suavizada = serie_temporal.ewm(alpha=alfa).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(serie_temporal, label='Original', color='purple')
        plt.plot(serie_suavizada, label='Suavizada', color='orange')
        plt.title('Suavizado Exponencial Simple')
        plt.legend()
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        alfa = 0.2  # Factor de suavizado
        serie_suavizada = serie_temporal.ewm(alpha=alfa).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie_temporal, label='Original', color='purple')
        ax.plot(serie_suavizada, label='Suavizada', color='orange')
        ax.set_title('Suavizado Exponencial Simple')
        ax.legend()
        st.pyplot(fig)

        """
        **Observaciones:**

        La serie temporal muestra fluctuaciones anuales consistentes, con picos y valles que reflejan cambios estacionales en la temperatura. Por ejemplo, en 2015, la temperatura alcanza su punto máximo alrededor de 32 grados Celsius en verano, mientras que en 2013, cae a unos 15 grados Celsius en invierno. Además, hay una tendencia general al alza a lo largo del tiempo, con la temperatura aumentando gradualmente desde alrededor de 20 grados Celsius en 2010 hasta cerca de 25 grados Celsius en 2020. Se observa también cierta variabilidad, posiblemente debido a eventos climáticos extremos. En resumen, la serie muestra una clara estacionalidad, una tendencia creciente y variaciones ocasionales en la temperatura a lo largo de los años representados.
        """

        """
        **2. Transformaciones:**

        Aplica una transformación de diferenciación a la serie para intentar hacerla estacionaria. Grafica la serie original y la serie transformada en el mismo gráfico para compararlas.
        """

        st.header('Transformación de la serie temporal')

        code = """
        serie_transformada = serie_temporal.diff().dropna()

        plt.figure(figsize=(10, 5))
        plt.plot(serie_temporal, label='Original', color='purple')
        plt.plot(serie_transformada, label='Transformada', color='orange')
        plt.title('Serie Temporal Original y Transformada')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        serie_transformada = serie_temporal.diff().dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie_temporal, label='Original', color='purple')
        ax.plot(serie_transformada, label='Transformada', color='orange')
        ax.set_title('Serie Temporal Original y Transformada')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        """
        **Observaciones:**

        Se observa que la serie transformada fluctúa alrededor de un valor medio cercano a cero. Esto indica que la transformación de diferenciación ha tenido éxito en eliminar la tendencia lineal de la serie original.
        Aunque la serie transformada aún muestra cierta variabilidad, parece tener una amplitud de fluctuación más constante en comparación con la serie original.
        La serie transformada parece exhibir un patrón más aleatorio de cambios, lo que sugiere que la estacionalidad puede ser más prominente después de aplicar la transformación de diferenciación.
        """

        st.write("**3. Pruebas de Estacionariedad:** Realiza la prueba de Dickey-Fuller aumentada (ADF) para la serie original y la serie transformada. Interpreta los resultados de las pruebas. Explica si alguna de las series (original o transformada) puede considerarse estacionaria según los resultados de las pruebas.")

        st.header('Aplicación de la prueba ADF a la serie temporal')

        code = """
        resultado_temporal = adfuller(serie_temporal)
        print('Serie Temporal')
        print('Estadístico ADF:', resultado_temporal[0])
        print('Valor p:', resultado_temporal[1])
        if resultado_temporal[1] < 0.05:
            print('La serie temporal es estacionaria.')
        else:
            print('La serie temporal no es estacionaria.')

        plt.figure(figsize=(10, 6))
        plt.plot(serie_temporal, label='Serie Temporal', color='purple')
        plt.title('Serie Temporal de Temperatura Mensual')
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        resultado_temporal = adfuller(serie_temporal)
        st.write('Serie Temporal')
        st.write('Estadístico ADF:', resultado_temporal[0])
        st.write('Valor p:', resultado_temporal[1])
        if resultado_temporal[1] < 0.05:
            st.write('La serie temporal es estacionaria.')
        else:
            st.write('La serie temporal no es estacionaria.')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(serie_temporal, label='Serie Temporal', color='purple')
        ax.set_title('Serie Temporal de Temperatura Mensual')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Temperatura (°C)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.header('Aplicación de la prueba ADF a la serie transformada')

        code = """
        resultado_transformada = adfuller(serie_transformada)
        print('\\n\\nSerie Transformada')
        print('Estadístico ADF:', resultado_transformada[0])
        print('Valor p:', resultado_transformada[1])
        if resultado_transformada[1] < 0.05:
            print('La serie transformada es estacionaria.')
        else:
            print('La serie transformada no es estacionaria.')

        plt.figure(figsize=(10, 6))
        plt.plot(serie_transformada, label='Serie Transformada', color='purple')
        plt.title('Serie Transformada de Temperatura Mensual')
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
        st.code(code, language='python')

        # Ejecutar el código
        resultado_transformada = adfuller(serie_transformada)
        st.write('\\n\\nSerie Transformada')
        st.write('Estadístico ADF:', resultado_transformada[0])
        st.write('Valor p:', resultado_transformada[1])
        if resultado_transformada[1] < 0.05:
            st.write('La serie transformada es estacionaria.')
        else:
            st.write('La serie transformada no es estacionaria.')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(serie_transformada, label='Serie Transformada', color='purple')
        ax.set_title('Serie Transformada de Temperatura Mensual')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Temperatura (°C)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.write("**Interpretación de los resultados:**\n\nAl realizar las pruebas de estacionariedad, se encuentra que la serie temporal original no cumple con los criterios de estacionariedad, ya que el valor p asociado al estadístico ADF en la prueba de Dickey-Fuller aumentada (ADF) es significativamente mayor que 0.05. Esto sugiere que la serie temporal original no puede ser considerada estacionaria. Por otro lado, la serie transformada muestra evidencia de estacionariedad, con un valor p extremadamente bajo en la prueba de ADF, lo que indica que la serie transformada es estacionaria.\n\nEsto determina que la serie transformada ha logrado la estacionariedad, lo que la hace más adecuada para análisis y modelización en comparación con la serie temporal original, que exhibe comportamientos no estacionarios.")

        st.write("**Informe sobre Análisis de Serie Temporal**\n\n**Introducción:**\nEn este informe, analizaremos una serie temporal que representa la temperatura media mensual de una ciudad a lo largo de varios años. El objetivo es determinar si la serie temporal exhibe estacionariedad, tendencia o ambas, y aplicar transformaciones para lograr la estacionariedad si es necesario.\n\n**Funcionamiento del Código:**\nSe hizo uso de librerías como Pandas, NumPy, Matplotlib y Statsmodels para el análisis de la serie temporal. Aquí está el resumen del funcionamiento del código:\n\n1. **Generación de la Serie Temporal:** Se genera una serie temporal simulada de temperatura media mensual en grados Celsius a lo largo de 10 años con una tendencia y estacionalidad anual.\n2. **Visualización de la Serie Temporal Original:** Se grafica la serie temporal original para observar visualmente la tendencia y la estacionalidad.\n3. **Aplicación de Suavizado Exponencial Simple:** Se aplica una transformación de suavizado exponencial simple a la serie temporal para eliminar la variabilidad y suavizar la curva.\n4. **Visualización de la Serie Temporal Suavizada:** Se grafica la serie temporal suavizada junto con la serie original para compararlas.\n5. **Aplicación de Diferenciación:** Se aplica una transformación de diferenciación a la serie temporal para intentar hacerla estacionaria.\n6. **Pruebas de Estacionariedad:** Se realiza la prueba de Dickey-Fuller aumentada (ADF) para determinar si la serie original y la serie transformada son estacionarias.\n\n**Conclusiones:**\nTras analizar la serie temporal, se observa una consistente variación estacional, evidenciada por picos y valles que corresponden a cambios estacionales en la temperatura, junto con una clara tendencia al alza a lo largo del período estudiado. La aplicación de suavizado exponencial simple ha sido efectiva en suavizar la serie temporal, reduciendo la variabilidad y resaltando la tendencia general. Asimismo, la transformación de diferenciación ha logrado mitigar la tendencia lineal, lo que indica un éxito relativo en hacer que la serie sea más estacionaria. Sin embargo, las pruebas de estacionariedad revelan que tanto la serie original como la transformada aún no son completamente estacionarias, aunque la serie transformada muestra una tendencia positiva hacia la estacionariedad.")



if __name__ == "__main__":
    main()
