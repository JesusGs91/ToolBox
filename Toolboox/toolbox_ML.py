import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway, ttest_ind


def describe_df(df):

    """
    Genera un resumen informativo sobre las columnas de un DataFrame.

    El resumen incluye:
    - Tipo de dato de cada columna.
    - Porcentaje de valores nulos en cada columna.
    - Número de valores únicos en cada columna.
    - Porcentaje de cardinalidad (valores únicos en relación al total de filas).

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame del cual se generará el resumen.

    Retorna:
    --------
    pandas.DataFrame:
        Un DataFrame con una columna por cada columna del DataFrame original y cuatro filas:
        - 'tipo': El tipo de dato (dtype) de cada columna.
        - 'porcentaje_nulos': Porcentaje de valores nulos en cada columna.
        - 'valores_unicos': Número de valores únicos en cada columna.
        - 'porcentaje_cardinalidad': El porcentaje de valores únicos respecto al total de filas.
    
    Ejemplo de uso:
    ---------------
    df_resumen = resumen_dataframe(mi_dataframe)
    print(df_resumen)
    """
    # Crear un diccionario para almacenar la información
    resumen = {
        'Tipo de Dato': df.dtypes,
        '% Valores Nulos': df.isnull().mean() * 100,
        'Valores Únicos': df.nunique(),
        '% Cardinalidad': (df.nunique() / len(df)) * 100
    }
    
    # Crear un DataFrame a partir del diccionario
    resumen_df = pd.DataFrame(resumen)
    
    # Ajustar el formato de la salida (por ejemplo, redondear los porcentajes)
    resumen_df['% Valores Nulos'] = resumen_df['% Valores Nulos'].round(2)
    resumen_df['% Cardinalidad'] = resumen_df['% Cardinalidad'].round(2)
    
    return resumen_df.T



def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.90):

    """
    Esta función sugiere el tipo de cada columna del DataFrame basándose en la cardinalidad
    y en el porcentaje de valores únicos en relación al tamaño del DataFrame. Los tipos de
    variables sugeridos son: "Binaria", "Categórica", "Numérica Discreta" o "Numérica Continua".
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame sobre el que se realizará la clasificación de tipos de variables.
    
    umbral_categoria : int, opcional (por defecto 10)
        El umbral de cardinalidad. Si una columna tiene menos valores únicos que este
        umbral, se considera categórica. Si tiene más, se considera numérica.

    umbral_continua : float, opcional (por defecto 0.90)
        Umbral de porcentaje de cardinalidad sobre el tamaño del DataFrame. Si el
        porcentaje de valores únicos en la columna es superior o igual a este umbral
        y la cardinalidad supera el umbral de categorías, la variable se considera 
        "Numérica Continua". En caso contrario, se considera "Numérica Discreta".

    Retorna:
    --------
    pandas.DataFrame:
        Un DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido", que contiene 
        el nombre de la columna original y el tipo de variable sugerido.
    """
     # Asegúrate de que umbral_categoria es un entero
    try:
        umbral_categoria = int(umbral_categoria)
    except ValueError:
        print(f"Error: umbral_categoria debe ser un número, se recibió: {umbral_categoria}")
        return None

    # Inicializar una lista para almacenar el resultado
    sugerencias = []

    # Recorrer cada columna del DataFrame
    for col in df.columns:
        # Calcular la cardinalidad (número de valores únicos)
        cardinalidad = df[col].nunique()
        # Calcular el porcentaje de cardinalidad
        porcentaje_cardinalidad = (cardinalidad / len(df)) * 100
        
        # Determinar el tipo sugerido
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo_sugerido = "Numerica Continua"
        else:
            tipo_sugerido = "Numerica Discreta"
        
        # Añadir la sugerencia a la lista
        sugerencias.append({
            'nombre_variable': col,
            'tipo_sugerido': tipo_sugerido
        })
    
    # Convertir la lista de sugerencias en un DataFrame
    resultado_df = pd.DataFrame(sugerencias)
    
    return resultado_df


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    La función identifica columnas numéricas en un dataframe que estén correlacionadas
    con la columna objetivo (target_col), utilizando umbrales de correlación y p-value.

    Argumentos de la función:
    df (pd.DataFrame): dataframe que contiene los datos.
    target_col (str): nombre de la columna objetivo para el análisis de regresión.
    umbral_corr (float): valor absoluto mínimo de correlación para seleccionar las características.
    pvalue (float, opcional): filtra según la significancia de la correlación. Si es None, no se realiza este test.

    Retorna:
    list: Una lista con los nombres de las columnas que están correlacionadas con la columna objetivo por encima del umbral.
    """
    # Verifica que la columna objetivo existe y es numérica
    if target_col not in df.columns:
        print("Error: target_col debe ser una columna existente en el DataFrame.")
        return None

    # Verificar que la columna objetivo es numérica
    if df[target_col].dtype not in ['int64', 'float64']:
        print("Error: target_col debe ser numérica.")
        return None

    # Filtrar las columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Excluir la columna objetivo de las columnas a analizar
    numeric_cols = [col for col in numeric_cols if col != target_col]

    # Lista para almacenar las columnas seleccionadas.
    selected_columns = []

    # Para cada columna numérica, calcular la correlación con la columna objetivo
    for col in numeric_cols:
        correlacion, p_val = pearsonr(df[col].dropna(), df[target_col].dropna())
        
        # Si la correlación supera el umbral
        if abs(correlacion) >= umbral_corr:
            # Verificar si el p-value es significativo si se especifica
            if pvalue is not None:
                if p_val <= pvalue:
                    selected_columns.append(col)
            else:
                selected_columns.append(col)

    return selected_columns




def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Esta función genera gráficos para mostrar la relación entre variables numéricas
    y una variable objetivo numérica, dependiendo del nivel de correlación y el p-value.

    Argumentos:
    df (pd.DataFrame): dataframe con los datos.
    target_col (str): nombre de la columna objetivo.
    columns (list): lista de columnas a analizar.
    umbral_corr (float): valor mínimo de correlación.
    pvalue (float, opcional): filtra las columnas con un p-value significativo.

    Retorna:
    list: Lista con las columnas que están correlacionadas por encima del umbral.
    """

    # Paso 1: Verificar que 'target_col' está en el DataFrame
    if target_col not in df.columns:
        print("Error: La columna objetivo no está en el DataFrame.")
        return None

    # Paso 2: Verificar que 'target_col' es numérica
    if not (df[target_col].dtype == 'int64' or df[target_col].dtype == 'float64'):
        print("Error: La columna objetivo debe ser numérica (entero o decimal).")
        return None

    # Paso 3: Si no hay columnas especificadas, seleccionar todas las numéricas
    if not columns:
        columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

    # Paso 4: Excluir la columna objetivo de las columnas a analizar
    if target_col in columns:
        columns.remove(target_col)

    # Lista para guardar las columnas que cumplen con los criterios
    selected_columns = []

    # Paso 5: Calcular la correlación para cada columna numérica
    for col in columns:
        correlacion, p_val = pearsonr(df[col].dropna(), df[target_col].dropna())

        if abs(correlacion) >= umbral_corr:
            if pvalue is not None:
                if p_val <= pvalue:
                    selected_columns.append(col)
            else:
                selected_columns.append(col)

    # Crear gráficos pairplot en bloques
    for i in range(0, len(selected_columns), 5):
        subset_columns = [target_col] + selected_columns[i:i+5]
        sns.pairplot(df[subset_columns])
        plt.show()

    # Retornar las columnas seleccionadas
    return selected_columns





def get_features_cat_regression(df, target_col, pvalue=0.05, umbral_cardinalidad=10):
    '''
Esta función identifica las columnas categóricas en un DataFrame que tienen una 
    relación estadísticamente significativa con una columna numérica objetivo ('target_col'), 
    mediante pruebas estadísticas como ANOVA o la prueba t de Student.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame que contiene los datos.
    
    target_col : str
        Nombre de la columna numérica continua o discreta con alta cardinalidad 
        que será la variable objetivo del análisis.
    
    pvalue : float, opcional (por defecto 0.05)
        Umbral para el p-valor en las pruebas estadísticas. Si el p-valor es menor que 
        este umbral, la relación entre la variable categórica y la variable objetivo se 
        considerará estadísticamente significativa.
    
    umbral_cardinalidad : int, opcional (por defecto 10)
        Valor que define el límite máximo de valores únicos (cardinalidad) para 
        considerar una columna como categórica.

    Retorna:
    --------
    columnas_significativas : list
        Lista con los nombres de las columnas categóricas que tienen una relación
        estadísticamente significativa con la variable objetivo.

    Si los argumentos de entrada no son válidos, la función retorna `None` e imprime 
    un mensaje de error.
'''
    # Comprobaciones de entrada
    if target_col not in df.columns:
        print("Error: La columna 'target_col' no existe en el DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: La columna 'target_col' no es una variable numérica.")
        return None
    
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("Error: El valor de 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    if not isinstance(umbral_cardinalidad, int) or umbral_cardinalidad <= 0:
        print("Error: El 'umbral_cardinalidad' debe ser un entero positivo.")
        return None
    
    # Identificar columnas categóricas basadas en la cardinalidad
    columnas_categoricas = [col for col in df.columns if df[col].nunique() < umbral_cardinalidad]
    
    # Lista para almacenar las columnas que cumplen con el criterio
    columnas_significativas = []
    
    # Iterar sobre las columnas categóricas y realizar el test adecuado
    for col in columnas_categoricas:
        categorias = df[col].dropna().unique()
        
        if len(categorias) == 2:
            # Prueba t de Student para variables binarias
            grupo1 = df[df[col] == categorias[0]][target_col].dropna()
            grupo2 = df[df[col] == categorias[1]][target_col].dropna()
            stat, p = ttest_ind(grupo1, grupo2)
        elif len(categorias) > 2:
            # ANOVA para variables categóricas con más de dos categorías
            grupos = [df[df[col] == categoria][target_col].dropna() for categoria in categorias]
            stat, p = f_oneway(*grupos)
        else:
            continue  # Si la columna no tiene suficientes categorías, pasar a la siguiente
        
        # Verificar si el p-valor es menor que el umbral
        if p < pvalue:
            columnas_significativas.append(col)
    
    return columnas_significativas


def plot_features_cat_regression(df, target_col, columns, pvalue=0.05, with_individual_plot=False):
    """
    Esta función analiza las variables categóricas o numéricas de un DataFrame y pinta histogramas
    agrupados para la variable 'target_col' en función de las variables de 'columns', si el test 
    estadístico de relación entre ellas es significativo.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame que contiene los datos a analizar.

    target_col : str, opcional (por defecto "")
        La columna objetivo, debe ser numérica. Si no es una columna numérica, se imprime un mensaje de error.

    columns : list de str, opcional (por defecto [])
        Lista de columnas a evaluar. Si está vacía, se seleccionarán automáticamente las columnas numéricas del DataFrame.

    pvalue : float, opcional (por defecto 0.05)
        Umbral para el valor p. Las relaciones serán consideradas significativas si el p-valor es menor que este umbral.

    with_individual_plot : bool, opcional (por defecto False)
        Si es True, se pintarán los histogramas agrupados de 'target_col' para cada columna categórica seleccionada.

    Retorna:
    --------
    list:
        Lista de columnas que tienen una relación significativa con 'target_col' según el p-valor.

    Ejemplo de uso:
    ---------------
    df_result = analizar_variables(df, target_col='precio', columns=['marca', 'color'], pvalue=0.05, with_individual_plot=True)
    """
    
    # Verificaciones de entrada
    if target_col == "":
        print("Error: Debes especificar una columna objetivo 'target_col'.")
        return None
    
    if target_col not in df.columns:
        print(f"Error: La columna objetivo '{target_col}' no existe en el DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna objetivo '{target_col}' no es numérica.")
        return None
    
    if not isinstance(columns, list):
        print("Error: El argumento 'columns' debe ser una lista de strings.")
        return None
    
    if len(columns) == 0:
        # Si la lista está vacía, seleccionamos las columnas categóricas del DataFrame
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("Error: El valor de 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    columnas_significativas = []
    
    # Iterar sobre las columnas para analizar la relación con la variable 'target_col'
    for col in columns:
        if col not in df.columns:
            print(f"Advertencia: La columna '{col}' no existe en el DataFrame. Se omite.")
            continue
        
        # Eliminar los valores nulos de los datos
        df_clean = df[[col, target_col]].dropna()

        categorias = df_clean[col].unique()

        if len(categorias) < 2:
            print(f"Advertencia: La columna '{col}' tiene menos de 2 categorías, se omite.")
            continue
        
        # Si la columna tiene dos categorías, aplicamos la prueba t de Student
        if len(categorias) == 2:
            grupo1 = df_clean[df_clean[col] == categorias[0]][target_col]
            grupo2 = df_clean[df_clean[col] == categorias[1]][target_col]
            stat, p = ttest_ind(grupo1, grupo2, equal_var=False)  # Prueba t para dos grupos independientes
            print (p)
        
        # Si la columna tiene más de dos categorías, aplicamos ANOVA
        else:
            grupos = [df_clean[df_clean[col] == cat][target_col] for cat in categorias]
            stat, p = f_oneway(*grupos)
        
        # Si la relación es significativa, añadimos la columna a la lista
        if p < pvalue:
            columnas_significativas.append(col)
            
            # Si se debe plotear el histograma
            if with_individual_plot:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df_clean, x=target_col, hue=col, multiple="stack", kde=False)
                plt.title(f"Histograma de '{target_col}' agrupado por '{col}'")
                plt.show()

    # Retornar las columnas que tienen una relación significativa con 'target_col'
    return columnas_significativas