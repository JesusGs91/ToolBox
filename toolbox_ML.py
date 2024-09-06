def resumen_dataframe(df):

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



def sugerir_tipo_variable(df, umbral_categoria, umbral_continua):

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
    La función identifica columnas numéricas en un dataframe que esten correlacionadas
    con la columna objetivo (target_col), utilizando umbrales de correlación y el p-value.

    Argumentos de la función:
    df (pd.DataFrame): dataframe que contiene los datos.
    target_col (str): nombre de la columna objetivo para el análisis de regresión.
    umbral_corr (float): valor absoluto mínimo de correlación para seleccionar las características.
    pvalue (float y opcional): filtra según la significancia de la correlación. Si es None, no se realiza este test.

    Retorna:
    list: Una lista con los nombres de las columnas que están correlacionadas con la columna objetivo por encima del umbral.
    """
    # Verifica que la columna objetivo existe y es numérica. Si no lo es, mostramos un mensaje de error y terminamos la función.
    if target_col not in df.columns:
        print("Error: target_col debe ser una columna existente en el DataFrame.")
        return None
    
    # Obtener el resumen del DataFrame para verificar el tipo de dato de la columna objetivo
    resumen = resumen_dataframe(df)

    # Verificar que la columna objetivo es numérica
    if resumen[target_col]['Tipo de Dato'] not in ['int64', 'float64']:
        print("Error: target_col debe ser numérica.")
        return None
    
     # Usar la función sugerir_tipo_variable para obtener las columnas numéricas
    tipificacion = sugerir_tipo_variable(df, umbral_categoria=10, umbral_continua=0.05) #Umbral_categoria se puede modificar si tenemos menos valores únicos
    
    # Filtrar las columnas que son numéricas (continuas o discretas)
    numeric_cols = tipificacion[tipificacion['tipo_sugerido'].isin(['Numerica Continua', 'Numerica Discreta'])]['nombre_variable']
    
    # Excluir la columna objetivo de la lista de columnas a analizar
    numeric_cols = numeric_cols[numeric_cols != target_col]

    # Lista para almacenar las columnas seleccionadas.
    selected_columns = []

    # Para cada columna numérica, calcular la correlación con la columna objetivo
    for col in numeric_cols:
        correlacion, p_val = pearsonr(df[col], df[target_col]) #Se usa pearsonr para saber como de fuerte es la relación entre el tarjet y las columnas numéricas.
        
        # Si la correlación supera el umbral
        if abs(correlacion) >= umbral_corr: #abs se utiliza para identificar la magnitud de la relación. Con esto vemos lo fuerte que es la realación entre el tarjet y otra columna. Mayor o igual al umbral que definimos
            #Si la columna no está bien correlacionada la descartamos por no tener datos relevantes.
            if pvalue is not None: # Si se especifica pvalue, también verifica la significancia estadística
                if p_val <= (1 - pvalue): #Buscamos que si pvalue es menor o igual 1, la confianza es mayor al 95%.
                    selected_columns.append(col)
            else:
                selected_columns.append(col)

    return selected_columns



from scipy.stats import f_oneway, ttest_ind

def columnas_significativas(df, target_col, pvalue=0.05, umbral_cardinalidad=10):
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