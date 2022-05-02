"""
Análisis de Sentimientos usando Naive Bayes
-----------------------------------------------------------------------------------------

El archivo `amazon_cells_labelled.txt` contiene una serie de comentarios sobre productos
de la tienda de amazon, los cuales están etiquetados como positivos (=1) o negativos (=0)
o indterminados (=NULL). En este taller se construirá un modelo de clasificación usando
Naive Bayes para determinar el sentimiento de un comentario.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """

    # Lea el archivo `amazon_cells_labelled.tsv` y cree un DataFrame usando pandas.
    # Etiquete la primera columna como `msg` y la segunda como `lbl`. Esta función
    # retorna el dataframe con las dos columnas.
    df = pd.read_csv(
        'amazon_cells_labelled.tsv',
        sep='\t',
        header=None,
        names=['message','label'],
    )
    #print(df['message'].__class__)
    #print(df.head(6))
    #df["label"].value_counts()

    # Separe los grupos de mensajes etiquetados y no etiquetados.
    #http://exponentis.es/como-encontrar-valores-nan-en-un-dataframe-python-pandas-y-modificarlos
    df_untagged = df[df.isnull().any(1)]
    df_tagged = df.dropna()
    
    #df_tagged = ____[____["____"].____()]
    #df_untagged = ____[____["____"].____()]

    x_tagged = df_tagged["message"]
    y_tagged = df_tagged["label"]

    x_untagged = df_untagged["message"]
    y_untagged = df_untagged["label"]
    #print(y_untagged.shape)

    # Retorne los grupos de mensajes
    return x_tagged, y_tagged, x_untagged, y_untagged


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos generados en la pregunta 01.
    x_tagged, y_tagged, x_untagged, y_untagged = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use el 10% de patrones para la muestra de prueba.
    x_train, x_test, y_train, y_test = train_test_split(
        x_tagged,
        y_tagged,
        test_size=0.1,
        random_state=12345,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return x_train, x_test, y_train, y_test


def pregunta_03():
    """
    Construcción de un analizador de palabras
    -------------------------------------------------------------------------------------
    """
    # Importe el stemmer de Porter
    # Importe CountVectorizer
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer

    # Cree un stemeer que use el algoritmo de Porter.
    stemmer = PorterStemmer()

    # Cree una instancia del analizador de palabras (build_analyzer)
    analyzer = vectorizer().built_analyzer()

    # Retorne el analizador de palabras
    return lambda x: (stemmer.stem(w) for w in analyzer(x))


def pregunta_04():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe CountVetorizer
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe BernoulliNB
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import BernoulliNB

    # Cargue las variables.
    x_train, x_test, y_train, y_test = pregunta_02()

    # Obtenga el analizador de la pregunta 3.
    analyzer = pregunta_03()

    # Cree una instancia de CountVectorizer que use el analizador de palabras
    # de la pregunta 3. Esta instancia debe retornar una matriz binaria. El
    # límite superior para la frecuencia de palabras es del 100% y un límite
    # inferior de 5 palabras. Solo deben analizarse palabras conformadas por
    # letras.
    countVectorizer = CountVectorizer(
        analyzer= "word",
        lowercase= True,
        stop_words="english",
        token_pattern=r"[a-z]+",
        #token_pattern=r"(?u)\b\w\w+\b",
        binary=True,
        max_df=1.0,
        min_df=5,
    )

    # Cree un pipeline que contenga el CountVectorizer y el modelo de BernoulliNB.
    #https://gist.github.com/trungngv/5a157abf22ad130e83db4a5bde96b92a
    pipeline = Pipeline(
        steps=[
            ("cnt", countVectorizer),
            ("Bernoulli", BernoulliNB()),
        ],
    )

    # Defina un diccionario de parámetros para el GridSearchCV. Se deben
    # considerar 10 valores entre 0.1 y 1.0 para el parámetro alpha de
    # BernoulliNB.
    param_grid = {
        "Bernoulli": np.random(0.1, 1.0, 10),
    }

    # Defina una instancia de GridSearchCV con el pipeline y el diccionario de
    # parámetros. Use cv = 5, y "accuracy" como métrica de evaluación
    gridSearchCV = GridSearchCV(
        estimator= pipeline,
        param_grid= param_grid,
        cv=5,
        scoring="accuracy",
        #refit = True (valor de inicialización predeterminado), una vez que se encuentre el mejor modelo
        #(estimador) durante la validación cruzada, se volverá a entrenar en todo el conjunto de entrenamiento
        refit=True,
        return_train_score=False,
    )

    # Búsque la mejor combinación de regresores
    gridSearchCV.fit(x_train, y_train)

    # Retorne el mejor modelo
    return gridSearchCV


def pregunta_05():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe confusion_matrix
    from sklearn.metrics import confusion_matrix

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred= gridSearchCV.predict_proba(X_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=gridSearchCV.predict_proba(X_test),
    )

    # Retorne la matriz de confusion de entrenamiento y prueba
    return cfm_train, cfm_test


def pregunta_06():
    """
    Pronóstico
    -------------------------------------------------------------------------------------
    """

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue los datos generados en la pregunta 01.
    X_tagged, y_tagged, X_untagged, y_untagged = pregunta_01()

    # pronostique la polaridad del sentimiento para los datos
    # no etiquetados
    y_untagged_pred = X_untagged.predict(y_untagged)

    # Retorne el vector de predicciones
    return y_untagged_pred
