from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import Response
from fastapi.encoders import jsonable_encoder 
import logging
import gc
import json


app = FastAPI()


#---------- END POINT NRO 1 --------------
# ------ usar 'ebi-hime' como dato para consulta

@app.get("/get_data_ep1/")
async def desarrollador(id_desarrollador: str):  # Agregar anotación de tipo y descripción
    """
    ***Esta función devuelve datos según el ID del desarrollador proporcionado***.<br>
    ***param id_desarrollador:***       El ID del desarrollador para realizar la consulta.<br>
    ***return:***      Datos consultados correspondientes al ID del desarrollador.
    """
    dataset = query_desarrollador(id_desarrollador)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
    
def query_desarrollador(id_desarrollador):
    try:
                
        columns = ['release_date','item_id', 'developer','genres_Free to Play']
        df = pd.read_csv('CSV//output_steaam_games.csv', usecols=columns, sep=",", encoding="UTF-8")
        
        desarrollador = id_desarrollador
        df_desarrollador = df[df['developer'] == desarrollador]

        
        del df          # libero recursos
        gc.collect()

        # Agrupar los datos por "Año" y contar la cantidad de "item" por año
        df_consulta= df_desarrollador.groupby('release_date')['item_id'].nunique().reset_index()
        df_consulta.rename(columns={'item_id': 'Cantidad de articulos', 'release_date':'Año'}, inplace=True)

        # Agrupar los datos por "Año" y contar la cantidad de "item" por año
        item_free_to_play = df_desarrollador.groupby('release_date')['genres_Free to Play'].sum().reset_index()
        df_consulta['Contenido gratis'] = item_free_to_play['genres_Free to Play']/df_consulta['Cantidad de articulos']

        del desarrollador   # libero recursos
        gc.collect()

        # Formatear la columna 'Porcentaje' como porcentaje
        df_consulta['Contenido gratis'] = df_consulta['Contenido gratis'].apply(lambda x: '{:.2%}'.format(x))

        return df_consulta.to_dict(orient='records')
        
    except Exception as e:
        return {"error": str(e)}



#---------- END POINT NRO 2 --------------
# ------ usar 'I_DID_911_JUST_SAYING' como dato para consulta
@app.get("/get_data_ep2/")
async def userdata(User_id):
    """
    ***Esta función devuelve datos según el ID del Usuario proporcionado***.<br>
    ***param User_id:***       El ID del Usuario para realizar la consulta.<br>
    ***return:***      Datos consultados correspondientes al ID del usuario.
    """
    dataset = query_userdata(User_id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
    
def query_userdata(User_id):
    try:
        dataframe = pd.read_csv('CSV//df_ep_2.csv', sep=',', encoding='UTF-8')
        usuario_data = dataframe[dataframe['user_id'] == User_id]
    
        dinero_gastado = usuario_data['price'].sum()
    
        total_juegos = len(usuario_data)
    
        porcentaje_recomendacion_negativa = (usuario_data['recommend_False'].sum() / total_juegos) * 100
        porcentaje_recomendacion_positiva = (usuario_data['recommend_True'].sum() / total_juegos) * 100
        porcentaje_recomendacion_negativa = '{:.0f} %'.format(porcentaje_recomendacion_negativa)
        porcentaje_recomendacion_positiva = '{:.0f} %'.format(porcentaje_recomendacion_positiva)
    
        return {
            "dinero_gastado": dinero_gastado,
            "porcentaje_recomendacion_negativa": porcentaje_recomendacion_negativa,
            "porcentaje_recomendacion_positiva": porcentaje_recomendacion_positiva,
            "total_juegos": total_juegos
        }
        
    except Exception as e:
        return {"error": str(e)}
    

#---------- END POINT NRO 3 --------------
# ------ usar 'genres_Action' como dato para consulta

@app.get("/get_data_ep3/")
async def UserforGenre(genero: str):
    """
    ***Esta función devuelve datos según el genero del juego proporcionado***.<br>
    ***param genero:***       El genero del juego para realizar la consulta.<br>
    ***return:***      Datos consultados correspondientes al genero del juego.
    """
    dataset = query_data3(genero)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data3(genero):
    try:
        ep3_items_columns = ['user_id', 'item_id', 'playtime_forever']
        df_users_items = pd.read_csv('CSV//australian_users_items.csv', usecols=ep3_items_columns, sep=',', encoding='UTF-8')

        df_steam_games = pd.read_csv('CSV//output_steaam_games.csv', sep=',', encoding='UTF-8')

        df_merged3 = df_users_items.merge(df_steam_games, on='item_id')
        df_merged3['item_id'] = df_merged3['item_id'].astype(int)
        df_merged3['release_date'] = df_merged3['release_date'].astype(int)
        df_merged3['playtime_forever'] = df_merged3['playtime_forever'].astype(int)
        df_merged3['genres_Action'] = df_merged3['genres_Action'].astype(int)
        df_merged3['genres_Adventure'] = df_merged3['genres_Adventure'].astype(int)
        df_merged3['genres_Animation and Modeling'] = df_merged3['genres_Animation and Modeling'].astype(int)
        
        # Verificar qué columnas tienen solo valores de cero
        zero_columns = df_merged3.columns[df_merged3.eq(0).all()].to_list()
        
        # Eliminamos las columnas que solo tienen valores 0
        df_merged3.drop(columns=zero_columns, inplace=True)

        df_filtrado = df_merged3[df_merged3[genero] == 1]

        # Agrupa por usuario y suma las horas de juego
        resumen = df_filtrado.groupby('user_id')['playtime_forever'].sum()

        usuario_mas_horas = resumen.idxmax()
        max_horas_jugadas = resumen.max()

        df1 = df_filtrado[df_filtrado['user_id'] == usuario_mas_horas]

        # Agrupa por usuario y año, y suma las horas de juego
        resumen = df1.groupby(['user_id', 'release_date'])['playtime_forever'].sum().reset_index()

        # Convierte el resultado en una lista de horas acumuladas por año
        horas_acumuladas_por_año = []
        for _, row in resumen.iterrows():
            horas_acumuladas_por_año.append({"Año": row["release_date"], "Horas": row["playtime_forever"]})

        return {
            "Usuario con más horas jugadas para Género " + genero: usuario_mas_horas,
            "Horas jugadas": horas_acumuladas_por_año
        }
    except Exception as e:
        return {"error": str(e)}
    


#---------- END POINT NRO 4 --------------
# ------ usar '2019' como dato para consulta

@app.get("/get_data_ep4/")
async def best_developer_year(anio):
    dataset = query_data4(anio)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data4(anio: int):
    try:
        columnas = ['posted year','item_id', 'recommend']
        recomendaciones_df = pd.read_csv('CSV//australian_user_reviews.csv', usecols=columnas, sep=',', encoding='UTF-8')
        desarrollador_anio = recomendaciones_df[recomendaciones_df['posted year'] == anio]
        desarrollador_anio = desarrollador_anio.drop(columns=['posted year'])
       
        desarrollador_anio = desarrollador_anio[desarrollador_anio['recommend'] == True]

        # Agrupa por "item_id" y cuenta el número de repeticiones en cada grupo
        desarrollador_anio['total_repeticiones'] = desarrollador_anio.groupby('item_id')['item_id'].transform('count')

        del recomendaciones_df
        gc.collect()


        columnas = ['item_id', 'developer']
        desarrolladores_df = pd.read_csv('CSV//output_steaam_games.csv', sep=',', usecols=columnas, encoding='UTF-8')
        
         # Unir los DataFrames de desarrolladores y desarrollador_anio
        desarrolladores_y_recomendaciones = desarrolladores_df.merge(desarrollador_anio, on='item_id', how='left')

        del desarrolladores_df
        del desarrollador_anio
        gc.collect()

        desarrolladores_y_recomendaciones = desarrolladores_y_recomendaciones[desarrolladores_y_recomendaciones['recommend'] == True]

        #Agrupaños por developer y sumamos las recomendaciones
        recomendaciones_contadas = desarrolladores_y_recomendaciones.groupby('developer')['recommend'].sum().reset_index()

        del desarrolladores_y_recomendaciones
        gc.collect()

        # Ordenar los desarrolladores por el número de recomendaciones en orden descendente
        desarrolladores_ordenados = recomendaciones_contadas.sort_values(by='recommend', ascending=False)

        # Seleccionamos los primeros tres con recomendaciones mas elevadas
        top_desarrolladores = desarrolladores_ordenados.head(3)
        del desarrolladores_ordenados
        gc.collect()

        resultado = top_desarrolladores.to_dict(orient='records')
        
        return resultado
    except Exception as e:
        return {"error": str(e)}








#---------- END POINT NRO 5 --------------
# ------ usar 'Poppermost Productions' como dato para consulta

@app.get("/get_data_ep5/")
async def desarrollador_reviews_analysis(id):
    dataset = query_data5(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data5(id: str):
    try:
        desarrollador = id
        columnas = ['item_id', 'sentiment_analysis']
        reviews_df = pd.read_csv('CSV//australian_user_reviews.csv', sep=',', usecols=columnas, encoding='UTF-8')    

        columnas = ['item_id', 'developer']
        desarrollador_df = pd.read_csv('CSV//output_steaam_games.csv', sep=',', usecols=columnas, encoding='UTF-8')

        df_merged5 = desarrollador_df.merge(reviews_df, on='item_id')

        # Filtrar el DataFrame por el nombre del desarrollador
        desarrollador_df = df_merged5[df_merged5['developer'] == desarrollador]

        # Contar la cantidad de 0, 1 y 2 en la columna de sentimiento
        conteo_sentimientos = desarrollador_df['sentiment_analysis'].value_counts().to_dict()

        # Crear una lista a partir del diccionario de conteo
        conteo_lista = [conteo_sentimientos.get(0, 0), conteo_sentimientos.get(1, 0), conteo_sentimientos.get(2, 0)]
    
        # Agregar un título antes de cada elemento de la lista
        titulos = ["Negativo: ", "Neutral: ", "Positivo: "]
        
        return {desarrollador: [f"{titulo}: {conteo}" for titulo, conteo in zip(titulos, conteo_lista)]}
      
    except Exception as e:
        return {"error": str(e)}
    


    
#---------- END POINT NRO 6 --------------
# ------ usar '' como dato para consulta

@app.get("/get_data_ep6/")
async def ML(id):
    dataset = query_data6(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data6(id: str):
    try:
        
        import numpy as np
        import seaborn as sns
        
        import pandas as pd
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from keras.models import Sequential
        from keras.layers import Dense

        df_merge = pd.read_csv('CSV/df_modelo.csv', sep=',', encoding='UTF-8')

        X = df_merge[['t_recom','t_sent_neg','t_sent_pos']]  # Denotamos X con mayúscula ya que
                                                             # incluye más de un atributo
        y = df_merge.title # Etiqueta a predecir


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Crear un objeto codificador
        encoder = OneHotEncoder()
        encoder = OneHotEncoder(handle_unknown='ignore')


        # Ajustar y transformar las etiquetas del conjunto de entrenamiento
        encoded_labels_train = encoder.fit_transform([[label] for label in y_train])

        # Ajustar y transformar las etiquetas del conjunto de prueba
        encoded_labels_test = encoder.transform([[label] for label in y_test])

        # El resultado será una matriz dispersa (sparse matrix)
        # Si deseas obtener un array NumPy, puedes hacerlo de la siguiente manera:
        encoded_labels_train_array = encoded_labels_train.toarray()
        encoded_labels_test_array = encoded_labels_test.toarray()



        # Definir la arquitectura del modelo
        model = Sequential()
        model.add(Dense(64, input_dim=3, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2050, activation='softmax'))  # 4 neuronas para las 4 clases

        # Compilar el modelo
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Entrenar el modelo
        model.fit(X_train, encoded_labels_train_array, epochs=3, batch_size=32)
        # Evaluar el modelo
        loss, accuracy = model.evaluate(X_test, encoded_labels_test_array)

        new_title_data = df_merge.loc[df_merge['title'] == id, ['t_recom', 't_sent_neg', 't_sent_pos']].values
        # Realizar la predicción
        y_pred = model.predict(new_title_data)
        # Hacer predicciones
        y_pred = model.predict(X_test)

        # Encuentra el índice del título recomendado en cada fila de y_pred
        recommended_indices = np.argmax(y_pred, axis=1)

        # Obtiene los nombres correspondientes a los índices
        recommended_names = [encoder.categories_[0][i] for i in recommended_indices]
        unique_recommended_names = list(set(recommended_names))
        
        return {"Juego: " : unique_recommended_names
        }
      
    except Exception as e:
        return {"error": str(e)}





