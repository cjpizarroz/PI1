from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import Response
from fastapi.encoders import jsonable_encoder 
import logging
import gc


app = FastAPI()


#---------- END POINT NRO 1 --------------
# ------ usar 'ebi-hime' como dato para consulta
@app.get("/get_data_ep1/")
async def desarrollador(id):
    dataset = query_data(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
    
def query_data(id):
    try:
        desarrollador = id
        
        columns = ['release_date','item_id', 'developer','genres_Free to Play']
        df = pd.read_csv('CSV//output_steaam_games.csv', usecols=columns, sep=",", encoding="UTF-8")
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
async def userdata(id):
    dataset = query_data_ep2(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
    
def query_data_ep2(id):
    try:
        dataframe = pd.read_csv('CSV//df_ep_2.csv', sep=',', encoding='UTF-8')
        usuario_data = dataframe[dataframe['user_id'] == id]
    
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
async def UserforGenre(id):
    dataset = query_data3(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data3(id: str):
    try:
        genero = id
        ep3_items_columns = ['user_id', 'item_id', 'playtime_forever']
        df_users_items = pd.read_csv('CSV\\australian_users_items.csv',usecols=ep3_items_columns, sep=',', encoding='UTF-8')

        ep3_items_columns1 = ['item_id','release_date','genres_Action', 'genres_Adventure', 'genres_Animation and Modeling']
        df_steam_games = pd.read_csv('CSV\\output_steaam_games.csv', usecols=ep3_items_columns1, sep=",", encoding="UTF-8")

        df_merged3 = df_users_items.merge(df_steam_games, on='item_id')

        df_merged3['release_date'] = df_merged3['release_date'].map(int)
        df_merged3.reset_index(drop=True, inplace=True)
                               
        df_filtrado = df_merged3[df_merged3[genero] == 1]
        
        # Paso 2: Encontrar el usuario con más tiempo jugado
        usuario_mas_tiempo = df_filtrado[df_filtrado["playtime_forever"] == df_filtrado["playtime_forever"].max()]
        usuario_mas_tiempo = usuario_mas_tiempo[['user_id', 'playtime_forever']]

        # Paso 3: Agrupar por año y sumar el tiempo jugado
        acumulacion_por_anio = df_filtrado.groupby("release_date")["playtime_forever"].sum().reset_index()

        # Paso 4: Crear una lista de la acumulación de horas jugadas por año
        acumulacion_por_anio_list = acumulacion_por_anio.values.tolist()
        resp_usuario = usuario_mas_tiempo['user_id'].values[0]
        horas_totales = usuario_mas_tiempo['playtime_forever'].values[0]
       
       
        return {"Usuario con más tiempo jugado para": genero,
                "Usuario: ":resp_usuario,
                "Horas totales jugadas: ":horas_totales,
                "Acumulación de horas jugadas por año":acumulacion_por_anio_list
                }
    except Exception as e:
        return {"error": str(e)}
    


#---------- END POINT NRO 4 --------------
# ------ usar '2019' como dato para consulta

@app.get("/get_data_ep4/")
async def best_developer_year(id):
    dataset = query_data4(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data4(id: int):
    try:
        anio_consulta = id
        columnas = ['posted year','item_id', 'recommend']
        recomendaciones_df = pd.read_csv('CSV\\australian_user_reviews.csv', usecols=columnas, sep=',', encoding='UTF-8')
        desarrollador_anio = recomendaciones_df[recomendaciones_df['posted year'] == anio_consulta]
        desarrollador_anio = desarrollador_anio.drop(columns=['posted year'])
        
        del recomendaciones_df
        gc.collect()

        columnas = ['item_id', 'developer']
        desarrolladores_df = pd.read_csv('CSV\\output_steaam_games.csv', sep=',', usecols=columnas, encoding='UTF-8')
        # Unir los DataFrames de desarrolladores y desarrollador_anio
        desarrolladores_y_recomendaciones = desarrolladores_df.merge(desarrollador_anio, on='item_id', how='left')

        del desarrolladores_df
        del desarrollador_anio
        gc.collect()

        # Pasamos a variables dummies el contenido de recommend
        dummies = pd.get_dummies(desarrolladores_y_recomendaciones['recommend'], prefix='recommend')
        # Concatenar las variables dummies al DataFrame original y eliminar la columna original
        desarrolladores_y_recomendaciones = pd.concat([desarrolladores_y_recomendaciones, dummies], axis=1)
        desarrolladores_y_recomendaciones.drop('recommend', axis=1, inplace=True)

        desarrolladores_y_recomendaciones.drop(columns='recommend_False', inplace=True)

        desarrolladores_y_recomendaciones['recommend_True'] = desarrolladores_y_recomendaciones['recommend_True'].astype(int)
        #Eliminamos de desarrolladoresy recomendacion positiva los registros con valores NaN
        desarrolladores_y_recomendaciones = desarrolladores_y_recomendaciones.dropna(subset=['recommend_True'])
        #Agrupaños por developer y sumamos las recomendaciones
        recomendaciones_contadas = desarrolladores_y_recomendaciones.groupby('developer')['recommend_True'].sum().reset_index()

        del desarrolladores_y_recomendaciones
        gc.collect()
        # Ordenar los desarrolladores por el número de recomendaciones en orden descendente
        desarrolladores_ordenados = recomendaciones_contadas.sort_values(by='recommend_True', ascending=False)

        # Seleccionamos los primeros tres con recomendaciones mas elevadas
        top_desarrolladores = desarrolladores_ordenados.head(3)
        del desarrolladores_ordenados
        gc.collect()

        primero = top_desarrolladores.iloc[0,0]
        segundo = top_desarrolladores.iloc[1,0]
        tercero = top_desarrolladores.iloc[2,0]

        return {'Puesto 1: ': primero,
                'Puesto 2: ': segundo,
                'Puesto 3: ': tercero
                
                }
      
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
        reviews_df = pd.read_csv('CSV\\australian_user_reviews.csv', sep=',', usecols=columnas, encoding='UTF-8')    

        columnas = ['item_id', 'developer']
        desarrollador_df = pd.read_csv('CSV\\output_steaam_games.csv', sep=',', usecols=columnas, encoding='UTF-8')

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




