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
@app.get("/get_data/")
async def endpoints_1_Desarrollador(id):
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
async def endpoints_2_User_id(id):
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

@app.get("/get_dataep3/")
async def endpoints_3_UserforGenre(id):
    dataset = query_data3(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data3(id: str):
    try:
        genero = id
        dataframe = pd.read_csv('CSV//consulta3.csv', sep=',', encoding='UTF-8')
        df_filtrado = dataframe[dataframe[genero] == 1]
        del dataframe          # libero recursos
        gc.collect()

        # Paso 2: Encontrar el usuario con más tiempo jugado
        usuario_mas_tiempo = df_filtrado[df_filtrado["playtime_forever"] == df_filtrado["playtime_forever"].max()]
        usuario_mas_tiempo = usuario_mas_tiempo[['user_id', 'playtime_forever']]
        # Paso 3: Agrupar por año y sumar el tiempo jugado
        acumulacion_por_anio = df_filtrado.groupby("release_date")["playtime_forever"].sum().reset_index()

        # Paso 4: Crear una lista de la acumulación de horas jugadas por año
        acumulacion_por_anio_list = acumulacion_por_anio.values.tolist()
        resp_usuario = usuario_mas_tiempo['user_id'].values[0]
        horas_totales = usuario_mas_tiempo['playtime_forever'].values[0].astype(str)

        return {"Usuario con más tiempo jugado para": genero,
                "Usuario: ":resp_usuario,
                "Horas totales jugadas: ":horas_totales,
                "Acumulación de horas jugadas por año":acumulacion_por_anio_list
                }
    except Exception as e:
        return {"error": str(e)}
    


#---------- END POINT NRO 4 --------------
# ------ usar '2019' como dato para consulta

@app.get("/get_dataep4/")
async def endpoints_4_best_developer_year(id):
    dataset = query_data4(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)
        
    
def query_data4(id: int):
    try:
        anio_consulta = id
        columnas = ['release_date','item_id', 'developer']
        desarrolladores_df = pd.read_csv('CSV//output_steaam_games.csv', sep=',', usecols=columnas, encoding='UTF-8')

        columnas = ['item_id', 'recommend']
        recomendaciones_df = pd.read_csv('CSV//australian_user_reviews.csv', usecols=columnas, sep=',', encoding='UTF-8')

        # Pasamos a variables dummies el contenido de recommend
        dummies = pd.get_dummies(recomendaciones_df['recommend'], prefix='recommend')
        # Concatenar las variables dummies al DataFrame original y eliminar la columna original
        recomendaciones_df = pd.concat([recomendaciones_df, dummies], axis=1)
        recomendaciones_df.drop('recommend', axis=1, inplace=True)

        recomendaciones_df.drop(columns='recommend_False', inplace=True)

        recomendaciones_df = recomendaciones_df.rename(columns={'recommend_True': 'recomendacion_positiva'})

        recomendaciones_df['recomendacion_positiva'] = recomendaciones_df['recomendacion_positiva'].astype(int)

        # Filtrar el DataFrame de recomendaciones para el año de consulta
        desarrollador_anio = desarrolladores_df[desarrolladores_df['release_date'] == anio_consulta]

        # Agrupar y contar las recomendaciones por item_id
        recomendaciones_contadas = recomendaciones_df.groupby('item_id')['recomendacion_positiva'].sum().reset_index()

        # Unir los DataFrames de desarrolladores y recomendaciones
        desarrolladores_y_recomendaciones = desarrolladores_df.merge(recomendaciones_contadas, on='item_id', how='left')
        desarrolladores_y_recomendaciones.drop_duplicates(inplace=True)

        # Ordenar los desarrolladores por el número de recomendaciones en orden descendente
        desarrolladores_ordenados = desarrolladores_y_recomendaciones.sort_values(by='recomendacion_positiva', ascending=False)


        # Obtener los tres desarrolladores con más recomendaciones
        top_desarrolladores = desarrolladores_ordenados.head(3)
        
        return top_desarrolladores.to_dict(orient='records')
      
    except Exception as e:
        return {"error": str(e)}



