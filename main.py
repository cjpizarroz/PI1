from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.responses import JSONResponse
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
# ------ usar 'ebi-hime' como dato para consulta
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


