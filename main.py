from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder 


app = FastAPI()


#---------- END POINT NRO 1 --------------
# ------ usar 'ebi-hime' como dato para consulta
@app.get("/get_data")
async def get_data(id):
    dataset = query_data(id)  # Realiza la consulta para obtener el conjunto de datos
    return JSONResponse(content=dataset)

def query_data(id):
    desarrollador = id
    #'Poolians.com'
    columns = ['release_date','item_id', 'developer','genres_Free to Play']
    df = pd.read_csv('Datasets\CSV\output_steaam_games.csv', usecols=columns, sep=",", encoding="UTF-8")
    df_desarrollador = df[df['developer'] == desarrollador]

    # Agrupar los datos por "Año" y contar la cantidad de "item" por año
    item_por_ano = df_desarrollador.groupby('release_date')['item_id'].nunique().reset_index()
    df_consulta = item_por_ano
    df_consulta.rename(columns={'item_id': 'Cantidad de articulos', 'release_date':'Año'}, inplace=True)

    # Agrupar los datos por "Año" y contar la cantidad de "item" por año
    item_free_to_play = df_desarrollador.groupby('release_date')['genres_Free to Play'].sum().reset_index()
    df_consulta['Contenido gratis'] = item_free_to_play['genres_Free to Play']/item_por_ano['Cantidad de articulos']

    # Formatear la columna 'Porcentaje' como porcentaje
    df_consulta['Contenido gratis'] = df_consulta['Contenido gratis'].apply(lambda x: '{:.2%}'.format(x))

    
    return df_consulta.to_dict(orient='records')






