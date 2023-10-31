

*<h1>Aprendizaje automático e ingeniería de datos</h1>*

Sobre el proyecto
Basado en datos de Steam, plataforma líder en distribución de juegos para PC.
Este proyecto tiene como objetivo: ser un sistema de recomendación de juegos eficaz y personalizado utilizando técnicas de análisis de datos y aprendizaje automático. 
El objetivo es impulsar las ventas, proporcionando informacion relevante de los usuarios. Establecer sugerencias de juegos más relevantes, mejorar la satisfacción del usuario, aumentar la participación.


<h2>Construido con:</h2>

pandas
Phyton
PyArrow
API rápida

<h2>Empezando</h2>

Este es un ejemplo de cómo puede dar instrucciones sobre cómo configurar su proyecto localmente. Para poner en funcionamiento una copia local, siga estos sencillos pasos de ejemplo.

<h2>Requisitos previos:</h2>

pip install requirements.txt

Instalación

Clonar el repositorio

git clone https://github.com/cjpizarroz/PI1

Ejecutar main.py

Ingrese su API enlocalhost

https://localhost:8000

<h2>Uso</h2>
Host local FastAPI

<h2>Uso web</h2>
https://data16-pi1.onrender.com/

<h1>Metodología de Desarrollo del Sistema de Recomendación:</h2>

En el proceso de desarrollo y despliegue de un proyecto, a menudo nos enfrentamos a desafíos técnicos y limitaciones de recursos que pueden influir en las decisiones que tomamos. Una de esas limitaciones a menudo cruciales es la capacidad de la plataforma de renderización (render) que utilizamos para alojar nuestro proyecto.

La "renderización" se refiere al proceso de ejecutar y mostrar una aplicación en un servidor remoto para que los usuarios finales puedan acceder a ella a través de la web. Sin embargo, estas plataformas de renderización pueden tener restricciones en términos de recursos informáticos, como la cantidad de potencia de CPU y memoria RAM disponibles. Estas restricciones pueden llevar a una reducción o "achicamiento" de los datos, lo que significa que debemos optimizar y reducir el tamaño de nuestros recursos, como bases de datos o modelos de machine learning, para que se ajusten a estas limitaciones.

*Achicar los datos puede implicar:*

Reducción de la Granularidad: Eliminar detalles innecesarios o redundantes de los datos para simplificar su estructura.

Compresión de Datos: Utilizar algoritmos de compresión para reducir el tamaño de los datos sin perder información crítica.

Muestreo: En lugar de utilizar todo el conjunto de datos, trabajar con una muestra representativa de él.

Optimización de Modelos: Si estamos utilizando modelos de machine learning, podemos optimizarlos para que sean más livianos en términos de recursos.

Resolución Baja: Si trabajamos con imágenes o medios visuales, reducir la resolución puede ser una opción.

La razón detrás de esta reducción de datos radica en la necesidad de que nuestro proyecto sea viable en la plataforma de renderización sin comprometer su rendimiento. Si nuestros datos son demasiado grandes para ser manejados por los recursos limitados de la plataforma de renderización, esto podría llevar a problemas como tiempos de carga lentos o bloqueos del sistema.

En resumen, achicar los datos es una estrategia clave para abordar las limitaciones de las plataformas de renderización y garantizar que nuestro proyecto sea accesible y funcione sin problemas para los usuarios finales, incluso cuando se despliega en un entorno con recursos limitados.

Recopilación de Datos:
Se obtuvieron amplios conjuntos de datos de Steam, que incluyeron información de usuarios, detalles de juegos, reseñas y recomendaciones. Se emplearon API y técnicas de web scraping para recopilar estos datos.

Limpieza de Datos:
Una vez recopilados, los datos se sometieron a un proceso de limpieza para eliminar información redundante o no deseada. Se abordaron datos duplicados, corrección de errores y se excluyeron datos de usuarios que no cumplían los criterios del proyecto.

Transformación de Datos:
Los datos se transformaron para facilitar su análisis. Esto incluyó la conversión de tipos de datos, normalización y la creación de nuevas variables a partir de las existentes.

Carga de Datos:
Una vez preparados, los datos se cargaron en un conjunto de datos de destino, como un archivo en formato parquet con compresión gzip.

Exploración de Datos:
Se exploraron los datos en profundidad para identificar patrones y tendencias en el comportamiento de los usuarios. Se utilizaron herramientas de visualización y técnicas de minería de datos para descubrir correlaciones y patrones.

Ingeniería de Características:
En esta etapa, se crearon nuevas variables a partir de las existentes para mejorar la precisión del modelo. Esto incluyó variables relacionadas con la interacción del usuario con los juegos, la popularidad de los juegos y las estadísticas de los desarrolladores.

<h1>Modelado:</h1>
Se implementaron técnicas de aprendizaje automático para desarrollar un modelo capaz de predecir la probabilidad de que un usuario juegue un juego específico. Se utilizaron algoritmos de aprendizaje supervisado, como regresión logística y similitud de cosenos.

Evaluación del Modelo:
Una vez desarrollado, se evaluó la precisión del modelo utilizando métricas adecuadas, como precisión, recuperación y puntuación F1. Se aplicaron técnicas de validación cruzada para asegurar la generalización del modelo a nuevos datos.

Implementación del Modelo:
El modelo se desplegó en un entorno de producción, permitiendo predicciones en tiempo real. Se utilizó una arquitectura de microservicios para garantizar escalabilidad y eficiencia.

Desarrollo del Sistema de Recomendación:
Se diseñó un sistema que analiza los datos de Steam para ofrecer recomendaciones de juegos personalizadas basadas en el comportamiento y preferencias de cada usuario.

Pruebas y Validación:
El sistema de recomendación se sometió a pruebas exhaustivas y validación, empleando métricas de evaluación apropiadas para medir precisión y utilidad.

Resultados Esperados:
Se espera obtener un sistema de recomendación de juegos personalizado para cada usuario de Steam, mejorando su experiencia en la plataforma. Esto conduce a una mayor satisfacción de los usuarios, un mayor compromiso con la plataforma y un proceso más eficiente de recomendación de juegos, lo que se traduce en un aumento de ventas e ingresos para Steam.

Este enfoque permitirá brindar a los usuarios de Steam recomendaciones de juegos más precisas y satisfactorias, mejorando su experiencia en la plataforma.

( volver arriba )

<h1>Contribuyendo</h1>
Las contribuciones son el alma de la comunidad de código abierto, y cada aporte es valioso y apreciado. Aquí hay varias formas en las que puedes contribuir:

Bifurcación y Solicitud de Extracción (Pull Request): Si tienes una mejora o corrección para el proyecto, bifurca el repositorio principal, realiza tus cambios en tu bifurcación y crea una solicitud de extracción. Esto permite que tus contribuciones se revisen y se integren en el proyecto principal.

Apertura de Problemas: Si encuentras errores, problemas o tienes sugerencias de mejora, puedes abrir un problema en el repositorio. Proporcionar detalles sobre el problema que has encontrado o la mejora propuesta es de gran ayuda.

Mejora y Documentación: Contribuir a la documentación del proyecto es igual de importante. Si encuentras partes de la documentación que necesitan ser actualizadas o mejoradas, o si puedes proporcionar ejemplos y guías adicionales, estas contribuciones son muy valiosas.

Colaboración en Discusiones: Participar en discusiones relacionadas con el proyecto es una forma de contribución. Compartir tus conocimientos y perspectivas puede ayudar a resolver problemas o impulsar ideas innovadoras.

Difusión y Estrellas: Darle una estrella al proyecto en la plataforma en la que se encuentra (por ejemplo, en GitHub) es una forma sencilla de mostrar apoyo al proyecto y ayudar a que otros lo descubran.

Contribuciones de Código: Si eres un desarrollador, puedes contribuir con mejoras de código al proyecto. Asegúrate de seguir las pautas de contribución y los estándares del proyecto.

Traducción: Si el proyecto tiene una audiencia internacional, contribuir con traducciones al idioma localizado puede ser muy útil para hacerlo accesible a más personas.

Pruebas y Retroalimentación: Probar el proyecto y proporcionar retroalimentación sobre su usabilidad y cualquier problema que encuentres es otra forma de contribuir.

Cualquiera que sea la forma en que decidas contribuir, ten en cuenta que tu participación es valiosa y apreciada por la comunidad de código abierto. ¡Gracias por ser parte de esta comunidad increíble!

</h1>Contacto</h1>
Javier PIzarro [pizarrocarlosjavier@gmail.com](mailto:pizarrocarlosjavier@gmail.com)






