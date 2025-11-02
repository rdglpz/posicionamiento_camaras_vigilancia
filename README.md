# Crime Visibility Index

Índice de Visibilidad Criminal (Criem Visibility Index CVI)

El Índice de Visibilidad Criminal (CVI) clasifica la idoneidad de cualquier posible ubicación de cámara utilizando datos bidimensionales.
El IVC se emplea para identificar las n ubicaciones principales que, con base en su configuración espacial y la incidencia delictiva, requerirían cámaras de vigilancia.

Mapa de Sensibilidad de Detección Criminal

Depende de las siguientes tres variables espaciales:

La cercanía a la posición de la cámara, descrita mediante un kernel Gaussiano (K).

La densidad delictiva (D), obtenida al aplicar un kernel Gaussiano a la imagen ráster de delitos georreferenciados.

El rango visible de cada cámara, descrito por su isovista (V), que es un mapa binario.

![cvi_example](https://github.com/rdglpz/posicionamiento_camaras_vigilancia/blob/main/imgs/CVI_example.png?raw=true)

La calificación de cada cámara se obtiene como la suma de los valores de la imagen R.

El Índice de Visibilidad Criminal describe la capacidad teórica de observación de delitos de una red de cámaras de vigilancia.

La documentación de la colección de funciones propias que calculan el CVI para cada punto georreferenciado se encuentra en este repositorio dentro de la carpeta:
[src/camera_allocation_functions.py](https://github.com/rdglpz/posicionamiento_camaras_vigilancia/blob/main/src/camera_allocation_functions.py).

El ejemplo de implementación y optimización de la red de cámaras de vigilancia está en el cuaderno:
notebooks/optimize_camera_allocation.ipynb

Los mapas generados representan la red de cámaras de vigilancia cuasi-óptima, considerando los delitos ocurridos en 2018 en el centro de Aguascalientes.

![](https://github.com/rdglpz/posicionamiento_camaras_vigilancia/blob/main/imgs/output_example.png?raw=true)



Referenncias:

Tapia-McClung, R., & Lopez-Farías, R. (2024). An Approach for Spatial Optimization on Positioning Surveillance Cameras. In O. Gervasi, B. Murgante, C. Garau, D. Taniar, A. M. A. C. Rocha, & M. N. Faginas Lago (Eds.), Computational Science and Its Applications – ICCSA 2024 Workshops (Lecture Notes in Computer Science, Vol. 14819, pp. 366-380). Springer, Cham. https://dl.acm.org/doi/abs/10.1007/978-3-031-65282-0_24




