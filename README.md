# Crime Visibility Index

The Criminal Visibility Index (CVI) rank the suitability of any possible camera location using bi dimensional data. The CVI is used to identify the top $n$ locations that, based on their spatial configuration and criminal incidence, would require surveillance cameras.

The Criminal Detection Sensitivity Map

It depends on the following three spatial variables:
The closeness to the camera position described by a Gaussian kernel (K)
The crime density as a result of applying a Gaussian kernel to the raster image of geo-referenced crimes (D).
The visible range of each camera described by its isovist, which is a binary map (V).`

![cvi_example](https://github.com/rdglpz/posicionamiento_camaras_vigilancia/blob/main/imgs/CVI_example.png?raw=true)

The rank of each camera is the sum of the values of $R$ image.

The Criminal Visibility index describes the theoretical capacity of observing crimes of a surveillance camera network.

The documentation of the collection of functions that calculates the CVI of each georeferenced point is in this repostiroy inside `src` folder:

Link: ```https://github.com/rdglpz/posicionamiento_camaras_vigilancia/blob/main/src/camera_allocation_functions.py```

Example of the implementation and optimization of the surveillance camera networks is in the notebook `notebooks/optimize_camera_allocation.ipynb`

References:

Tapia-McClung, R., & Lopez-Farías, R. (2024). An Approach for Spatial Optimization on Positioning Surveillance Cameras. In O. Gervasi, B. Murgante, C. Garau, D. Taniar, A. M. A. C. Rocha, & M. N. Faginas Lago (Eds.), Computational Science and Its Applications – ICCSA 2024 Workshops (Lecture Notes in Computer Science, Vol. 14819, pp. 366-380). Springer, Cham. https://dl.acm.org/doi/abs/10.1007/978-3-031-65282-0_24




