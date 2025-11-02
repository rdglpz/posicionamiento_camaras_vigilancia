import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt  # necesario para show_surveillance_coverage
from skimage.draw import line    # se usa en isovista
import requests                   # (no se usa en este fragmento, pero lo dejo como en el original)

def gkern(kernlen: int = 101, nsig: float = 2) -> np.ndarray:
    """
    Genera un kernel Gaussiano 2D separable y normalizado.

    Parámetros
    ----------
    kernlen : int
        Tamaño del kernel (lado). Debe ser impar para centrado simétrico.
    nsig : float
        Número de desviaciones estándar que cubre el soporte del kernel.

    Retorna
    -------
    np.ndarray
        Kernel 2D Gaussiano normalizado (suma = 1).
    """
    # Puntos equiespaciados que cubren [-nsig, nsig]
    x = np.linspace(-nsig, nsig, kernlen + 1)
    # Diferencias del CDF normal → aproxima un kernel 1D
    kern1d = np.diff(st.norm.cdf(x))
    # Kernel 2D por producto externo (separable)
    kern2d = np.outer(kern1d, kern1d)
    # Normalización para que la suma total sea 1
    return kern2d / kern2d.sum()


def perimeter(img: np.ndarray) -> list[tuple[int, int]]:
    """
    Devuelve las coordenadas (fila, columna) del perímetro de una imagen/matriz.

    Parámetros
    ----------
    img : np.ndarray
        Matriz 2D (alto x ancho).

    Retorna
    -------
    list[tuple[int, int]]
        Lista con todas las celdas que forman el borde de la matriz.
    """
    perim = []

    # Borde superior (fila 0)
    for j in range(img.shape[1]):
        perim.append((0, j))

    # Borde inferior (última fila)
    for j in range(img.shape[1]):
        perim.append((img.shape[0] - 1, j))

    # Borde izquierdo (columna 0)
    for i in range(img.shape[0]):
        perim.append((i, 0))

    # Borde derecho (última columna)
    for i in range(img.shape[0]):
        perim.append((i, img.shape[1] - 1))

    return perim


def isovista(img: np.ndarray) -> np.ndarray:
    """
    Calcula una máscara de visibilidad estilo 'isovista' desde el centro del recorte.

    Para cada punto del perímetro, traza una línea (ray casting) desde el centro
    del recorte hasta ese punto. Si en el rayo encuentra el primer píxel con valor 1
    (considerado obstáculo/oclusor), anula (pone a 0) todos los píxeles a partir de
    ese impacto en esa misma línea.

    Supuestos:
    - La matriz `img` es cuadrada y de lado impar (para tener un centro exacto).
    - Los valores 1 en `img` representan paredes/oclusiones; el resto, espacio libre.

    Parámetros
    ----------
    img : np.ndarray
        Recorte/localidad de trabajo (p. ej., un submapa alrededor de un sensor).

    Retorna
    -------
    np.ndarray
        Máscara binaria (1 = visible, 0 = no visible) del mismo tamaño que `img`.
    """
    # Centro del recorte (asume lado impar)
    si = (int(img.shape[0] / 2), int(img.shape[1] / 2))

    # Comenzamos con todo visible (1)
    mask = np.ones(img.shape, dtype=float)

    # Coordenadas del perímetro (destinos de los rayos)
    edges = perimeter(img)

    # Para cada punto del borde, trazar una línea desde el centro
    for x in edges:
        rr, cc = line(si[0], si[1], x[0], x[1])  # coordenadas del rayo
        ray = img[rr, cc]                        # valores a lo largo del rayo

        # Índices donde hay obstáculo (valor == 1)
        ix = np.argwhere(ray == 1)

        if len(ix) != 0:
            # Primer impacto con una pared (índice en el rayo)
            ixw = ix[0][0]
            # Desde el impacto hacia afuera, anular visibilidad
            mask[rr[ixw:], cc[ixw:]] = 0

    return mask


def F(
    X: np.ndarray,
    S: np.ndarray,
    Walls: np.ndarray,
    CD: np.ndarray,
    L: int = 50,
    K: np.ndarray = None
) -> float:
    """
    Función objetivo para evaluar una configuración de sensores X.

    La idea:
    - Para cada sensor, se toma un recorte (ventana cuadrada de lado 2L+1) de:
      * S  : mapa binario de paredes/obstáculos (1 = pared)
      * CD : mapa de densidad de crimen (o interés) en la misma escala
    - Se calcula una máscara de visibilidad (isovista) en ese recorte.
    - Se pondera la visibilidad con un kernel Gaussiano K (prioriza cercanía).
    - Se acumula la cobertura 'óptima' entre sensores (máximo por celda).
    - Se añade un término de cercanía a pared (Walls) en las posiciones de sensores.

    Retorna el negativo de la combinación lineal de:
      alpha * (cobertura acumulada) + (1-alpha) * (cercanía a pared)

    Notas importantes:
    - Se asume que todos los sensores tienen al menos L celdas de margen alrededor.
      Si no, los recortes S[si[0]-L:si[0]+L+1, ...] se saldrán de los límites.
    - `S` debe tener paredes como 1 si se quiere que `isovista` ocluya.

    Parámetros
    ----------
    X : np.ndarray
        Arreglo de coordenadas (n_sensores x 2) o vector que se puede reshaped a (-1, 2).
    S : np.ndarray
        Mapa binario de paredes/obstáculos.
    Walls : np.ndarray
        Mapa (mismas dimensiones) con una métrica de "cercanía a pared" (o penalización/bono).
    CD : np.ndarray
        Mapa de densidad de crimen/interés.
    L : int
        Radio del recorte alrededor de cada sensor (ventana de tamaño 2L+1).
    K : np.ndarray
        Kernel Gaussiano (si es None, se genera con gkern(2L+1, 4)).

    Retorna
    -------
    float
        Valor a minimizar (negativo de la utilidad): más negativo = mejor cobertura.
    """
    X = X.astype(int)
    X_resh = X.reshape(-1, 2)
    n_sensors = len(X_resh)

    alpha = 1.0  # peso de la cobertura vs. cercanía a pared (0..1)
    if K is None:
        K = gkern(L * 2 + 1, 4)

    # Tensor para acumulación de coberturas por sensor (y una capa vacía inicial)
    COVERS = np.zeros((n_sensors + 1, S.shape[0], S.shape[1]), dtype=float)

    for i, x in enumerate(X_resh):
        si = tuple(x)

        # Recortes alrededor del sensor (asumen que no se salen de los límites)
        S_sub = np.copy(S[si[0] - L:si[0] + L + 1, si[1] - L:si[1] + L + 1])
        CD_sub = np.copy(CD[si[0] - L:si[0] + L + 1, si[1] - L:si[1] + L + 1])

        # Máscara de visibilidad desde el centro del recorte
        mask = isovista(S_sub)

        # Contribución de este sensor a la cobertura global (ponderada y ocluida)
        COVERS[i + 1, si[0] - L:si[0] + L + 1, si[1] - L:si[1] + L + 1] = CD_sub * K * mask

    # Sumar, por celda del mapa, la mayor cobertura lograda por cualquier sensor
    max_covers = np.sum(np.max(COVERS, axis=0))

    # Medida de cercanía a pared en las posiciones de los sensores
    cercania_pared = np.sum(Walls[X_resh[:, 0], X_resh[:, 1]])

    # Negativo de la utilidad (si se optimiza por minimización)
    return -(alpha * max_covers + (1 - alpha) * cercania_pared)


def show_surveillance_coverage(X: np.ndarray, L: int, S: np.ndarray, CD: np.ndarray) -> None:
    """
    Visualiza, para cada sensor en X, los recortes y productos intermedios:
      - S_sub + posición del sensor
      - Máscara de isovista
      - Isovista ponderada con K
      - Densidad de crimen recortada y ocluida
      - Producto final CD_sub * K * mask (y suma informativa)

    Notas:
    - Asume que las ventanas (2L+1) alrededor de cada sensor no se salen de los límites.
    - Usa matplotlib para mostrar una fila de 5 paneles por sensor.
    """
    X_resh = X.reshape(-1, 2).astype(int)

    for i, x in enumerate(X_resh):
        si = tuple(x)
        print(f"Sensor {i}: {si}")

        # Marcador del centro (sensor) para superponer en las imágenes
        Z = np.zeros((L * 2 + 1, L * 2 + 1))
        Z[(L, L)] = 1

        # Recortes locales
        S_sub = np.copy(S[si[0] - L:si[0] + L + 1, si[1] - L:si[1] + L + 1])
        CD_sub = np.copy(CD[si[0] - L:si[0] + L + 1, si[1] - L:si[1] + L + 1])

        # Cálculos intermedios
        mask = isovista(S_sub)
        K = gkern(L * 2 + 1, 4)

        # Visualización
        fig, axs = plt.subplots(1, 5, figsize=(12, 3), sharey=True)

        axs[0].imshow(S_sub + Z)
        axs[0].set_title("Mapa S (paredes) + sensor")

        axs[1].imshow(mask + Z)
        axs[1].set_title("Isovista")

        axs[2].imshow(mask * K)
        axs[2].set_title("Isovista ponderada (K)")

        axs[3].imshow(CD_sub * mask)
        axs[3].set_title("Densidad crimen (ocluida)")

        axs[4].imshow(CD_sub * K * mask + Z * np.max(CD_sub * mask))
        axs[4].set_title(f"IV ponderada x Dens.\nSuma={np.sum(CD_sub * K * mask):.2f}")

        plt.tight_layout()
        plt.show()

    # Nota: En el original se intentaba escribir en COVERS aquí, pero la
    # variable es local a F. Si quieres acumular aquí, declara y devuelve COVERS.
