"""Extrae la posición de cajas y urnas del mapa marcado por la comunidad.

Cada marcador es un cuadrito de color puro (~12 px en la imagen de 2000 px):
    Rojo     = caja en superficie      Amarillo = urna en superficie
    Azul     = caja subterránea        Verde    = urna subterránea
    Morado   = caja en azotea          Rosa     = urna en azotea

Uso:
    python extraer_nodos.py                 # usa mapas/todos.webp
    python extraer_nodos.py --debug         # además guarda salidas/deteccion.png
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).parent

# (tipo, capa, color RGB de referencia)
CLASES = {
    "rojo":     ("caja", "superficie", (255, 0, 0)),
    "amarillo": ("urna", "superficie", (254, 216, 0)),
    "azul":     ("caja", "subterraneo", (0, 149, 255)),
    "verde":    ("urna", "subterraneo", (94, 255, 1)),
    "morado":   ("caja", "azotea", (127, 0, 111)),
    "rosa":     ("urna", "azotea", (254, 127, 181)),
}
TOLERANCIA = 60          # distancia RGB máxima al color de referencia
# el ícono dorado del centro del mapa se parece al amarillo; se exige más pureza
TOLERANCIA_CLASE = {"amarillo": 35}
AREA_MIN, AREA_MAX = 30, 400
LADO_MARCADOR = 13       # lado aproximado de un cuadrito, para separar pegados


def componentes(mascara):
    """Etiquetado 4-conexo con BFS; devuelve listas de (y, x)."""
    visitado = np.zeros_like(mascara, dtype=bool)
    alto, ancho = mascara.shape
    grupos = []
    for y0, x0 in zip(*np.nonzero(mascara)):
        if visitado[y0, x0]:
            continue
        cola = deque([(y0, x0)])
        visitado[y0, x0] = True
        pix = []
        while cola:
            y, x = cola.popleft()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < alto and 0 <= nx < ancho and mascara[ny, nx] and not visitado[ny, nx]:
                    visitado[ny, nx] = True
                    cola.append((ny, nx))
        grupos.append(np.array(pix))
    return grupos


def partir(pix):
    """Si varios cuadritos quedaron unidos, los separa en una rejilla."""
    ys, xs = pix[:, 0], pix[:, 1]
    filas = max(1, round((ys.max() - ys.min() + 1) / LADO_MARCADOR))
    cols = max(1, round((xs.max() - xs.min() + 1) / LADO_MARCADOR))
    if filas == 1 and cols == 1:
        return [(xs.mean(), ys.mean())]
    centros = []
    fy = np.minimum((ys - ys.min()) * filas // (ys.max() - ys.min() + 1), filas - 1)
    fx = np.minimum((xs - xs.min()) * cols // (xs.max() - xs.min() + 1), cols - 1)
    for i in range(filas):
        for j in range(cols):
            sel = (fy == i) & (fx == j)
            if sel.sum() >= AREA_MIN:
                centros.append((xs[sel].mean(), ys[sel].mean()))
    return centros


def extraer(ruta_imagen):
    img = np.asarray(Image.open(ruta_imagen).convert("RGB")).astype(np.int32)
    alto, ancho, _ = img.shape
    nodos = []
    for color, (tipo, capa, ref) in CLASES.items():
        dist = np.sqrt(((img - np.array(ref)) ** 2).sum(-1))
        for pix in componentes(dist < TOLERANCIA_CLASE.get(color, TOLERANCIA)):
            if len(pix) < AREA_MIN:
                continue
            alto_c = np.ptp(pix[:, 0]) + 1
            ancho_c = np.ptp(pix[:, 1]) + 1
            # descarta manchas grandes y no rectangulares (íconos del mapa)
            if len(pix) > AREA_MAX * 6 or len(pix) < 0.5 * alto_c * ancho_c:
                continue
            for x, y in partir(pix):
                nodos.append({
                    "tipo": tipo, "capa": capa, "color": color,
                    # coordenadas normalizadas [0,1] para no depender de la resolución
                    "x": round(float(x) / ancho, 5), "y": round(float(y) / alto, 5),
                })
    nodos.sort(key=lambda n: (n["y"], n["x"]))
    for i, n in enumerate(nodos, start=1):
        n["id"] = i
    return nodos, (ancho, alto)


def guardar_debug(ruta_imagen, nodos, destino):
    img = Image.open(ruta_imagen).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for n in nodos:
        x, y = n["x"] * w, n["y"] * h
        d.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(255, 255, 255), width=2)
    img.save(destino)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagen", default=RAIZ / "mapas" / "todos.webp")
    p.add_argument("--salida", default=RAIZ / "datos" / "nodos.json")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    nodos, tam = extraer(args.imagen)
    Path(args.salida).parent.mkdir(exist_ok=True)
    Path(args.salida).write_text(json.dumps({"imagen": Path(args.imagen).name, "tam": tam, "nodos": nodos}, indent=1), encoding="utf-8")

    resumen = {}
    for n in nodos:
        clave = f'{n["tipo"]} {n["capa"]}'
        resumen[clave] = resumen.get(clave, 0) + 1
    print(f"{len(nodos)} nodos -> {args.salida}")
    for k, v in sorted(resumen.items()):
        print(f"  {k:22s} {v}")

    if args.debug:
        (RAIZ / "salidas").mkdir(exist_ok=True)
        guardar_debug(args.imagen, nodos, RAIZ / "salidas" / "deteccion.png")
        print("debug -> salidas/deteccion.png")


if __name__ == "__main__":
    main()
