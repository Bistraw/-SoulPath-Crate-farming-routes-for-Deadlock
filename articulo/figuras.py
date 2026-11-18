"""Figuras del artículo a partir de los datos del proyecto.

    fig_nodos.png         nodos extraídos sobre el mapa, coloreados por capa
    fig_rutas.png         ruta voraz vs ruta SMA guiado (t_max = 90 s)
    fig_convergencia.svg  mejor valor promedio por iteración, por decodificador
    fig_paralelo.svg      tiempo de cómputo según número de procesos

Requiere salidas/experimentos.json (python experimentos.py).
Uso:  python articulo/figuras.py
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
from sma_deadlock import Instancia, sma, voraz  # noqa: E402

COLOR_CAPA = {"superficie": (230, 50, 40), "subterraneo": (30, 140, 255), "azotea": (170, 70, 200)}
LADO = 1000


def fuente(tam):
    for nombre in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def fondo():
    img = Image.open(RAIZ / "mapas" / "superficie.webp").convert("RGB").resize((LADO, LADO))
    # se aclara y desatura para que los marcadores destaquen en impresión
    gris = img.convert("L").convert("RGB")
    return Image.blend(gris, Image.new("RGB", gris.size, (255, 255, 255)), 0.55)


def marcar(d, n, r, color, contorno=None):
    x, y = n["x"] * LADO, n["y"] * LADO
    caja = (x - r, y - r, x + r, y + r)
    if n["tipo"] == "urna":
        d.ellipse(caja, fill=color, outline=contorno)
    else:
        d.rectangle(caja, fill=color, outline=contorno)


def fig_nodos(nodos, inicio):
    img = fondo()
    d = ImageDraw.Draw(img)
    for n in nodos:
        marcar(d, n, 4, COLOR_CAPA[n["capa"]])
    x, y = inicio["x"] * LADO, inicio["y"] * LADO
    d.polygon([(x, y - 12), (x + 12, y), (x, y + 12), (x - 12, y)], fill=(0, 0, 0))
    leyenda(d, [("Superficie", COLOR_CAPA["superficie"]), ("Subterráneo", COLOR_CAPA["subterraneo"]),
                ("Azotea", COLOR_CAPA["azotea"]), ("Inicio (base)", (0, 0, 0))])
    img.save(AQUI / "fig_nodos.png", optimize=True)


def leyenda(d, items, x0=20, y0=20):
    f = fuente(22)
    alto = 34 * len(items) + 16
    d.rectangle((x0, y0, x0 + 230, y0 + alto), fill=(255, 255, 255), outline=(60, 60, 60))
    for i, (texto, color) in enumerate(items):
        y = y0 + 14 + 34 * i
        d.rectangle((x0 + 12, y + 2, x0 + 32, y + 22), fill=color)
        d.text((x0 + 44, y), texto, fill=(20, 20, 20), font=f)


def dibujar_ruta(d, inst, nodos, inicio, ruta, color, ancho):
    puntos = [(inicio["x"] * LADO, inicio["y"] * LADO)]
    puntos += [(nodos[i - 1]["x"] * LADO, nodos[i - 1]["y"] * LADO) for i in ruta]
    if inst.regreso:
        puntos.append(puntos[0])
    d.line(puntos, fill=color, width=ancho, joint="curve")


def fig_rutas(nodos, inicio, cfg):
    inst = Instancia(nodos, dict(cfg, tmax=90))
    v = voraz(inst)
    s = sma(inst, "guiado", cfg["sma"]["poblacion"], 100, cfg["sma"]["z"], 0, 4, verbose=False)
    img = fondo()
    d = ImageDraw.Draw(img)
    for n in nodos:
        marcar(d, n, 3, (95, 95, 95))
    dibujar_ruta(d, inst, nodos, inicio, v["ruta"], (240, 120, 20), 5)
    dibujar_ruta(d, inst, nodos, inicio, s["ruta"], (20, 120, 40), 4)
    for i in set(v["ruta"]) | set(s["ruta"]):
        marcar(d, nodos[i - 1], 6, COLOR_CAPA[nodos[i - 1]["capa"]], (0, 0, 0))
    x, y = inicio["x"] * LADO, inicio["y"] * LADO
    d.polygon([(x, y - 12), (x + 12, y), (x, y + 12), (x - 12, y)], fill=(0, 0, 0))
    # recorte a la región de ambas rutas; la leyenda se dibuja después para que no se pierda
    img = img.crop(recorte([v["ruta"], s["ruta"]], nodos, inicio))
    leyenda(ImageDraw.Draw(img), [(f"Voraz: {v['almas']:.0f} almas", (240, 120, 20)),
                                  (f"SMA: {s['almas']:.0f} almas", (20, 120, 40)),
                                  ("No visitado", (95, 95, 95))], x0=img.width - 240, y0=10)
    img.save(AQUI / "fig_rutas.png", optimize=True)
    return {"voraz": v, "sma": {k: s[k] for k in ("almas", "tiempo", "ruta")}}


def recorte(rutas, nodos, inicio, margen=70):
    xs = [inicio["x"]] + [nodos[i - 1]["x"] for r in rutas for i in r]
    ys = [inicio["y"]] + [nodos[i - 1]["y"] for r in rutas for i in r]
    x0, x1 = min(xs) * LADO - margen, max(xs) * LADO + margen
    y0, y1 = min(ys) * LADO - margen, max(ys) * LADO + margen
    # incluir la leyenda (esquina superior izquierda) y mantener proporción cercana a 1:1
    lado = max(x1 - x0, y1 - y0, 420)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = max(0, cx - lado / 2), max(0, cy - lado / 2)
    return (int(x0), int(y0), int(min(LADO, x0 + lado)), int(min(LADO, y0 + lado)))


def svg_lineas(series, titulo_y, titulo_x, ancho=640, alto=390, linea_base=None):
    """series: lista de (etiqueta, color, valores, discontinua). La leyenda va en una fila bajo el eje."""
    m = {"l": 64, "r": 16, "t": 16, "b": 82}
    todos = [v for _, _, vals, _ in series for v in vals] + ([linea_base[2]] if linea_base else [])
    lo, hi = min(todos), max(todos)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    n = max(len(vals) for _, _, vals, _ in series)
    X = lambda i: m["l"] + i / (n - 1) * (ancho - m["l"] - m["r"])
    Y = lambda v: alto - m["b"] - (v - lo) / (hi - lo) * (alto - m["t"] - m["b"])
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" font-family="Times New Roman, serif" font-size="14">']
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        p.append(f'<line x1="{m["l"]}" x2="{ancho - m["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>')
        p.append(f'<text x="{m["l"] - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for k in range(6):
        i = round((n - 1) * k / 5)
        p.append(f'<text x="{X(i):.1f}" y="{alto - m["b"] + 18}" text-anchor="middle">{i + 1}</text>')
    p.append(f'<line x1="{m["l"]}" x2="{m["l"]}" y1="{m["t"]}" y2="{alto - m["b"]}" stroke="#000"/>')
    p.append(f'<line x1="{m["l"]}" x2="{ancho - m["r"]}" y1="{alto - m["b"]}" y2="{alto - m["b"]}" stroke="#000"/>')
    p.append(f'<text x="{(ancho + m["l"]) / 2}" y="{alto - m["b"] + 38}" text-anchor="middle">{titulo_x}</text>')
    p.append(f'<text transform="translate(16 {(alto - m["b"]) / 2}) rotate(-90)" text-anchor="middle">{titulo_y}</text>')
    if linea_base:
        etiqueta, color, v = linea_base
        series = series + [(etiqueta, color, [v] * n, True)]
    for etiqueta, color, vals, disc in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        dash = ' stroke-dasharray="6 4"' if disc else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    paso = (ancho - m["l"]) / len(series)
    for k, (etiqueta, color, _, disc) in enumerate(series):
        y = alto - 14
        x = m["l"] + k * paso
        dash = ' stroke-dasharray="6 4"' if disc else ""
        p.append(f'<line x1="{x}" x2="{x + 26}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"{dash}/>')
        p.append(f'<text x="{x + 32}" y="{y + 5}">{etiqueta}</text>')
    p.append("</svg>")
    return "\n".join(p)


def fig_convergencia(exp):
    r = exp["decodificadores"]["res"]
    colores = {"estricto": "#999999", "salto": "#c07a20", "insercion": "#3a6fb0", "guiado": "#1d7a35"}
    nombres = {"estricto": "Estricto", "salto": "Salto", "insercion": "Inserción", "guiado": "Guiado"}
    series = [(nombres[k], colores[k], r[k]["historia_mejor_media"], False) for k in colores]
    svg = svg_lineas(series, "Almas (mejor global, promedio)", "Iteración",
                     linea_base=("Voraz", "#000000", exp["decodificadores"]["voraz"]))
    (AQUI / "fig_convergencia.svg").write_text(svg, encoding="utf-8")


def fig_paralelo(exp):
    res = exp["paralelo"]["res"]
    ancho, alto, m = 640, 300, {"l": 64, "r": 16, "t": 20, "b": 52}
    hi = max(x["segundos"]["media"] for x in res) * 1.15
    bw = (ancho - m["l"] - m["r"]) / len(res)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" font-family="Times New Roman, serif" font-size="14">']
    Y = lambda v: alto - m["b"] - v / hi * (alto - m["t"] - m["b"])
    base = res[0]["segundos"]["media"]
    for i, x in enumerate(res):
        v = x["segundos"]["media"]
        x0 = m["l"] + i * bw + bw * 0.2
        p.append(f'<rect x="{x0:.1f}" y="{Y(v):.1f}" width="{bw * 0.6:.1f}" height="{alto - m["b"] - Y(v):.1f}" fill="#3a6fb0"/>')
        p.append(f'<text x="{x0 + bw * 0.3:.1f}" y="{Y(v) - 6:.1f}" text-anchor="middle">{v:.1f} s (×{base / v:.2f})</text>')
        p.append(f'<text x="{x0 + bw * 0.3:.1f}" y="{alto - m["b"] + 18}" text-anchor="middle">{x["procesos"]}</text>')
    p.append(f'<line x1="{m["l"]}" x2="{ancho - m["r"]}" y1="{alto - m["b"]}" y2="{alto - m["b"]}" stroke="#000"/>')
    p.append(f'<text x="{(ancho + m["l"]) / 2}" y="{alto - 10}" text-anchor="middle">Número de procesos</text>')
    p.append(f'<text transform="translate(18 {(alto - m["b"]) / 2}) rotate(-90)" text-anchor="middle">Tiempo de cómputo (s)</text>')
    p.append("</svg>")
    (AQUI / "fig_paralelo.svg").write_text("\n".join(p), encoding="utf-8")


def main():
    cfg = json.loads((RAIZ / "config.json").read_text(encoding="utf-8"))
    nodos = json.loads((RAIZ / "datos" / "nodos.json").read_text(encoding="utf-8"))["nodos"]
    fig_nodos(nodos, cfg["inicio"])
    exp = json.loads((RAIZ / "salidas" / "experimentos.json").read_text(encoding="utf-8"))
    fig_convergencia(exp)
    fig_paralelo(exp)
    rutas = fig_rutas(nodos, cfg["inicio"], cfg)
    (AQUI / "rutas_figura.json").write_text(json.dumps(rutas, indent=1), encoding="utf-8")
    print("figuras ->", AQUI, "| voraz", rutas["voraz"]["almas"], "sma", rutas["sma"]["almas"])


if __name__ == "__main__":
    main()
