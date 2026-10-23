"""Tiñe las piezas blancas de Kenney (Fantasy UI Borders, CC0) con la paleta del visor.

Cada pieza se compone de dos capas del pack: "panel-NNN" (relleno) y "panel-border-NNN"
(contorno). Se pintan de colores distintos, se apilan y se escalan 2x sin suavizado.
Salida en assets/ui/, usada por index.html con border-image.

Uso:  python visor/generar_ui.py
"""
from pathlib import Path

from PIL import Image

AQUI = Path(__file__).parent
FUENTE = AQUI / "assets" / "kenney"
DESTINO = AQUI / "assets" / "ui"
ESCALA = 2

ORO = (212, 176, 106)
ORO_CLARO = (245, 214, 140)
ORO_OSCURO = (122, 100, 60)
FONDO_CARD = (24, 28, 28)
FONDO_BOTON = (37, 43, 43)

# nombre de salida: (pieza, color de relleno o None, color de contorno)
PIEZAS = {
    "card":        ("027", FONDO_CARD, ORO),
    "marco":       ("014", None, ORO),
    "stat":        ("010", (18, 22, 22), ORO_OSCURO),
    "boton":       ("015", FONDO_BOTON, ORO_OSCURO),
    "boton-hover": ("015", FONDO_BOTON, ORO_CLARO),
    "boton-on":    ("015", ORO, ORO_CLARO),
}
DIVISORES = {"divisor": "divider-fade-003", "linea": "divider-fade-002"}


def teñir(ruta, color):
    img = Image.open(ruta).convert("RGBA")
    capa = Image.new("RGBA", img.size, color + (255,))
    capa.putalpha(img.getchannel("A"))
    return capa


def escalar(img):
    return img.resize((img.width * ESCALA, img.height * ESCALA), Image.NEAREST)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, (num, relleno, contorno) in PIEZAS.items():
        base = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        if relleno:
            base.alpha_composite(teñir(FUENTE / f"panel-{num}.png", relleno))
        base.alpha_composite(teñir(FUENTE / f"panel-border-{num}.png", contorno))
        escalar(base).save(DESTINO / f"{nombre}.png")
    for nombre, archivo in DIVISORES.items():
        img = escalar(teñir(FUENTE / f"{archivo}.png", ORO))
        img.save(DESTINO / f"{nombre}.png")
        img.transpose(Image.FLIP_LEFT_RIGHT).save(DESTINO / f"{nombre}-izq.png")
    print("piezas ->", DESTINO)


if __name__ == "__main__":
    main()
