# SoulPath

*Rutas de recolección de cajas en Deadlock con Slime Mould Algorithm*
(EN: *Crate-farming routes for Deadlock with the Slime Mould Algorithm*)

Proyecto de Paralelización de Algoritmos: ruta de recolección de cajas/urnas que maximiza
almas con un límite de tiempo (Orienteering Problem), resuelta con SMA (Li et al., 2020).

## Archivos

| Archivo | Qué hace |
|---|---|
| `mapas/*.webp` | Mapas marcados por la comunidad (Reddit). `todos.webp` tiene las 6 categorías. |
| `extraer_nodos.py` | Detecta los cuadritos de color del mapa y genera `datos/nodos.json` (602 nodos). |
| `config.json` | Parámetros del juego (velocidad, almas, tiempos) y del SMA. |
| `sma_deadlock.py` | Instancia (matriz C, almas S), decodificadores, SMA, línea base voraz. |
| `visor/index.html` | Visualización: ruta sobre el mapa, convergencia y reproducción por iteración. |

Leyenda de colores del mapa: rojo = caja superficie, amarillo = urna superficie,
azul = caja subterránea, verde = urna subterránea, morado = caja azotea, rosa = urna azotea.

## Uso

```bash
pip install -r requirements.txt
python extraer_nodos.py --debug          # solo si cambias el mapa; salidas/deteccion.png para revisar
python sma_deadlock.py --tmax 45 90 150  # una corrida por presupuesto de tiempo
```

Después abre `visor/index.html` en el navegador (doble clic basta). El visor está en español e
inglés: botones ES/EN arriba a la derecha, o `index.html?lang=en`. Los textos están en `TEXTOS`
dentro del mismo HTML.

Opciones útiles de `sma_deadlock.py`:

- `--decodificador estricto|salto|insercion|guiado` compara representaciones.
- `--repeticiones 10` corre con 10 semillas e imprime media, desviación, mejor y peor.
- `--procesos 4` evalúa la población en paralelo con `multiprocessing`.
- `--capas superficie subterraneo` ignora las azoteas, por ejemplo.
- `--iteraciones`, `--poblacion`, `--semilla` sobrescriben `config.json`.

## Modelo

- Nodo 0 = punto de inicio y regreso (`config.json > inicio`, coordenadas normalizadas 0–1).
- `c_ij` = caminar (distancia euclidiana × `ancho_mapa_m` × `tortuosidad` / `velocidad_ms`)
  + `penalizacion_capa_s` por cada nivel que se sube o baja + tiempo de romper la caja j.
- Maximizar Σ S_i sujeto a Σ c_ij ≤ t_max.

**Supuestos a calibrar:** el tamaño del mapa, la velocidad, las almas por caja/urna y los
tiempos son estimaciones. Mide en partida (p. ej. cuánto tardas de la base a medio mapa) y
ajusta `ancho_mapa_m` y `velocidad_ms`. La distancia es euclidiana: no conoce paredes, por eso
el factor `tortuosidad`.

## Decodificadores (vector de llaves X ∈ [0,1]^n → ruta)

| Nombre | Idea |
|---|---|
| `estricto` | El del artículo: ordenar llaves y agregar hasta la primera caja que no quepa. |
| `salto` | Igual, pero salta las que no caben y sigue intentando. |
| `insercion` | Orden por llave; cada caja se inserta en la posición más barata de la ruta. |
| `guiado` (default) | Inserción voraz por almas/tiempo, con la preferencia de cada caja multiplicada por e^(4·(X_j−0.5)). X = 0.5 reproduce el voraz; el SMA aprende qué cajas favorecer. |

Con 602 cajas, las llaves aleatorias puras (`estricto`, `insercion`) eligen cajas dispersas y
quedan por debajo del voraz. `guiado` conserva la representación RK-SMA y mejora al voraz.

Resultados de referencia (60 iteraciones, población 30, semilla 42):

| t_max | Voraz | SMA guiado | Mejora |
|---|---|---|---|
| 45 s | 55 | 70 | +27% |
| 90 s | 235 | 275 | +17% |
| 150 s | 450 | 535 | +19% |

## SMA implementado

Sigue a Li et al. (2020) adaptado a maximización: pesos W con `1 + r·log10(...)` para la mitad
buena y `1 − r·log10(...)` para la mala, `a = atanh(1 − t/T)`, `b = 1 − t/T`,
`p = tanh(|S_i − DF| / |DF|)` (normalizado porque las almas no están en [0,1]) y reinicio
aleatorio con probabilidad `z`.

## Paralelización (siguiente entrega)

El costo está en `evaluar_poblacion`: decodificar cada agente es independiente, así que es
paralelismo de datos directo. `--procesos N` ya lo reparte con `multiprocessing.Pool`
(≈2.2× con 4 procesos en la corrida de 90 s). Otros candidatos: repeticiones independientes en
paralelo, o el cálculo vectorizado de inserciones en GPU.

## Equipo

María Fernanda Barrón Álvarez, Leslie Miroslava Benítez Marín y Leonardo Merino Garfias.

## Agradecimientos

- **u/Awsmminer** (Reddit), por mapear a mano todas las cajas y urnas del mapa; los datos salen de su trabajo.
- **Kenney**, por [Fantasy UI Borders](https://kenney.nl/assets/fantasy-ui-borders) (CC0). Las piezas
  originales están en `visor/assets/kenney/`; `python visor/generar_ui.py` las tiñe con la paleta del
  visor y las guarda en `visor/assets/ui/`. Para probar otra pieza copia su `panel-NNN.png` y
  `panel-border-NNN.png` del pack y cambia el número en `PIEZAS`.

Tipografías: Cinzel Decorative y Marcellus (Google Fonts, OFL).
