"""Slime Mould Algorithm (SMA) secuencial para rutas de cajas en Deadlock.

Modelo: Orienteering Problem de un solo agente.
    Maximizar  Z = sum(S_i, i en R)      sujeto a   sum(c_ij, (i,j) en R) <= t_max
Representación: random keys (RK-SMA). Cada agente es un vector X en [0,1]^n; ordenar
las llaves de mayor a menor da la prioridad con que se intentan agregar las cajas.

Uso:
    python sma_deadlock.py                       # usa config.json
    python sma_deadlock.py --tmax 60 90 150      # varias corridas, una por t_max
    python sma_deadlock.py --repeticiones 10     # estadística con semillas distintas
    python sma_deadlock.py --procesos 4          # evaluación de la población en paralelo
Los resultados se guardan en salidas/ y en visor/resultados.js (para el visor HTML).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).parent
NIVEL_CAPA = {"subterraneo": 0, "superficie": 1, "azotea": 2}
EPS = 1e-12


# ----------------------------------------------------------------------------
# Instancia: matriz de tiempos C y almas S. El nodo 0 es el punto de inicio.
# ----------------------------------------------------------------------------
class Instancia:
    def __init__(self, nodos, cfg):
        ini = cfg["inicio"]
        xy = np.array([[ini["x"], ini["y"]]] + [[n["x"], n["y"]] for n in nodos])
        nivel = np.array([NIVEL_CAPA["superficie"]] + [NIVEL_CAPA[n["capa"]] for n in nodos])

        # tiempo de caminar (distancia euclidiana corregida por paredes/rodeos)
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        metros = np.hypot(dx, dy) * cfg["ancho_mapa_m"] * cfg["tortuosidad"]
        caminar = metros / cfg["velocidad_ms"]
        # subir a azoteas / bajar al subterráneo cuesta tiempo extra
        cambio_capa = np.abs(nivel[:, None] - nivel[None, :]) * cfg["penalizacion_capa_s"]
        # c_ij incluye romper la caja j al llegar
        romper = np.array([0.0] + [cfg["tiempo_romper_s"][n["tipo"]] for n in nodos])

        self.C = caminar + cambio_capa + romper[None, :]
        np.fill_diagonal(self.C, 0.0)
        self.S = np.array([0.0] + [float(cfg["almas"][n["tipo"]]) for n in nodos])
        self.n = len(nodos)
        self.tmax = float(cfg["tmax"])
        self.regreso = bool(cfg["regresar_al_inicio"])
        # cajas que ni solas caben en el presupuesto nunca se consideran
        ida_vuelta = self.C[0, 1:] + (self.C[1:, 0] if self.regreso else 0)
        self.alcanzable = np.concatenate([[False], ida_vuelta <= self.tmax])

    def tiempo_ruta(self, ruta):
        sec = [0] + list(ruta) + ([0] if self.regreso else [])
        return float(sum(self.C[a, b] for a, b in zip(sec[:-1], sec[1:])))

    def almas_ruta(self, ruta):
        return float(self.S[list(ruta)].sum()) if ruta else 0.0


# ----------------------------------------------------------------------------
# Decodificadores: vector de llaves X -> ruta factible (lista de nodos 1..n)
# ----------------------------------------------------------------------------
def decodificar_estricto(X, inst):
    """Tal cual el artículo: agrega en orden de llave y se detiene en la primera que no cabe."""
    return _secuencial(X, inst, saltar=False)


def decodificar_salto(X, inst):
    """Igual, pero si una caja no cabe la salta y sigue intentando con las siguientes."""
    return _secuencial(X, inst, saltar=True)


def _secuencial(X, inst, saltar):
    C, ruta, t, ult = inst.C, [], 0.0, 0
    for j in np.argsort(-X) + 1:
        if not inst.alcanzable[j]:
            continue
        vuelta = C[j, 0] if inst.regreso else 0.0
        if t + C[ult, j] + vuelta > inst.tmax:
            if saltar:
                continue
            break
        t += C[ult, j]
        ruta.append(j)
        ult = j
    return ruta


def decodificar_insercion(X, inst):
    """La llave decide la prioridad; cada caja se inserta en la posición más barata de la ruta."""
    C = inst.C
    sec = [0, 0] if inst.regreso else [0]
    t = 0.0
    for j in np.argsort(-X) + 1:
        if not inst.alcanzable[j]:
            continue
        a = np.array(sec[:-1])
        b = np.array(sec[1:])
        delta = C[a, j] + C[j, b] - C[a, b]
        if not inst.regreso:  # en ruta abierta también se puede agregar al final
            delta = np.append(delta, C[sec[-1], j])
        p = int(np.argmin(delta))
        if t + delta[p] <= inst.tmax:
            sec.insert(p + 1, j)
            t += delta[p]
    return sec[1:-1] if inst.regreso else sec[1:]


GAMMA_GUIADO = 4.0   # cuánto puede la llave alterar la preferencia voraz (factor e^±2)


def decodificar_guiado(X, inst):
    """Inserción voraz almas/tiempo donde la llave X_j multiplica la preferencia por la caja j.

    Con X = 0.5 en todo se obtiene exactamente el voraz; el SMA aprende qué cajas
    conviene favorecer o castigar para salir de las decisiones miopes del voraz.
    """
    return _insercion_voraz(inst, np.exp(GAMMA_GUIADO * (X - 0.5)))


DECODIFICADORES = {
    "estricto": decodificar_estricto,
    "salto": decodificar_salto,
    "insercion": decodificar_insercion,
    "guiado": decodificar_guiado,
}


# ----------------------------------------------------------------------------
# Evaluación de la población (el punto a paralelizar)
# ----------------------------------------------------------------------------
_INST = None
_DECO = None


def _init_trabajador(inst, deco):
    global _INST, _DECO
    _INST, _DECO = inst, deco


def _evaluar_uno(x):
    ruta = DECODIFICADORES[_DECO](x, _INST)
    return _INST.almas_ruta(ruta), ruta


def evaluar_poblacion(X, inst, deco, pool=None):
    if pool is None:
        res = [(inst.almas_ruta(r), r) for r in (DECODIFICADORES[deco](x, inst) for x in X)]
    else:
        res = pool.map(_evaluar_uno, list(X))
    fit = np.array([f for f, _ in res])
    rutas = [r for _, r in res]
    return fit, rutas


# ----------------------------------------------------------------------------
# SMA (Li et al., 2020) adaptado a maximización
# ----------------------------------------------------------------------------
def sma(inst, deco="guiado", poblacion=30, iteraciones=200, z=0.03, semilla=None, procesos=1, verbose=True):
    rng = np.random.default_rng(semilla)
    N, n, T = poblacion, inst.n, iteraciones
    X = rng.random((N, n))
    if deco == "guiado":
        X[0] = 0.5   # un agente arranca exactamente en la solución voraz

    DF, X_best, ruta_best = -np.inf, None, []
    historia_mejor, historia_media, mejoras = [], [], []
    pool = Pool(procesos, initializer=_init_trabajador, initargs=(inst, deco)) if procesos > 1 else None
    t0 = time.perf_counter()
    try:
        for it in range(T):
            fit, rutas = evaluar_poblacion(X, inst, deco, pool)

            orden = np.argsort(-fit)              # de mejor a peor
            bF, wF = fit[orden[0]], fit[orden[-1]]
            if bF > DF:
                DF, X_best, ruta_best = bF, X[orden[0]].copy(), rutas[orden[0]]
                mejoras.append({"iter": it, "almas": float(DF), "tiempo": inst.tiempo_ruta(ruta_best),
                                "ruta": [int(v) for v in ruta_best]})
            historia_mejor.append(float(DF))
            historia_media.append(float(fit.mean()))

            # pesos protoplasmáticos W, Ec. (3): mitad buena se refuerza, mitad mala se debilita
            razon = (bF - fit[orden]) / (bF - wF + EPS)          # 0 = mejor, 1 = peor
            r = rng.random((N, n))
            log = np.log10(razon[:, None] + 1.0)
            mitad = N // 2
            W = np.empty((N, n))
            W[orden[:mitad]] = 1.0 + r[:mitad] * log[:mitad]
            W[orden[mitad:]] = 1.0 - r[mitad:] * log[mitad:]

            # parámetros que decrecen con las iteraciones
            a = np.arctanh(1.0 - (it + 1) / (T + 1))
            b = 1.0 - it / T
            p = np.tanh(np.abs(fit - DF) / (abs(DF) + EPS))

            X_nuevo = np.empty_like(X)
            for i in range(N):
                if rng.random() < z:                              # exploración aleatoria
                    X_nuevo[i] = rng.random(n)
                    continue
                vb = rng.uniform(-a, a, n)
                vc = rng.uniform(-b, b, n)
                A, B = rng.integers(N, size=2)
                acercarse = rng.random(n) < p[i]
                X_nuevo[i] = np.where(acercarse,
                                      X_best + vb * (W[i] * X[A] - X[B]),   # hacia el mejor alimento
                                      vc * X[i])                            # oscilación local
            X = np.clip(X_nuevo, 0.0, 1.0)

            if verbose and (it % max(1, T // 10) == 0 or it == T - 1):
                print(f"  iter {it:4d}  mejor={DF:7.1f}  media={fit.mean():7.1f}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "almas": float(DF),
        "tiempo": inst.tiempo_ruta(ruta_best),
        "ruta": [int(v) for v in ruta_best],
        "historia_mejor": historia_mejor,
        "historia_media": historia_media,
        "mejoras": mejoras,
        "segundos_cpu": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------
# Línea base: voraz por razón almas / tiempo de inserción (para comparar)
# ----------------------------------------------------------------------------
def voraz(inst):
    ruta = _insercion_voraz(inst)
    return {"almas": inst.almas_ruta(ruta), "tiempo": inst.tiempo_ruta(ruta), "ruta": [int(v) for v in ruta]}


def _insercion_voraz(inst, sesgo=None):
    C = inst.C
    sec = [0, 0] if inst.regreso else [0]
    t = 0.0
    libres = np.nonzero(inst.alcanzable)[0]
    while len(libres):
        a = np.array(sec[:-1])
        b = np.array(sec[1:])
        delta = C[a][:, libres] + C[libres][:, b].T - C[a, b][:, None]   # (posiciones, candidatos)
        if not inst.regreso:
            delta = np.vstack([delta, C[sec[-1], libres]])
        pos = delta.argmin(0)
        costo = delta[pos, np.arange(len(libres))]
        cabe = t + costo <= inst.tmax
        if not cabe.any():
            break
        valor = inst.S[libres] / (costo + EPS)
        if sesgo is not None:
            valor = valor * sesgo[libres - 1]
        valor = np.where(cabe, valor, -np.inf)
        k = int(valor.argmax())
        sec.insert(int(pos[k]) + 1, int(libres[k]))
        t += costo[k]
        libres = np.delete(libres, k)
    return sec[1:-1] if inst.regreso else sec[1:]


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=RAIZ / "config.json")
    p.add_argument("--nodos", default=RAIZ / "datos" / "nodos.json")
    p.add_argument("--tmax", type=float, nargs="+", help="uno o varios presupuestos de tiempo (s)")
    p.add_argument("--decodificador", choices=DECODIFICADORES)
    p.add_argument("--poblacion", type=int)
    p.add_argument("--iteraciones", type=int)
    p.add_argument("--semilla", type=int)
    p.add_argument("--repeticiones", type=int, default=1)
    p.add_argument("--procesos", type=int, default=1)
    p.add_argument("--capas", nargs="+", choices=list(NIVEL_CAPA), help="usar solo estas capas")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    datos = json.loads(Path(args.nodos).read_text(encoding="utf-8"))
    nodos = datos["nodos"]
    if args.capas:
        nodos = [n for n in nodos if n["capa"] in args.capas]
    s = cfg["sma"]
    deco = args.decodificador or cfg["decodificador"]
    poblacion = args.poblacion or s["poblacion"]
    iteraciones = args.iteraciones or s["iteraciones"]
    semilla = args.semilla if args.semilla is not None else s["semilla"]

    corridas = []
    for tmax in (args.tmax or [cfg["tmax"]]):
        cfg_t = dict(cfg, tmax=tmax)
        inst = Instancia(nodos, cfg_t)
        print(f"\n=== t_max = {tmax:g} s | {inst.n} nodos | decodificador = {deco} ===")

        base = voraz(inst)
        print(f"  voraz: {base['almas']:.0f} almas, {len(base['ruta'])} cajas, {base['tiempo']:.1f} s")

        resultados = []
        for rep in range(args.repeticiones):
            sem = None if semilla is None else semilla + rep
            res = sma(inst, deco, poblacion, iteraciones, s["z"], sem, args.procesos,
                      verbose=args.repeticiones == 1)
            resultados.append(res)
            print(f"  SMA rep {rep + 1}: {res['almas']:.0f} almas, {len(res['ruta'])} cajas, "
                  f"{res['tiempo']:.1f} s de ruta, {res['segundos_cpu']:.1f} s de cómputo")
        almas = np.array([r["almas"] for r in resultados])
        if len(almas) > 1:
            print(f"  SMA {len(almas)} reps: media={almas.mean():.1f}  desv={almas.std():.1f}  "
                  f"mejor={almas.max():.0f}  peor={almas.min():.0f}")

        mejor = max(resultados, key=lambda r: r["almas"])
        corridas.append({
            "tmax": tmax, "decodificador": deco, "poblacion": poblacion, "iteraciones": iteraciones,
            "repeticiones": args.repeticiones,
            "almas_reps": almas.tolist(),
            "sma": mejor, "voraz": base,
        })

    # nodos con el índice que usa la instancia (0 = inicio)
    export = {
        "imagen": datos["imagen"],
        "config": cfg,
        "inicio": cfg["inicio"],
        "nodos": [dict(n, idx=i + 1) for i, n in enumerate(nodos)],
        "corridas": corridas,
    }
    (RAIZ / "salidas").mkdir(exist_ok=True)
    (RAIZ / "salidas" / "resultados.json").write_text(json.dumps(export), encoding="utf-8")
    (RAIZ / "visor").mkdir(exist_ok=True)
    (RAIZ / "visor" / "resultados.js").write_text("window.RESULTADOS = " + json.dumps(export) + ";\n", encoding="utf-8")
    print("\nGuardado en salidas/resultados.json y visor/resultados.js (abrir visor/index.html)")


if __name__ == "__main__":
    main()
