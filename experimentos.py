"""Experimentos reportados en el artículo.

1. Comparación de decodificadores (t_max = 90 s).
2. SMA guiado frente al voraz para varios presupuestos de tiempo.
3. Tiempo de cómputo con 1, 2, 4 y 8 procesos (evaluación paralela de la población).
4. Tiempo de cómputo con la evaluación en GPU (CUDA), si hay GPU y CuPy.

Cada configuración se repite con semillas 0..R-1. Resultados en salidas/experimentos.json.

Uso:  python experimentos.py [--reps 10] [--iteraciones 100] [--procesos 4]
      python experimentos.py --solo-gpu   # agrega [4] a un salidas/experimentos.json existente
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from sma_deadlock import GAMMA_GUIADO, Instancia, sma, voraz

RAIZ = Path(__file__).parent


def resumen(valores):
    v = np.asarray(valores, dtype=float)
    return {"media": float(v.mean()), "desv": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "mejor": float(v.max()), "peor": float(v.min())}


def correr(inst, deco, reps, iteraciones, poblacion, z, procesos):
    res = [sma(inst, deco, poblacion, iteraciones, z, semilla, procesos, verbose=False) for semilla in range(reps)]
    almas = [r["almas"] for r in res]
    return {
        "almas": almas,
        "resumen": resumen(almas),
        "cajas": resumen([len(r["ruta"]) for r in res]),
        "tiempo_ruta": resumen([r["tiempo"] for r in res]),
        "segundos": resumen([r["segundos_cpu"] for r in res]),
        "historia_mejor_media": np.mean([r["historia_mejor"] for r in res], axis=0).tolist(),
        # iteración en la que cada repetición alcanzó su mejor valor final
        "iter_mejor": resumen([r["mejoras"][-1]["iter"] for r in res]),
    }


def experimento_gpu(salida, nodos, cfg, args, pob, z):
    """[4] Mismas corridas que [2] y [3] con la población evaluada en GPU (CUDA).

    Las rutas son idénticas a las de CPU, así que las almas deben coincidir con [2].
    """
    import sma_cuda
    if not sma_cuda.disponible():
        print("[4] sin GPU CUDA / CuPy: se omite", flush=True)
        return
    import cupy as cp
    inst90 = Instancia(nodos, dict(cfg, tmax=90))
    ev = sma_cuda.EvaluadorCUDA(inst90, GAMMA_GUIADO)
    sma(inst90, "guiado", pob, 2, z, 0, verbose=False, gpu=ev)   # calentamiento: compila el kernel
    en_kernel = [0.0]

    def medido(X):   # mide cuánto del tiempo total se va en el kernel (incluye las copias CPU <-> GPU)
        t = time.perf_counter()
        rutas = ev(X)
        en_kernel[0] += time.perf_counter() - t
        return rutas

    tiempos = [sma(inst90, "guiado", pob, 30, z, sem, verbose=False, gpu=medido)["segundos_cpu"] for sem in range(3)]
    fraccion_kernel = en_kernel[0] / sum(tiempos)
    t1 = salida["paralelo"]["res"][0]["segundos"]["media"]
    props = cp.cuda.runtime.getDeviceProperties(0)
    salida["gpu"] = {
        "dispositivo": ev.nombre, "sms": props["multiProcessorCount"],
        "memoria_gb": props["totalGlobalMem"] / 2**30,
        "cc": f"{props['major']}.{props['minor']}", "cupy": cp.__version__,
        "cuda_nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "hilos_bloque": sma_cuda.HILOS,
        "paralelo": {"tmax": 90, "iteraciones": 30, "segundos": resumen(tiempos), "aceleracion": t1 / np.mean(tiempos),
                     "fraccion_kernel": fraccion_kernel},
        "presupuestos": [],
    }
    print(f"[4] GPU {ev.nombre}, 30 iteraciones: {np.mean(tiempos):.3f} s ({t1 / np.mean(tiempos):.0f}x)", flush=True)
    for tmax in [45, 90, 150]:
        inst = Instancia(nodos, dict(cfg, tmax=tmax))
        ev = sma_cuda.EvaluadorCUDA(inst, GAMMA_GUIADO)
        res = [sma(inst, "guiado", pob, args.iteraciones, z, sem, verbose=False, gpu=ev) for sem in range(args.reps)]
        salida["gpu"]["presupuestos"].append({"tmax": tmax, "almas": [r["almas"] for r in res],
                                              "resumen": resumen([r["almas"] for r in res]),
                                              "segundos": resumen([r["segundos_cpu"] for r in res])})
        print(f"[4] t_max {tmax}: {salida['gpu']['presupuestos'][-1]['resumen']}, "
              f"{salida['gpu']['presupuestos'][-1]['segundos']['media']:.2f} s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--iteraciones", type=int, default=100)
    p.add_argument("--procesos", type=int, default=4)
    p.add_argument("--solo-gpu", action="store_true", help="solo correr [4] sobre el JSON existente")
    args = p.parse_args()

    cfg = json.loads((RAIZ / "config.json").read_text(encoding="utf-8"))
    nodos = json.loads((RAIZ / "datos" / "nodos.json").read_text(encoding="utf-8"))["nodos"]
    pob, z = cfg["sma"]["poblacion"], cfg["sma"]["z"]
    if args.solo_gpu:
        ruta = RAIZ / "salidas" / "experimentos.json"
        salida = json.loads(ruta.read_text(encoding="utf-8"))
        P = salida["parametros"]
        args.reps, args.iteraciones = P["reps"], P["iteraciones"]
        experimento_gpu(salida, nodos, cfg, args, pob, z)
        ruta.write_text(json.dumps(salida, indent=1), encoding="utf-8")
        print("listo -> salidas/experimentos.json")
        return
    salida = {
        "equipo": {"procesador": platform.processor(), "nucleos_logicos": os.cpu_count(),
                   "python": platform.python_version(), "numpy": np.__version__},
        "parametros": {"reps": args.reps, "iteraciones": args.iteraciones, "poblacion": pob, "z": z,
                       "n_nodos": len(nodos), "config": cfg},
    }
    t0 = time.perf_counter()

    # 1. decodificadores
    inst90 = Instancia(nodos, dict(cfg, tmax=90))
    salida["decodificadores"] = {"tmax": 90, "voraz": voraz(inst90)["almas"], "res": {}}
    for deco in ["estricto", "salto", "insercion", "guiado"]:
        print(f"[1] decodificador {deco} ...", flush=True)
        salida["decodificadores"]["res"][deco] = correr(inst90, deco, args.reps, args.iteraciones, pob, z, args.procesos)
        print("    ", salida["decodificadores"]["res"][deco]["resumen"], flush=True)

    # 2. presupuestos de tiempo
    salida["presupuestos"] = []
    for tmax in [45, 90, 150]:
        print(f"[2] t_max {tmax} ...", flush=True)
        inst = Instancia(nodos, dict(cfg, tmax=tmax))
        v = voraz(inst)
        if tmax == 90:
            r = salida["decodificadores"]["res"]["guiado"]
        else:
            r = correr(inst, "guiado", args.reps, args.iteraciones, pob, z, args.procesos)
        salida["presupuestos"].append({"tmax": tmax, "voraz": {"almas": v["almas"], "cajas": len(v["ruta"]),
                                                                "tiempo": v["tiempo"]}, "sma": r})
        print("    voraz", v["almas"], "sma", r["resumen"], flush=True)

    # 3. escalamiento con procesos (misma semilla, mismas iteraciones)
    salida["paralelo"] = {"tmax": 90, "iteraciones": 30, "res": []}
    for procs in [1, 2, 4, 8]:
        tiempos = []
        for semilla in range(3):
            r = sma(inst90, "guiado", pob, 30, z, semilla, procs, verbose=False)
            tiempos.append(r["segundos_cpu"])
        salida["paralelo"]["res"].append({"procesos": procs, "segundos": resumen(tiempos)})
        print(f"[3] {procs} procesos: {np.mean(tiempos):.2f} s", flush=True)

    experimento_gpu(salida, nodos, cfg, args, pob, z)

    salida["duracion_total_s"] = time.perf_counter() - t0
    (RAIZ / "salidas").mkdir(exist_ok=True)
    (RAIZ / "salidas" / "experimentos.json").write_text(json.dumps(salida, indent=1), encoding="utf-8")
    print(f"listo en {salida['duracion_total_s'] / 60:.1f} min -> salidas/experimentos.json")


if __name__ == "__main__":
    main()
