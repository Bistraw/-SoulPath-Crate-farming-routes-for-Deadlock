"""Genera el artículo (HTML y PDF) a partir de plantilla.html y salidas/experimentos.json.

Todas las cifras del texto y las tablas salen de los datos, para que el artículo no se
desactualice si se repiten los experimentos.

Uso:  python articulo/generar_articulo.py          (requiere haber corrido figuras.py)
"""
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).parent
RAIZ = AQUI.parent
NOMBRE = "Articulo_SMA_Deadlock_IEEE"
NOMBRES_DECO = {"estricto": "Estricto (III-A)", "salto": "Salto", "insercion": "Inserción", "guiado": "Guiado (propuesto)"}
NOMBRES_CAPA = {"superficie": "Superficie", "subterraneo": "Subterráneo", "azotea": "Azotea"}


def f1(x):
    """Número con una cifra decimal y coma decimal."""
    return f"{x:.1f}".replace(".", ",")


def f2(x):
    return f"{x:.2f}".replace(".", ",")


def entero(x):
    return f"{x:.0f}"


def pct(a, b):
    return (a / b - 1) * 100


def datos():
    cfg = json.loads((RAIZ / "config.json").read_text(encoding="utf-8"))
    nodos = json.loads((RAIZ / "datos" / "nodos.json").read_text(encoding="utf-8"))["nodos"]
    exp = json.loads((RAIZ / "salidas" / "experimentos.json").read_text(encoding="utf-8"))
    rutas = json.loads((AQUI / "rutas_figura.json").read_text(encoding="utf-8"))
    return cfg, nodos, exp, rutas


def tabla_nodos(nodos):
    c = Counter((n["capa"], n["tipo"]) for n in nodos)
    filas = []
    for capa in ["superficie", "subterraneo", "azotea"]:
        cj, ur = c[(capa, "caja")], c[(capa, "urna")]
        filas.append(f'<tr><td class="izq">{NOMBRES_CAPA[capa]}</td><td>{cj}</td><td>{ur}</td><td>{cj + ur}</td></tr>')
    tc = sum(v for (_, t), v in c.items() if t == "caja")
    tu = sum(v for (_, t), v in c.items() if t == "urna")
    filas.append(f'<tr><th class="izq">Total</th><th>{tc}</th><th>{tu}</th><th>{tc + tu}</th></tr>')
    return "\n  ".join(filas)


def ms(r):
    return f'{f1(r["media"])} ± {f1(r["desv"])}'


def construir(cfg, nodos, exp, rutas):
    dec = exp["decodificadores"]["res"]
    v90 = exp["decodificadores"]["voraz"]
    pres = {p["tmax"]: p for p in exp["presupuestos"]}
    par = exp["paralelo"]["res"]
    t1 = par[0]["segundos"]["media"]
    sp = {x["procesos"]: t1 / x["segundos"]["media"] for x in par}
    g, e, s, ins = dec["guiado"], dec["estricto"], dec["salto"], dec["insercion"]
    eq = exp["equipo"]
    P = exp["parametros"]

    tok = {
        "n_nodos": str(len(nodos)),
        "g90_media": f1(g["resumen"]["media"]),
        "g90_mejora": f1(pct(g["resumen"]["media"], v90)),
        "e90_media": f1(e["resumen"]["media"]),
        "sp4": f2(sp[4]),
        "tabla_nodos": tabla_nodos(nodos),
        "L": entero(cfg["ancho_mapa_m"]), "v": f1(cfg["velocidad_ms"]), "tau": f1(cfg["tortuosidad"]),
        "delta": f1(cfg["penalizacion_capa_s"]),
        "beta_c": f1(cfg["tiempo_romper_s"]["caja"]), "beta_u": f1(cfg["tiempo_romper_s"]["urna"]),
        "s_c": entero(cfg["almas"]["caja"]), "s_u": entero(cfg["almas"]["urna"]),
        "cpu": "Intel Core i7-7700HQ a 2,80 GHz",
        "nucleos": f"4 núcleos físicos, {eq['nucleos_logicos']} hilos",
        "python": eq["python"], "numpy": eq["numpy"],
        "N": str(P["poblacion"]), "T": str(P["iteraciones"]), "z": f2(P["z"]),
        "reps": str(P["reps"]), "reps_m1": str(P["reps"] - 1),
        "voraz90": entero(v90),
        "voraz90_cajas": str(pres[90]["voraz"]["cajas"]),
    }

    filas = []
    for k in ["estricto", "salto", "insercion", "guiado"]:
        r = dec[k]
        filas.append(f'<tr><td class="izq">{NOMBRES_DECO[k]}</td><td>{ms(r["resumen"])}</td><td>{entero(r["resumen"]["mejor"])}</td>'
                     f'<td>{entero(r["resumen"]["peor"])}</td><td>{f1(r["cajas"]["media"])}</td><td>{f1(r["segundos"]["media"])}</td></tr>')
    tok["tabla_decodificadores"] = "\n  ".join(filas)

    filas = []
    for t in sorted(pres):
        p = pres[t]
        r = p["sma"]["resumen"]
        filas.append(f'<tr><td>{t}</td><td>{entero(p["voraz"]["almas"])}</td><td>{ms(r)}</td><td>{entero(r["mejor"])}</td>'
                     f'<td>+{f1(pct(r["media"], p["voraz"]["almas"]))} %</td><td>{f1(p["sma"]["segundos"]["media"])}</td></tr>')
    tok["tabla_presupuestos"] = "\n  ".join(filas)

    filas = []
    for x in par:
        k = x["procesos"]
        filas.append(f'<tr><td>{k}</td><td>{f2(x["segundos"]["media"])} ± {f2(x["segundos"]["desv"])}</td>'
                     f'<td>{f2(sp[k])}</td><td>{f1(sp[k] / k * 100)} %</td></tr>')
    tok["tabla_paralelo"] = "\n  ".join(filas)

    tok.update(tokens_gpu(exp, pres, sp))
    tok.update(textos(exp, dec, v90, pres, sp, rutas, P))
    return tok


def tokens_gpu(exp, pres, sp):
    gpu = exp["gpu"]
    gp = gpu["paralelo"]
    t1 = exp["paralelo"]["res"][0]["segundos"]["media"]
    tok = {
        "gpu_nombre": gpu["dispositivo"].replace("NVIDIA ", ""),
        "gpu_sms": str(gpu["sms"]),
        "gpu_nucleos": str(gpu["sms"] * 128),   # Pascal (cc 6.x): 128 núcleos CUDA por multiprocesador
        "gpu_mem": entero(gpu["memoria_gb"]),
        "gpu_cc": gpu["cc"].replace(".", ","),
        "cupy": gpu["cupy"], "nvrtc": gpu["cuda_nvrtc"].replace(".", ","),
        "gpu_hilos": str(gpu["hilos_bloque"]),
        "gpu_sp": entero(gp["aceleracion"]),
    }
    filas = []
    iguales = total = 0
    acel = {}
    for g in gpu["presupuestos"]:
        c = pres[g["tmax"]]["sma"]
        acel[g["tmax"]] = c["segundos"]["media"] / g["segundos"]["media"]
        n_ig = sum(a == b for a, b in zip(c["almas"], g["almas"]))
        iguales += n_ig
        total += len(g["almas"])
        filas.append(f'<tr><td>{g["tmax"]}</td><td>{f2(c["segundos"]["media"])}</td><td>{f2(g["segundos"]["media"])}</td>'
                     f'<td>{f1(acel[g["tmax"]])}</td><td>{n_ig} / {len(g["almas"])}</td></tr>')
    tok["tabla_gpu"] = "\n  ".join(filas)

    ts = sorted(acel)
    tok["texto_gpu"] = (
        f"Con la población evaluada en GPU, las 30 iteraciones del experimento anterior requirieron {f2(gp['segundos']['media'])} ± "
        f"{f2(gp['segundos']['desv'])} s, frente a {f2(t1)} s de la versión secuencial: una aceleración de {entero(gp['aceleracion'])}, "
        f"{entero(gp['aceleracion'] / sp[8])} veces mayor que la obtenida con 8 procesos. La Tabla VII repite las corridas de la Tabla V "
        f"({exp['parametros']['iteraciones']} iteraciones, {exp['parametros']['reps']} semillas) en GPU y las compara con la "
        f"versión de 4 procesos. En {iguales} de las {total} corridas la GPU obtuvo exactamente las mismas almas que la CPU con la "
        f"misma semilla, lo que confirma que el kernel reproduce la decodificación secuencial. La aceleración crece con el "
        f"presupuesto: de {f1(acel[ts[0]])} con {ts[0]} s a {f1(acel[ts[-1]])} con {ts[-1]} s. Con presupuestos cortos las rutas "
        f"tienen pocas cajas, la decodificación en CPU es barata y predominan los costos fijos de lanzar el kernel y copiar datos; "
        f"con presupuestos largos cada paso de la construcción evalúa cientos de candidatos, trabajo que en la GPU se reparte entre "
        f"los hilos del bloque.")
    tok["texto_gpu_2"] = (
        f"Tras la aceleración, el kernel (incluidas las copias entre CPU y GPU) representa solo el "
        f"{f1(gp['fraccion_kernel'] * 100)} % del tiempo de una iteración; el resto corresponde a la actualización del SMA y al cálculo "
        f"del fitness, que permanecen en CPU. Por la ley de Amdahl, esta parte secuencial es ahora el principal límite para seguir "
        f"reduciendo el tiempo. Además, con <i>N</i> = {exp['parametros']['poblacion']} se lanzan solo {exp['parametros']['poblacion']} "
        f"bloques para {gpu['sms']} multiprocesadores, por lo que poblaciones mayores aprovecharían mejor la GPU.")
    return tok


def textos(exp, dec, v90, pres, sp, rutas, P):
    g, e, s, ins = dec["guiado"], dec["estricto"], dec["salto"], dec["insercion"]
    gm, em, sm, im = (d["resumen"]["media"] for d in (g, e, s, ins))
    supera = sum(a > v90 for a in g["almas"])
    t = {}
    t["texto_decodificadores"] = (
        f"El decodificador estricto, que corresponde a la regla planteada originalmente, obtuvo {f1(em)} almas en promedio, "
        f"equivalente al {f1(em / v90 * 100)} % de la solución voraz. Omitir las cajas que no caben (salto) elevó el promedio a "
        f"{f1(sm)} almas, y la inserción por costo mínimo a {f1(im)}; sin embargo, ninguna de estas tres variantes alcanzó a la "
        f"heurística voraz ({entero(v90)} almas). El decodificador guiado obtuvo {f1(gm)} ± {f1(g['resumen']['desv'])} almas, "
        f"{f1(pct(gm, v90))} % por encima de la referencia, y superó a la solución voraz en {supera} de las {P['reps']} repeticiones.")
    t["texto_decodificadores_2"] = (
        f"La mejora en calidad tiene un costo: cada corrida del decodificador guiado requirió en promedio "
        f"{f1(g['segundos']['media'])} s, frente a {f1(e['segundos']['media'])} s del estricto, debido a la complejidad de la "
        f"construcción voraz descrita en la Sección IV-C.")

    partes = []
    for tm in sorted(pres):
        p = pres[tm]
        partes.append(f"{f1(pct(p['sma']['resumen']['media'], p['voraz']['almas']))} % con {tm} s")
    t["texto_presupuestos"] = (
        "El SMA guiado mejoró en promedio a la heurística voraz en todos los casos: " + ", ".join(partes[:-1]) +
        f" y {partes[-1]}. El tiempo de cómputo crece con el presupuesto, pues las rutas contienen más cajas y el costo de cada "
        f"decodificación aumenta de forma cuadrática con su longitud.")

    it = g["iter_mejor"]
    hm = g["historia_mejor_media"]
    t["texto_convergencia"] = (
        f"La Fig. 2 muestra la evolución del mejor valor global. Los decodificadores basados únicamente en llaves aleatorias "
        f"mejoran de forma gradual, pero permanecen por debajo de la referencia voraz durante las {P['iteraciones']} iteraciones. "
        f"El decodificador guiado inicia en el valor voraz, gracias al agente inicializado en 0,5, y alcanza un promedio de "
        f"{f1(hm[len(hm) // 4 - 1])} almas en la iteración {len(hm) // 4} y {f1(hm[-1])} al final. En promedio, su mejor solución "
        f"se encontró en la iteración {f1(it['media'] + 1)}, aunque con una dispersión amplia (de la iteración "
        f"{entero(it['peor'] + 1)} a la {entero(it['mejor'] + 1)}); dado que algunas repeticiones mejoraron cerca del final, no puede "
        f"descartarse que un mayor número de iteraciones produzca mejoras adicionales.")

    v, sm_ = rutas["voraz"], rutas["sma"]
    t["texto_rutas"] = (
        f"La Fig. 3 compara la ruta voraz ({entero(v['almas'])} almas, {len(v['ruta'])} contenedores, {f1(v['tiempo'])} s) con la "
        f"obtenida por el SMA guiado con la semilla 0 ({entero(sm_['almas'])} almas, {len(sm_['ruta'])} contenedores, "
        f"{f1(sm_['tiempo'])} s). La heurística voraz se dirige al grupo de cajas más próximo a la base, al noroeste, porque es el "
        f"que ofrece el mejor cociente inmediato. El SMA, en cambio, selecciona un grupo situado al norte y noreste, cuyo acceso "
        f"requiere un desplazamiento inicial mayor, pero que concentra más contenedores en poco espacio. Este comportamiento ilustra "
        f"la limitación miope de la construcción voraz que el sesgo aprendido por el SMA permite corregir.")

    par = exp["paralelo"]["res"]
    t1 = par[0]["segundos"]["media"]
    t["texto_paralelo"] = (
        f"La Tabla VI reporta el tiempo de 30 iteraciones del SMA guiado. Con 2 procesos la aceleración fue de {f2(sp[2])} y con 4 "
        f"procesos de {f2(sp[4])}, lo que equivale a una eficiencia de {f1(sp[4] / 4 * 100)} %. Con 8 procesos la aceleración aumentó "
        f"a {f2(sp[8])}, pese a que el procesador cuenta con 4 núcleos físicos; los 4 hilos adicionales que ofrece la tecnología "
        f"Hyper-Threading aprovechan los periodos en que un núcleo espera datos de memoria, aunque con una eficiencia por proceso "
        f"menor ({f1(sp[8] / 8 * 100)} %). La eficiencia por debajo del ideal se explica por la sobrecarga de serializar los vectores "
        f"y las rutas entre procesos en cada iteración, por la actualización del SMA, que permanece secuencial, y por el desbalance "
        f"de carga, ya que los agentes con rutas más largas tardan más en decodificarse. La desviación estándar del caso secuencial "
        f"({f2(par[0]['segundos']['desv'])} s) se debe a que cada semilla produce rutas de distinta longitud. Estos resultados "
        f"justifican la exploración de esquemas de memoria compartida en la siguiente etapa.")

    g45, g150 = pres[45]["sma"]["resumen"]["media"], pres[150]["sma"]["resumen"]["media"]
    t["texto_conclusion"] = (
        f"La decodificación directa de llaves aleatorias resultó inadecuada para una instancia de {exp['parametros']['n_nodos']} "
        f"nodos, con un desempeño inferior al de una heurística voraz simple. El decodificador guiado, que integra el SMA con dicha "
        f"heurística, superó a la referencia en promedio entre {f1(min(pct(pres[t_]['sma']['resumen']['media'], pres[t_]['voraz']['almas']) for t_ in pres))} % "
        f"y {f1(max(pct(pres[t_]['sma']['resumen']['media'], pres[t_]['voraz']['almas']) for t_ in pres))} % según el presupuesto "
        f"de tiempo. La evaluación paralela de la población con procesos obtuvo una aceleración de {f2(sp[4])} con cuatro procesos, "
        f"y la evaluación en GPU con CUDA, de {entero(exp['gpu']['paralelo']['aceleracion'])} respecto a la versión secuencial, "
        f"sin alterar los resultados.")
    return t


def exportar_pdf(html):
    candidatos = [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  shutil.which("chrome"), r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    nav = next((c for c in candidatos if c and Path(c).exists()), None)
    if not nav:
        print("No se encontró Edge ni Chrome; solo se generó el HTML.")
        return
    pdf = html.with_suffix(".pdf")
    # Edge puede regresar antes de escribir el archivo y no siempre sobrescribe: se borra y se espera
    pdf.unlink(missing_ok=True)
    subprocess.run([nav, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], capture_output=True, timeout=120)
    for _ in range(30):
        if pdf.exists() and pdf.stat().st_size > 0:
            print("PDF ->", pdf)
            return
        time.sleep(0.5)
    print("No se pudo generar el PDF; ábrelo en el navegador e imprímelo como PDF.")


def main():
    cfg, nodos, exp, rutas = datos()
    html = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    for k, v in construir(cfg, nodos, exp, rutas).items():
        html = html.replace("{{" + k + "}}", v)
    faltan = sorted(set(p.split("}}")[0] for p in html.split("{{")[1:]))
    if faltan:
        raise SystemExit(f"Marcadores sin valor: {faltan}")
    destino = AQUI / f"{NOMBRE}.html"
    destino.write_text(html, encoding="utf-8")
    print("HTML ->", destino)
    exportar_pdf(destino)


if __name__ == "__main__":
    main()
