"""
Tarea 3 - Ejercicio 1: Ajuste de la temperatura del Recocido Simulado (2-opt, TSP) con DOE
Opcion A: diseno factorial completo 3 (temperatura) x 3 (tamano), 10 replicas por punto.

Uso:  python doe_temperatura.py
Genera: figuras en ./figuras y resultados.json en ./
"""
import math
import random
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

MATRICULA = 2043329         
TAMANOS = [5, 10, 15]        
REPLICAS = 10
ITERACIONES = 100            
TF = 0.05                    
T_MIN, T_MAX = 1.0, 50.0     
ALFA = 0.05
N_BOOT = 10_000

os.makedirs("figuras", exist_ok=True)


def generar_ciudades(n, rng):
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]


def matriz_distancias(ciudades):
    n = len(ciudades)
    return [[math.dist(ciudades[i], ciudades[j]) for j in range(n)] for i in range(n)]


def tour_aleatorio(n, rng):
    t = list(range(n))
    rng.shuffle(t)
    return t


def costo_tour(tour, d):
    n = len(tour)
    return sum(d[tour[i]][tour[(i + 1) % n]] for i in range(n))



def temperatura(k, N, T0, Tf):
    return T0 * (Tf / T0) ** (k / (N - 1))


def recocido_simulado(d, iteraciones, T0, Tf, rng):
    n = len(d)
    tour = tour_aleatorio(n, rng)
    costo = costo_tour(tour, d)
    mejor = costo

    for k in range(iteraciones):
        T = temperatura(k, iteraciones, T0, Tf)
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i > j:
            i, j = j, i
        if i != j and not (i == 0 and j == n - 1):
            a = tour[i - 1]
            b = tour[(j + 1) % n]
            delta = d[a][tour[j]] + d[tour[i]][b] - d[a][tour[i]] - d[tour[j]][b]
            if delta < 0 or rng.random() < math.exp(-delta / T):
                tour[i:j + 1] = tour[i:j + 1][::-1]
                costo += delta
                if costo < mejor:
                    mejor = costo
    return mejor



def anova_dos_factores(df, fa="temperatura", fb="tamano", y="costo"):
    a_niv = sorted(df[fa].unique())
    b_niv = sorted(df[fb].unique())
    a, b = len(a_niv), len(b_niv)
    r = len(df) // (a * b)          
    gran = df[y].mean()

    media_a = df.groupby(fa)[y].mean()
    media_b = df.groupby(fb)[y].mean()
    media_ab = df.groupby([fa, fb])[y].mean()

    ss_a = b * r * ((media_a - gran) ** 2).sum()
    ss_b = a * r * ((media_b - gran) ** 2).sum()
    ss_ab = 0.0
    for ia in a_niv:
        for ib in b_niv:
            ss_ab += r * (media_ab[(ia, ib)] - media_a[ia] - media_b[ib] + gran) ** 2
    ss_e = sum(((g[y] - g[y].mean()) ** 2).sum() for _, g in df.groupby([fa, fb]))

    gl = {"temperatura": a - 1, "tamano": b - 1, "interaccion": (a - 1) * (b - 1), "error": a * b * (r - 1)}
    ss = {"temperatura": ss_a, "tamano": ss_b, "interaccion": ss_ab, "error": ss_e}
    ms = {k: ss[k] / gl[k] for k in ss}
    filas = []
    for k in ["temperatura", "tamano", "interaccion"]:
        F = ms[k] / ms["error"]
        p = stats.f.sf(F, gl[k], gl["error"])
        filas.append(dict(fuente=k, SS=ss[k], gl=gl[k], MS=ms[k], F=F, p=p))
    filas.append(dict(fuente="error", SS=ss_e, gl=gl["error"], MS=ms["error"], F=np.nan, p=np.nan))
    return pd.DataFrame(filas)


def bootstrap_mediana(x, rng_np, B=N_BOOT):
    x = np.asarray(x)
    meds = np.median(rng_np.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
   
    random.seed(MATRICULA)
    r = random.random()
    fracciones = [0.3 * r, 0.3 * r + 0.3, 0.3 * r + 0.6]
    nombres = ["bajo", "medio", "alto"]
    temps = [T_MIN + f * (T_MAX - T_MIN) for f in fracciones]
    print(f"r = {r:.6f}")
    for nom, f, t in zip(nombres, fracciones, temps):
        print(f"  nivel {nom:<5} fraccion = {f:.4f}  ->  T0 = {t:.3f}")

   
    d_inst = {}
    for n in TAMANOS:
        rng_inst = random.Random(MATRICULA + n)
        d_inst[n] = matriz_distancias(generar_ciudades(n, rng_inst))


    filas = []
    for it, n in enumerate(TAMANOS):
        for ik, (nom, T0) in enumerate(zip(nombres, temps)):
            for rep in range(REPLICAS):
                semilla = MATRICULA * 7 + 1000 * it + 100 * ik + rep
                rng = random.Random(semilla)
                c = recocido_simulado(d_inst[n], ITERACIONES, T0, TF, rng)
                filas.append(dict(tamano=n, nivel=nom, temperatura=round(T0, 3), rep=rep, costo=round(c, 6)))
    df = pd.DataFrame(filas)
    df.to_csv("datos_corridas.csv", index=False)
    print(f"\nCorridas totales: {len(df)}")

    
    tabla = anova_dos_factores(df)
    print("\nANOVA de dos factores (alfa = 0.05)")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    
    rng_np = np.random.default_rng(MATRICULA)
    boot = {}
    for n in TAMANOS:
        for T0 in temps:
            x = df[(df.tamano == n) & (df.temperatura == round(T0, 3))]["costo"].values
            boot[(n, round(T0, 3))] = bootstrap_mediana(x, rng_np)

    print("\nMediana bootstrap (IC 95%) por punto de diseno")
    for (n, T0), (m, lo, hi) in boot.items():
        print(f"  n={n:<2} T0={T0:<7} mediana={m:8.2f}  IC=[{lo:.2f}, {hi:.2f}]")

    
    resumen = []
    for n in TAMANOS:
        cand = {T0: boot[(n, T0)] for (nn, T0) in boot if nn == n}
        T_best = min(cand, key=lambda t: cand[t][0])
        grupos = [df[(df.tamano == n) & (df.temperatura == round(T0, 3))]["costo"].values for T0 in temps]
        if all(np.std(g) == 0 for g in grupos) and len({g[0] for g in grupos}) == 1:
            p_n = float("nan")
        else:
            p_n = float(stats.f_oneway(*grupos).pvalue)
        m, lo, hi = cand[T_best]
        resumen.append(dict(tamano=n, T_optima=T_best, mediana=m, ic_inf=lo, ic_sup=hi, p_anova_tamano=p_n))
    res = pd.DataFrame(resumen)
    print("\nResumen por tamano")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    
if __name__ == "__main__":
    main()
