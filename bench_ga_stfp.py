#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_ga_stfp.py — Experimento STFP-GA: rodada completa P1-P12
================================================================
Script de orquestração para rodar o Genetic Algorithm de formação de equipes
nos mesmos projetos usados no bench do ILP, salvando resultados compatíveis
com a tabela do ILP.

Objetivo:
  - Rodar o GA calibrado com CCC fixo nos projetos P1-P12.
  - Opcionalmente rodar também o baseline CCC.
  - Salvar resultados por execução/seed e resumo agregado por projeto.
  - Facilitar posterior comparação com o ILP por gap percentual e tempo.

Modo atual:
  AE_FULL_SURROGATE
  O GA usa o avaliador surrogate Pipeline.evaluate_teams_sur.avaliar_equipe_surrogate,
  que retorna AE/fitness, AT_cont e AC_cont.

Uso dentro da pasta do experimento ILP, por exemplo:
  python bench_ga_stfp.py

Uso recomendado para comparação robusta:
  python bench_ga_stfp.py --configs both --seeds 30 --restart

Se quiser rodar somente o GA calibrado:
  python bench_ga_stfp.py --configs calibrado --seeds 30 --restart

Se a raiz do projeto STFP não for detectada automaticamente:
  python bench_ga_stfp.py --project-root D:\\DOCUMENTS\\works\\STFP
"""

from __future__ import annotations

# =============================================================================
# CONSTANTES CONFIGURÁVEIS — parte principal que você pode alterar
# =============================================================================

CAMINHO_PROJETOS = "target_projects.json"
CAMINHO_SAIDA_RUNS = "resultados_ga_12projetos_runs.csv"
CAMINHO_SAIDA_RESUMO = "resultados_ga_12projetos_resumo.csv"
CAMINHO_SAIDA_COMPARACAO = "comparacao_ga_calibrado_vs_baseline.csv"

MODO = "AE_FULL_SURROGATE"

PROJETOS_IDS = [
    "P1", "P2", "P3", "P4", "P5", "P6",
    "P7", "P8", "P9", "P10", "P11", "P12",
]

SEEDS = list(range(1, 31))  # 30 execuções independentes por projeto/configuração

# Configuração escolhida pelo racing adaptativo com CCC fixo.
GA_CALIBRADO_CCC = {
    "population_size": 150,
    "generations": 50,
    "elitism_count": 1,
    "mutation_rate": 0.01,
    "crossover_rate": 0.60,
    "stable_gens": 20,
    "crossover_operator": "ccc",
}

# Baseline reaproveitado dos experimentos anteriores.
GA_BASELINE_CCC = {
    "population_size": 100,
    "generations": 100,
    "elitism_count": 1,
    "mutation_rate": 0.05,
    "crossover_rate": 1.00,
    "stable_gens": 10,
    "crossover_operator": "ccc",
}

CONFIGURACOES = {
    "GA_CALIBRADO_CCC": GA_CALIBRADO_CCC,
    "GA_BASELINE_CCC": GA_BASELINE_CCC,
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import contextlib
import csv
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bench STFP-GA para rodar o GA nos mesmos projetos do ILP."
    )
    parser.add_argument(
        "--target-projects",
        type=str,
        default=CAMINHO_PROJETOS,
        help="Caminho para target_projects.json. Padrão: target_projects.json na pasta atual.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Raiz do projeto STFP contendo Algorithms/ e Pipeline/. Se omitido, tenta detectar.",
    )
    parser.add_argument(
        "--output-runs",
        type=str,
        default=CAMINHO_SAIDA_RUNS,
        help="CSV de saída com uma linha por execução/seed.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=CAMINHO_SAIDA_RESUMO,
        help="CSV de saída com resumo por projeto/configuração.",
    )
    parser.add_argument(
        "--output-comparison",
        type=str,
        default=CAMINHO_SAIDA_COMPARACAO,
        help="CSV com comparação pareada GA calibrado vs baseline, se ambos forem rodados.",
    )
    parser.add_argument(
        "--configs",
        choices=["calibrado", "baseline", "both"],
        default="both",
        help="Quais configurações rodar. Padrão: both.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help="Quantidade de seeds por projeto/configuração. Padrão: 30.",
    )
    parser.add_argument(
        "--projects",
        type=str,
        default=",".join(PROJETOS_IDS),
        help="Lista de projetos separados por vírgula. Padrão: P1,...,P12.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Apaga os CSVs de saída antes de iniciar.",
    )
    return parser.parse_args()


# =============================================================================
# RESOLUÇÃO DE CAMINHOS E IMPORTAÇÃO DO PROJETO
# =============================================================================

def detectar_project_root(script_dir: Path, cli_project_root: str | None) -> Path:
    if cli_project_root:
        root = Path(cli_project_root).resolve()
        if not (root / "Algorithms").exists() or not (root / "Pipeline").exists():
            raise RuntimeError(f"project-root informado não contém Algorithms/ e Pipeline/: {root}")
        return root

    candidatos = [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
    ]

    for c in candidatos:
        if (c / "Algorithms").exists() and (c / "Pipeline").exists():
            return c.resolve()

    raise RuntimeError(
        "Não consegui detectar a raiz do projeto STFP. "
        "Use --project-root CAMINHO_DA_PASTA_STFP."
    )


def importar_engine(project_root: Path):
    """
    Importa o engine do GA e força o avaliador surrogate.
    Protege contra argparse interno do engine.py limpando sys.argv temporariamente.
    """
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]

        try:
            import Feature_Extraction.Dimension_Scoring.dimension_scoring as _ds
            _ds.DEBUG_DIM = False
        except Exception:
            pass

        import Algorithms.GA.engine as engine
        from Pipeline.evaluate_teams_sur import avaliar_equipe_surrogate

        engine.avaliar_equipe = avaliar_equipe_surrogate
        return engine
    finally:
        sys.argv = original_argv


@contextlib.contextmanager
def silent_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


# =============================================================================
# LEITURA DOS PROJETOS
# =============================================================================

def carregar_projetos(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Formato 1: {"projects": [{"id": "P1", ...}, ...]}
    if isinstance(data, dict) and isinstance(data.get("projects"), list):
        return {str(p["id"]): p for p in data["projects"]}

    # Formato 2: {"P1": {...}, "P2": {...}}
    if isinstance(data, dict):
        projetos = {}
        for pid, p in data.items():
            if isinstance(p, dict):
                p2 = dict(p)
                p2.setdefault("id", pid)
                projetos[str(pid)] = p2
        if projetos:
            return projetos

    raise ValueError(
        "Formato de target_projects.json não reconhecido. "
        "Use {'projects': [...]} ou {'P1': {...}, ...}."
    )


def project_payload(project: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dominio": project.get("dominio", []),
        "ecossistema": project.get("ecossistema", []),
        "linguagens": project.get("linguagens", []),
    }


# =============================================================================
# CSV INCREMENTAL
# =============================================================================

RUN_FIELDS = [
    # Campos compatíveis com o bench do ILP
    "projeto",
    "modo",
    "status",
    "AT",
    "AC",
    "AE",
    "equipe",
    "tempo_s",
    "must_dom_ok",
    "must_eco_ok",
    "must_ling_ok",

    # Campos específicos do GA
    "project_name",
    "metodo",
    "seed",
    "fitness_final",
    "gens_executed",
    "stop_reason",
    "dom_score",
    "eco_score",
    "ling_score",
    "population_size",
    "generations",
    "elitism_count",
    "mutation_rate",
    "crossover_rate",
    "stable_gens",
    "crossover_operator",
    "timestamp",
]


def ensure_csv(path: Path, fields: List[str], restart: bool = False) -> None:
    if restart and path.exists():
        path.unlink()
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def append_csv(path: Path, fields: List[str], row: Dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow({k: row.get(k, "") for k in fields})


def ler_runs_existentes(path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    if not path.exists():
        return {}

    idx = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (str(row["metodo"]), str(row["projeto"]), int(row["seed"]))
                idx[key] = row
            except Exception:
                continue
    return idx


# =============================================================================
# EXECUÇÃO DO GA
# =============================================================================

def rodar_ga_uma_vez(engine, projeto: Dict[str, Any], config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    engine.ELITISMO = int(config["elitism_count"])

    t0 = time.perf_counter()
    with silent_stdout():
        result = engine.run_ga_com_config(
            PROJETO_ALVO_EXTERNO=project_payload(projeto),
            team_size=int(projeto.get("team_size", 4)),
            pop_size=int(config["population_size"]),
            geracoes=int(config["generations"]),
            seed=int(seed),
            crossover_rate=float(config["crossover_rate"]),
            mutation_rate=float(config["mutation_rate"]),
            stable_gens=int(config["stable_gens"]),
            crossover_operator=str(config["crossover_operator"]),
            verbose=False,
            report=False,
            log_eval=False,
        )
    elapsed = time.perf_counter() - t0

    best_team = result.get("best_team", [])
    fitness = float(result.get("best_fitness", 0.0))

    at = ""
    ac = ""
    ae = fitness
    dom_score = ""
    eco_score = ""
    ling_score = ""

    # Reavalia a melhor equipe para registrar AT, AC e escores por dimensão.
    if best_team:
        try:
            with silent_stdout():
                eval_res = engine.avaliar_equipe(
                    best_team,
                    project_payload(projeto),
                    log=False,
                )
            if isinstance(eval_res, dict):
                at = eval_res.get("AT_cont", "")
                ac = eval_res.get("AC_cont", "")
                ae = eval_res.get("media_AE", fitness)
                scores = eval_res.get("scores", {})
                if isinstance(scores, dict):
                    dom_score = scores.get("dominio", {}).get("score", "")
                    eco_score = scores.get("ecossistema", {}).get("score", "")
                    ling_score = scores.get("linguagens", {}).get("score", "")
        except Exception:
            # O fitness principal já foi retornado pelo GA; métricas auxiliares ficam vazias.
            pass

    return {
        "status": "OK",
        "AT": at,
        "AC": ac,
        "AE": ae,
        "fitness_final": fitness,
        "equipe": json.dumps(best_team, ensure_ascii=False),
        "tempo_s": elapsed,
        "gens_executed": int(result.get("gens_executed", 0)),
        "stop_reason": str(result.get("stop_reason", "")),
        "dom_score": dom_score,
        "eco_score": eco_score,
        "ling_score": ling_score,
    }


# =============================================================================
# RESUMOS
# =============================================================================

def to_float(x: Any) -> float | None:
    try:
        if x in (None, ""):
            return None
        return float(x)
    except Exception:
        return None


def media(vals: Iterable[float]) -> float:
    vals = list(vals)
    return statistics.mean(vals) if vals else 0.0


def desvio(vals: Iterable[float]) -> float:
    vals = list(vals)
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def mediana(vals: Iterable[float]) -> float:
    vals = list(vals)
    return statistics.median(vals) if vals else 0.0


def gerar_resumo(runs_path: Path, summary_path: Path) -> List[Dict[str, Any]]:
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grupos: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        key = (str(r["projeto"]), str(r["metodo"]))
        grupos.setdefault(key, []).append(r)

    summary_fields = [
        "projeto", "project_name", "metodo", "modo", "n_runs",
        "AE_mean", "AE_best", "AE_std",
        "AT_mean", "AC_mean",
        "tempo_mean", "tempo_median", "tempo_total",
        "gens_mean", "best_seed", "best_team",
    ]

    resumo = []
    for (pid, metodo), rs in sorted(grupos.items()):
        aes = [to_float(r["AE"]) for r in rs]
        ats = [to_float(r["AT"]) for r in rs]
        acs = [to_float(r["AC"]) for r in rs]
        tempos = [to_float(r["tempo_s"]) for r in rs]
        gens = [to_float(r["gens_executed"]) for r in rs]

        aes_f = [x for x in aes if x is not None]
        ats_f = [x for x in ats if x is not None]
        acs_f = [x for x in acs if x is not None]
        tempos_f = [x for x in tempos if x is not None]
        gens_f = [x for x in gens if x is not None]

        best_row = max(rs, key=lambda r: to_float(r.get("AE")) or -1.0)

        resumo.append({
            "projeto": pid,
            "project_name": best_row.get("project_name", ""),
            "metodo": metodo,
            "modo": best_row.get("modo", MODO),
            "n_runs": len(rs),
            "AE_mean": round(media(aes_f), 8),
            "AE_best": round(max(aes_f), 8) if aes_f else "",
            "AE_std": round(desvio(aes_f), 8),
            "AT_mean": round(media(ats_f), 8) if ats_f else "",
            "AC_mean": round(media(acs_f), 8) if acs_f else "",
            "tempo_mean": round(media(tempos_f), 6),
            "tempo_median": round(mediana(tempos_f), 6),
            "tempo_total": round(sum(tempos_f), 6),
            "gens_mean": round(media(gens_f), 6),
            "best_seed": best_row.get("seed", ""),
            "best_team": best_row.get("equipe", ""),
        })

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(resumo)

    return resumo


def sign_test_p_value(wins: int, losses: int) -> float:
    import math
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, p)


def gerar_comparacao_calibrado_baseline(runs_path: Path, comparison_path: Path) -> None:
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "OK"]

    by_pair = {}
    for r in rows:
        try:
            key = (r["projeto"], int(r["seed"]))
            by_pair.setdefault(key, {})[r["metodo"]] = r
        except Exception:
            continue

    comp_by_project: Dict[str, List[Tuple[float, float]]] = {}
    for (pid, _seed), m in by_pair.items():
        if "GA_CALIBRADO_CCC" in m and "GA_BASELINE_CCC" in m:
            cal = to_float(m["GA_CALIBRADO_CCC"].get("AE"))
            bas = to_float(m["GA_BASELINE_CCC"].get("AE"))
            if cal is not None and bas is not None:
                comp_by_project.setdefault(pid, []).append((cal, bas))

    fields = [
        "projeto", "n_pares", "calibrado_mean", "baseline_mean",
        "absolute_gain", "relative_gain_percent",
        "wins", "losses", "ties", "paired_mean_diff", "sign_test_p_value",
    ]

    out = []
    for pid, pairs in sorted(comp_by_project.items()):
        cal_vals = [x for x, _ in pairs]
        bas_vals = [y for _, y in pairs]
        diffs = [x - y for x, y in pairs]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        cal_mean = statistics.mean(cal_vals)
        bas_mean = statistics.mean(bas_vals)
        rel = 100.0 * (cal_mean - bas_mean) / bas_mean if bas_mean != 0 else ""
        out.append({
            "projeto": pid,
            "n_pares": len(pairs),
            "calibrado_mean": round(cal_mean, 8),
            "baseline_mean": round(bas_mean, 8),
            "absolute_gain": round(cal_mean - bas_mean, 8),
            "relative_gain_percent": round(rel, 6) if rel != "" else "",
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "paired_mean_diff": round(statistics.mean(diffs), 8),
            "sign_test_p_value": round(sign_test_p_value(wins, losses), 8),
        })

    if not out:
        return

    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    target_projects_path = Path(args.target_projects).resolve()
    if not target_projects_path.exists():
        raise FileNotFoundError(f"Arquivo de projetos não encontrado: {target_projects_path}")

    output_runs = Path(args.output_runs).resolve()
    output_summary = Path(args.output_summary).resolve()
    output_comparison = Path(args.output_comparison).resolve()

    if args.restart:
        for p in [output_runs, output_summary, output_comparison]:
            if p.exists():
                p.unlink()

    ensure_csv(output_runs, RUN_FIELDS, restart=False)
    runs_existentes = ler_runs_existentes(output_runs)

    project_root = detectar_project_root(script_dir, args.project_root)
    projetos = carregar_projetos(target_projects_path)

    projetos_ids = [p.strip() for p in args.projects.split(",") if p.strip()]
    seeds = list(range(1, int(args.seeds) + 1))

    if args.configs == "calibrado":
        configs = {"GA_CALIBRADO_CCC": GA_CALIBRADO_CCC}
    elif args.configs == "baseline":
        configs = {"GA_BASELINE_CCC": GA_BASELINE_CCC}
    else:
        configs = CONFIGURACOES

    print(f"\n{'#'*70}")
    print(f"#  BENCH STFP-GA  |  Modo: {MODO}")
    print(f"#  Projetos: {projetos_ids}")
    print(f"#  Configurações: {list(configs.keys())}")
    print(f"#  Seeds: 1..{len(seeds)}")
    print(f"{'#'*70}\n")

    print(f"[BENCH-GA] project_root    : {project_root}")
    print(f"[BENCH-GA] target_projects : {target_projects_path}")
    print(f"[BENCH-GA] output_runs     : {output_runs}")
    print(f"[BENCH-GA] output_summary  : {output_summary}\n")

    ids_invalidos = [pid for pid in projetos_ids if pid not in projetos]
    if ids_invalidos:
        print(f"[BENCH-GA] AVISO: projetos não encontrados: {ids_invalidos}")
    ids_validos = [pid for pid in projetos_ids if pid in projetos]

    engine = importar_engine(project_root)

    total_planejado = len(ids_validos) * len(configs) * len(seeds)
    contador = 0

    for pid in ids_validos:
        projeto = projetos[pid]
        print(f"\n{'='*70}")
        print(f"  PROJETO {pid} — {projeto.get('name', '')}")
        print(f"{'='*70}")

        for metodo, config in configs.items():
            print(f"\n[CONFIG] {metodo}: {config}")

            for seed in seeds:
                contador += 1
                key = (metodo, pid, int(seed))
                if key in runs_existentes:
                    print(f"  [{contador:4d}/{total_planejado}] {pid} {metodo} seed={seed:02d} já existe; pulando")
                    continue

                try:
                    r = rodar_ga_uma_vez(engine, projeto, config, seed)
                except Exception as e:
                    r = {
                        "status": f"Erro: {e}",
                        "AT": "", "AC": "", "AE": "", "fitness_final": "",
                        "equipe": "[]", "tempo_s": 0.0,
                        "gens_executed": "", "stop_reason": "",
                        "dom_score": "", "eco_score": "", "ling_score": "",
                    }

                row = {
                    "projeto": pid,
                    "modo": MODO,
                    "status": r["status"],
                    "AT": r["AT"],
                    "AC": r["AC"],
                    "AE": r["AE"],
                    "equipe": r["equipe"],
                    "tempo_s": round(float(r["tempo_s"]), 6),
                    "must_dom_ok": "",
                    "must_eco_ok": "",
                    "must_ling_ok": "",
                    "project_name": projeto.get("name", ""),
                    "metodo": metodo,
                    "seed": seed,
                    "fitness_final": r["fitness_final"],
                    "gens_executed": r["gens_executed"],
                    "stop_reason": r["stop_reason"],
                    "dom_score": r["dom_score"],
                    "eco_score": r["eco_score"],
                    "ling_score": r["ling_score"],
                    "population_size": config["population_size"],
                    "generations": config["generations"],
                    "elitism_count": config["elitism_count"],
                    "mutation_rate": config["mutation_rate"],
                    "crossover_rate": config["crossover_rate"],
                    "stable_gens": config["stable_gens"],
                    "crossover_operator": config["crossover_operator"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }

                append_csv(output_runs, RUN_FIELDS, row)
                runs_existentes[key] = row

                ae = to_float(row["AE"])
                at = to_float(row["AT"])
                ac = to_float(row["AC"])
                ae_txt = f"{ae:.5f}" if ae is not None else "-"
                at_txt = f"{at:.5f}" if at is not None else "-"
                ac_txt = f"{ac:.5f}" if ac is not None else "-"
                print(
                    f"  [{contador:4d}/{total_planejado}] {pid} {metodo} seed={seed:02d} "
                    f"AE={ae_txt} AT={at_txt} AC={ac_txt} "
                    f"gens={row['gens_executed']} t={row['tempo_s']:.2f}s"
                )

    resumo = gerar_resumo(output_runs, output_summary)
    gerar_comparacao_calibrado_baseline(output_runs, output_comparison)

    print(f"\n\n{'='*70}")
    print("  RESUMO FINAL — GA")
    print(f"{'='*70}")
    header = f"{'projeto':<8} {'metodo':<18} {'n':>4} {'AE_mean':>9} {'AE_best':>9} {'AT_mean':>9} {'AC_mean':>9} {'tempo_m':>9}"
    print(header)
    print("-" * len(header))
    for r in resumo:
        print(
            f"{r['projeto']:<8} {r['metodo']:<18} {r['n_runs']:>4} "
            f"{float(r['AE_mean']):>9.4f} {float(r['AE_best']):>9.4f} "
            f"{float(r['AT_mean']) if r['AT_mean'] != '' else 0:>9.4f} "
            f"{float(r['AC_mean']) if r['AC_mean'] != '' else 0:>9.4f} "
            f"{float(r['tempo_mean']):>9.2f}s"
        )

    print(f"\n[BENCH-GA] CSV de execuções salvo em: {output_runs}")
    print(f"[BENCH-GA] CSV de resumo salvo em    : {output_summary}")
    if output_comparison.exists():
        print(f"[BENCH-GA] CSV de comparação salvo em: {output_comparison}")


if __name__ == "__main__":
    main()
