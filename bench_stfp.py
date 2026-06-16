"""
bench_stfp.py — Experimento STFP-ILP: rodada completa P1-P6
=============================================================
Script de orquestracao que carrega dados, resolve o ILP para cada
projeto e salva resultados em CSV.

MODO atual: AT_ONLY
  A base real nao possui OSF/SLF. O AC e estimado via grafo de
  colaboracao (Graph_DB_real.json), mas o experimento corrente
  compara com o GA apenas em AT.
  Para ativar AE_FULL, altere MODO abaixo.


"""

# =============================================================================
# CONSTANTES CONFIGURÁVEIS — unica parte que muda entre experimentos
# =============================================================================

CAMINHO_BASE     = "base_final_com_rdas_anonimizado.json"
CAMINHO_PROJETOS = "target_projects.json"
CAMINHO_GRAFO    = "Graph_DB_real.json"
CAMINHO_SAIDA    = "resultados_ilp.csv"

MODO             = "AT_ONLY"    # "AT_ONLY" | "AE_FULL"
PROJETOS_IDS     = ["P1", "P2", "P3", "P4", "P5", "P6"]
MAX_DEVS         = None         # None = todos; int = limitar para teste
TEMPO_LIMITE     = 120          # segundos por projeto

# =============================================================================
import csv

from stfp_ilp import (
    carregar_base,
    carregar_projetos,
    carregar_grafo,
    calcular_ac_medios,
    resolver_ilp,
    exibir_resultado,
)


def _must_ok(cobertura: dict, dim: str) -> bool:
    """True se todos os must da dimensao estao cobertos."""
    chaves = [k for k in cobertura if k.startswith(dim + ".")]
    return all(cobertura[k] for k in chaves) if chaves else True


def main() -> None:
    print(f"\n{'#'*70}")
    print(f"#  BENCH STFP-ILP  |  Modo: {MODO}  |  Projetos: {PROJETOS_IDS}")
    print(f"{'#'*70}\n")

    # ------------------------------------------------------------------
    # Carga de dados
    # ------------------------------------------------------------------
    print("[BENCH] Carregando base de desenvolvedores...")
    devs = carregar_base(CAMINHO_BASE, MAX_DEVS)
    print(f"[BENCH] {len(devs)} devs com pelo menos 1 competencia carregados")

    print("[BENCH] Carregando projetos...")
    projetos = carregar_projetos(CAMINHO_PROJETOS)

    print("[BENCH] Carregando grafo de colaboracao...")
    grafo = carregar_grafo(CAMINHO_GRAFO)
    print(f"[BENCH] {len(grafo)} arestas carregadas")

    print("[BENCH] Calculando AC medios...")
    ac_medios = calcular_ac_medios(devs, grafo)
    devs_com_ac = sum(1 for v in ac_medios.values() if v > 0)
    print(f"[BENCH] AC medios prontos | devs com AC>0: {devs_com_ac}/{len(devs)}\n")

    ids_invalidos = [pid for pid in PROJETOS_IDS if pid not in projetos]
    if ids_invalidos:
        print(f"[BENCH] AVISO: projetos nao encontrados: {ids_invalidos}")
    ids_validos = [pid for pid in PROJETOS_IDS if pid in projetos]

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    resultados = []

    for pid in ids_validos:
        projeto = projetos[pid]
        print(f"\n{'='*60}")
        print(f"  PROJETO {pid} — {projeto['name']}")
        print(f"{'='*60}")

        try:
            resultado = resolver_ilp(
                projeto        = projeto,
                devs           = devs,
                ac_medios      = ac_medios,
                modo           = MODO,
                tamanho_equipe = projeto.get("team_size", 4),
                tempo_limite   = TEMPO_LIMITE,
            )
        except Exception as e:
            print(f"[BENCH] ERRO no projeto {pid}: {e}")
            resultado = {
                "status": f"Erro: {e}", "objetivo": None,
                "AT": None, "AC": None, "AE": None,
                "equipe": [], "tempo_s": 0.0,
                "escores_dim": {}, "cobertura_must": {},
                "_devs": devs, "_selecionados": [], "_info": {},
            }

        exibir_resultado(resultado, projeto)

        resultados.append({
            "projeto":      pid,
            "modo":         MODO,
            "status":       resultado["status"],
            "AT":           resultado["AT"]  if resultado["AT"]  is not None else "",
            "AC":           resultado["AC"]  if resultado["AC"]  is not None else "",
            "AE":           resultado["AE"]  if resultado["AE"]  is not None else "",
            "equipe":       str(resultado["equipe"]),
            "tempo_s":      resultado["tempo_s"],
            "must_dom_ok":  _must_ok(resultado["cobertura_must"], "dominio"),
            "must_eco_ok":  _must_ok(resultado["cobertura_must"], "ecossistema"),
            "must_ling_ok": _must_ok(resultado["cobertura_must"], "linguagens"),
        })

    # ------------------------------------------------------------------
    # Resumo final
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  RESUMO FINAL")
    print(f"{'='*70}")
    header = f"{'projeto':<8} {'modo':<10} {'status':<10} {'AT':>7} {'AC':>7} {'AE':>7} {'tempo_s':>8}"
    print(header)
    print("-" * len(header))
    for r in resultados:
        at = f"{r['AT']:.4f}" if r["AT"] != "" else "  -   "
        ac = f"{r['AC']:.4f}" if r["AC"] != "" else "  -   "
        ae = f"{r['AE']:.4f}" if r["AE"] != "" else "  -   "
        print(f"{r['projeto']:<8} {r['modo']:<10} {r['status']:<10} "
              f"{at:>7} {ac:>7} {ae:>7} {r['tempo_s']:>8.2f}s")

    # ------------------------------------------------------------------
    # Salvar CSV
    # ------------------------------------------------------------------
    campos = ["projeto", "modo", "status", "AT", "AC", "AE",
              "equipe", "tempo_s", "must_dom_ok", "must_eco_ok", "must_ling_ok"]
    try:
        with open(CAMINHO_SAIDA, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\n[BENCH] CSV salvo em: {CAMINHO_SAIDA}")
    except Exception as e:
        print(f"\n[BENCH] Erro ao salvar CSV: {e}")


if __name__ == "__main__":
    main()
