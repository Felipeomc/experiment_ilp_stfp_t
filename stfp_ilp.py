"""
stfp_ilp.py  —  STFP-T: Software Team Formation Problem (variante Técnica)
Baseline MILP para comparação com o Algoritmo Genético do TeamPlus.

Implementação fiel à formalização em formalizacao_STFP_linearizado.pdf,
incluindo todos os casos (A/B/C) e linearizações (McCormick exato,
clamp, ρ = max(η,·), produto contínuo t = clamp·ρ).

Dependência única: pip install pulp
Python 3.8+
"""

# =============================================================================
# CONSTANTES CONFIGURÁVEIS — altere apenas aqui
# =============================================================================
CAMINHO_BASE        = "base_final_com_rdas.json"
CAMINHO_PROJETOS    = "target_projects.json"
ID_PROJETO          = "P1"   # qual projeto rodar (P1–P6)
TAMANHO_EQUIPE      = 4      # k
MAX_DEVS            = None   # None = todos; número = limitar para teste rápido
TEMPO_LIMITE_SOLVER = 60     # segundos

# =============================================================================
# COEFICIENTES CALIBRADOS (iguais para DOM, ECO, LING)
# =============================================================================

# Caso B (sit2) — clamp linearizado
B_INTERCEPT = -0.333333
B_A_FULL    =  0.166667
B_A_RED2    =  1.166667

# Caso C (sit1) — produto triplo + ρ + clamp + produto contínuo
C_INTERCEPT = 0.0
C_W_COVM    = 0.071429
C_W_RED2M   = 0.428571
C_W_INTMS   = 0.357143
C_W_HELPC   = 0.5

# Pesos AT (wmean — espelham a Rede Bayesiana)
W_DOM  = 3
W_ECO  = 1
W_LING = 5

# Parâmetros adicionais
ETA        = 0.1   # floor da penalização Must
MU_NEUTRO  = 0.5   # escore de dimensão sem requisitos (Caso A)

# =============================================================================
# EXTENSÕES FUTURAS (não implementadas)
# =============================================================================
# TODO: Rodar todos os 6 projetos em sequência e salvar CSV comparativo com GA
# TODO: Incluir AC e AE (STFP completo) com variáveis m, M, b para min/max:
#       m ≤ AT, m ≤ AC
#       M ≥ AT, M ≥ AC, M ≤ AT + (1-b), M ≤ AC + b
#       AE = (5*m + 1*M) / 6
#       Requer dados de compatibilidade PC_ij entre pares de devs.

# =============================================================================
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pulp


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DE DADOS
# ---------------------------------------------------------------------------

def carregar_base(caminho: str, max_devs: Optional[int] = None) -> List[Dict]:
    """
    Carrega e normaliza a base de desenvolvedores.
    Aceita tanto 'id' (base antiga) quanto 'user_id' (base nova).
    Converte competências para lowercase.
    """
    print(f"\n{'='*70}")
    print("CARREGANDO BASE DE DESENVOLVEDORES")
    print(f"{'='*70}")

    with open(caminho, encoding="utf-8") as f:
        raw = json.load(f)

    devs = []
    for obj in raw:
        uid = obj.get("user_id") or obj.get("id")
        if uid is None:
            continue
        dev = {
            "id":         int(uid),
            "nome":       obj.get("nomeCompleto", obj.get("nome", f"Dev#{uid}")),
            "dominio":    [c.lower() for c in (obj.get("dominio") or [])],
            "ecossistema":[c.lower() for c in (obj.get("ecossistema") or [])],
            "linguagens": [c.lower() for c in (obj.get("linguagens") or [])],
        }
        devs.append(dev)

    if max_devs is not None:
        devs = devs[:max_devs]

    print(f"Total de devs carregados: {len(devs)}")
    print(f"\n{'Dev':>5}  {'Nome':<35}  DOM  ECO  LING")
    print("-" * 65)
    for d in devs:
        print(
            f"{d['id']:>5}  {d['nome']:<35}  "
            f"{len(d['dominio']):>3}  "
            f"{len(d['ecossistema']):>3}  "
            f"{len(d['linguagens']):>4}"
        )

    return devs


def carregar_projetos(caminho: str) -> Dict[str, Dict]:
    """Carrega o arquivo de projetos-alvo e retorna dict {id → projeto}."""
    with open(caminho, encoding="utf-8") as f:
        raw = json.load(f)
    return {p["id"]: p for p in raw["projects"]}


# ---------------------------------------------------------------------------
# 2. CONSTRUÇÃO DO MODELO MILP
# ---------------------------------------------------------------------------

def _competencias_projeto(projeto: Dict, dim: str) -> Tuple[List[str], List[str], List[str]]:
    """Retorna (must, should, could) em lowercase para uma dimensão."""
    req = projeto.get(dim, {})
    must   = [c.lower() for c in req.get("must",   [])]
    should = [c.lower() for c in req.get("should", [])]
    could  = [c.lower() for c in req.get("could",  [])]
    return must, should, could


def _universo(devs: List[Dict], dim: str) -> List[str]:
    """Universo de competências presentes na base para uma dimensão."""
    s = set()
    for d in devs:
        s.update(d[dim])
    return sorted(s)


def _a(dev: Dict, dim: str, comp: str) -> int:
    """Parâmetro binário: 1 se dev possui comp na dim."""
    return 1 if comp in dev[dim] else 0


def construir_modelo_ilp(
    devs: List[Dict],
    projeto: Dict,
    k: int,
) -> Tuple[pulp.LpProblem, Dict, Dict]:
    """
    Constrói o modelo MILP conforme a formalização matemática (Seção 2).

    Retorna:
        modelo   — LpProblem configurado
        x        — dict {i: LpVariable} de seleção de devs
        info_dim — dict {dim: {'caso', 'DIM', 'must', 'should', 'could'}}
    """
    N = len(devs)
    DIMS = ["dominio", "ecossistema", "linguagens"]
    DIM_LABEL = {"dominio": "DOM", "ecossistema": "ECO", "linguagens": "LING"}

    print(f"\n{'='*70}")
    print("CONSTRUINDO MODELO MILP — STFP-T")
    print(f"{'='*70}")
    print(f"Projeto: {projeto['id']} — {projeto['name']}")
    print(f"Tamanho da equipe k = {k}  |  Candidatos N = {N}")

    # ------------------------------------------------------------------
    # PARTE 1 — Modelo e variável de decisão xi
    # ------------------------------------------------------------------
    print(f"\n[Parte 1] Variáveis de decisão xi  (binário, i=0..{N-1})")
    modelo = pulp.LpProblem("STFP_T", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(N)}

    # Restrição de tamanho da equipe  ∑ xi = k
    modelo += (pulp.lpSum(x[i] for i in range(N)) == k, "tamanho_equipe")

    # ------------------------------------------------------------------
    # PARTE 2 — Variáveis de cobertura y e z  (Seção 2.3)
    # ------------------------------------------------------------------
    print("[Parte 2] Contagem e cobertura h, y, z por dimensão")

    # Universo de competências (união de todos os devs + requisitos)
    universo: Dict[str, List[str]] = {}
    for dim in DIMS:
        must, should, could = _competencias_projeto(projeto, dim)
        comp_proj = set(must + should + could)
        comp_devs = set()
        for d in devs:
            comp_devs.update(d[dim])
        universo[dim] = sorted(comp_proj | comp_devs)

    # h[dim][u] = ∑_i a_iu * xi   (expressão linear, não variável)
    h: Dict[str, Dict[str, pulp.LpAffineExpression]] = {}
    y: Dict[str, Dict[str, pulp.LpVariable]] = {}
    z: Dict[str, Dict[str, pulp.LpVariable]] = {}

    for dim in DIMS:
        h[dim] = {}
        y[dim] = {}
        z[dim] = {}
        for u in universo[dim]:
            # h_u = ∑_i a_iu * xi
            h[dim][u] = pulp.lpSum(_a(devs[i], dim, u) * x[i] for i in range(N))

            safe = u.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")
            label = f"{DIM_LABEL[dim]}_{safe}"

            # y_u ∈ {0,1}: cobertura simples (≥1 membro com u)
            y[dim][u] = pulp.LpVariable(f"y_{label}", cat="Binary")
            modelo += (y[dim][u] <= h[dim][u],             f"y_ub_{label}")
            modelo += (y[dim][u] >= (1/k) * h[dim][u],    f"y_lb_{label}")

            # z_u ∈ {0,1}: cobertura redundante (≥2 membros com u)
            z[dim][u] = pulp.LpVariable(f"z_{label}", cat="Binary")
            modelo += (2 * z[dim][u] <= h[dim][u],              f"z_ub_{label}")
            modelo += (z[dim][u] >= (1/k) * (h[dim][u] - 1),   f"z_lb_{label}")

    # ------------------------------------------------------------------
    # PARTE 3 — Features de cobertura MoSCoW (Seção 2.4)
    # ------------------------------------------------------------------
    print("[Parte 3] Features full e red2 por prioridade MoSCoW")

    req: Dict[str, Tuple[List, List, List]] = {}
    for dim in DIMS:
        req[dim] = _competencias_projeto(projeto, dim)   # (must, should, could)

    def _full(dim: str, prioridade: str) -> pulp.LpAffineExpression:
        """
        fullδ,π = (1/|Rδ,π|) ∑_{u∈Rδ,π} yδ_u,  ou 1 se vazio.
        """
        idx = {"must": 0, "should": 1, "could": 2}[prioridade]
        reqs = req[dim][idx]
        if not reqs:
            return pulp.lpSum([1.0])   # constante 1
        return (1.0 / len(reqs)) * pulp.lpSum(y[dim][u] for u in reqs if u in y[dim])

    def _red2(dim: str, prioridade: str) -> pulp.LpAffineExpression:
        """
        red2δ,π = (1/|Rδ,π|) ∑_{u∈Rδ,π} zδ_u,  ou 0 se vazio.
        """
        idx = {"must": 0, "should": 1, "could": 2}[prioridade]
        reqs = req[dim][idx]
        if not reqs:
            return pulp.lpSum([0.0])
        return (1.0 / len(reqs)) * pulp.lpSum(z[dim][u] for u in reqs if u in z[dim])

    # ------------------------------------------------------------------
    # PARTE 4 — Escore técnico por dimensão (Seção 2.5)
    # ------------------------------------------------------------------
    print("[Parte 4] Escore técnico DIM por dimensão (casos A/B/C)")

    info_dim: Dict[str, Dict] = {}
    DIM_vars: Dict[str, Any] = {}   # variável ou constante para cada dim

    for dim in DIMS:
        must, should, could = req[dim]
        label = DIM_LABEL[dim]

        pis_ativas = []
        if must:   pis_ativas.append("must")
        if should: pis_ativas.append("should")
        if could:  pis_ativas.append("could")

        n_ativas = len(pis_ativas)
        info_dim[dim] = {
            "must": must, "should": should, "could": could,
            "pis_ativas": pis_ativas,
        }

        print(f"\n  Dimensão {label}: must={must}, should={should}, could={could}")

        # ---- Caso A: nenhuma prioridade ativa ----------------------
        if n_ativas == 0:
            info_dim[dim]["caso"] = "A"
            DIM_vars[dim] = MU_NEUTRO
            print(f"  → Caso A (neutro): DIM{label} = {MU_NEUTRO}")

        # ---- Caso B: exatamente uma prioridade ativa ---------------
        elif n_ativas == 1:
            info_dim[dim]["caso"] = "B"
            pi_star = pis_ativas[0]
            full_pi = _full(dim, pi_star)
            red2_pi = _red2(dim, pi_star)

            # v_B = b + a_full * full + a_red2 * red2  (linear)
            v_B = B_INTERCEPT + B_A_FULL * full_pi + B_A_RED2 * red2_pi

            # DIM_B ∈ [0,1], DIM_B ≤ v_B  → clamp linearizado (maximização)
            DIM_B = pulp.LpVariable(f"DIM_{label}_B", lowBound=0.0, upBound=1.0)
            modelo += (DIM_B <= v_B, f"DIM_{label}_B_clamp")

            DIM_vars[dim] = DIM_B
            print(f"  → Caso B (π*={pi_star}): DIM{label}_B com clamp linearizado")

        # ---- Caso C: duas ou mais prioridades ativas ---------------
        else:
            info_dim[dim]["caso"] = "C"
            full_M  = _full(dim, "must")
            red2_M  = _red2(dim, "must")
            full_S  = _full(dim, "should")
            full_C  = _full(dim, "could")

            print(f"  → Caso C ({n_ativas} prioridades ativas): linearizações McCormick")

            # ---- C.1: Produto fullM * fullS via McCormick exato ----
            # p = (1 / |M|·|S|) ∑_{i,j} w_MS_ij
            # Nota: full_M = (1/|M|) ∑_i y_ui, logo
            #       full_M * full_S = (1/(|M|·|S|)) ∑_i ∑_j y_ui · y_uj
            must_list   = [u for u in must   if u in y[dim]]
            should_list = [u for u in should if u in y[dim]]
            could_list  = [u for u in could  if u in y[dim]]

            w_MS: Dict[Tuple, pulp.LpVariable] = {}
            if must_list and should_list:
                for ui in must_list:
                    for uj in should_list:
                        safe_i = ui.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                        safe_j = uj.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                        vname = f"wMS_{label}_{safe_i}_{safe_j}"[:60]
                        w = pulp.LpVariable(vname, cat="Binary")
                        w_MS[(ui, uj)] = w
                        modelo += (w <= y[dim][ui],                    f"{vname}_ub1")
                        modelo += (w <= y[dim][uj],                    f"{vname}_ub2")
                        modelo += (w >= y[dim][ui] + y[dim][uj] - 1,  f"{vname}_lb")

            denom_MS = len(must_list) * len(should_list) if (must_list and should_list) else 1
            p_delta = (
                (1.0 / denom_MS) * pulp.lpSum(w_MS.values())
                if w_MS else pulp.lpSum([0.0])
            )

            # ---- C.2: Produto triplo fullM * fullC * (1 - fullS) ----
            # Passo 1: w_MC_ij = y_ui AND y_uj  (ui∈M, uj∈C)
            w_MC: Dict[Tuple, pulp.LpVariable] = {}
            if must_list and could_list:
                for ui in must_list:
                    for uj in could_list:
                        safe_i = ui.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                        safe_j = uj.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                        vname = f"wMC_{label}_{safe_i}_{safe_j}"[:60]
                        w = pulp.LpVariable(vname, cat="Binary")
                        w_MC[(ui, uj)] = w
                        modelo += (w <= y[dim][ui],                    f"{vname}_ub1")
                        modelo += (w <= y[dim][uj],                    f"{vname}_ub2")
                        modelo += (w >= y[dim][ui] + y[dim][uj] - 1,  f"{vname}_lb")

            # Passo 2: v_MCS_ijl = w_MC_ij AND (1 - y_ul)  (ul∈S)
            v_MCS: Dict[Tuple, pulp.LpVariable] = {}
            for (ui, uj), w_mc in w_MC.items():
                for ul in should_list:
                    safe_i = ui.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                    safe_j = uj.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                    safe_l = ul.replace(" ","_").replace("/","_").replace("(","").replace(")","").replace(",","")
                    vname = f"vMCS_{label}_{safe_i}_{safe_j}_{safe_l}"[:60]
                    v = pulp.LpVariable(vname, cat="Binary")
                    v_MCS[(ui, uj, ul)] = v
                    modelo += (v <= w_mc,                              f"{vname}_ub1")
                    modelo += (v <= 1 - y[dim][ul],                   f"{vname}_ub2")
                    modelo += (v >= w_mc + (1 - y[dim][ul]) - 1,      f"{vname}_lb")

            denom_MCS = (len(must_list) * len(could_list) * len(should_list)
                         if (must_list and could_list and should_list) else 1)
            r_delta = (
                (1.0 / denom_MCS) * pulp.lpSum(v_MCS.values())
                if v_MCS else pulp.lpSum([0.0])
            )
            # Se não há should, helpC = full_M * full_C (sem penalização por S)
            # Tratado implicitamente: v_MCS = {} → r_delta = 0 → whelpC contribui 0
            # (caso correto: quando should vazio, o termo helpC não se aplica)

            # ---- C.3: ρ = max(η, fullM) ----
            if not must_list:
                # |R^{δ,M}| = 0 → ρ = 1 (conforme Seção 2.5, C.3)
                rho = 1.0
                print(f"     ρ_{label} = 1.0 (must vazio)")
            else:
                b_rho = pulp.LpVariable(f"b_rho_{label}", cat="Binary")
                rho   = pulp.LpVariable(f"rho_{label}", lowBound=ETA, upBound=1.0)

                modelo += (rho >= ETA,                             f"rho_{label}_eta")
                modelo += (rho >= full_M,                          f"rho_{label}_full")
                modelo += (rho <= full_M + (1 - b_rho),           f"rho_{label}_ub_full")
                modelo += (rho <= ETA + b_rho,                    f"rho_{label}_ub_eta")
                modelo += (full_M >= ETA - (1 - b_rho),           f"rho_{label}_b_cond")
                print(f"     ρ_{label} = max({ETA}, fullM) — linearizado")

            # ---- C.4: Clamp e produto final DIM = clamp(v_C) · ρ ----
            v_C = (C_INTERCEPT
                   + C_W_COVM  * full_M
                   + C_W_RED2M * red2_M
                   + C_W_INTMS * p_delta
                   + C_W_HELPC * r_delta)

            DIM_clamp = pulp.LpVariable(f"DIM_{label}_clamp", lowBound=0.0, upBound=1.0)
            modelo += (DIM_clamp <= v_C, f"DIM_{label}_clamp_ub")

            # t = DIM_clamp · ρ  (McCormick para [0,1] × [η,1])
            t = pulp.LpVariable(f"t_{label}", lowBound=0.0, upBound=1.0)
            modelo += (t <= DIM_clamp,                      f"t_{label}_ub1")
            modelo += (t <= rho,                            f"t_{label}_ub2")
            modelo += (t >= DIM_clamp + rho - 1,           f"t_{label}_lb")
            modelo += (t >= 0,                              f"t_{label}_nn")

            DIM_vars[dim] = t
            print(f"     DIM{label} = t_{label} = clamp(v_C) · ρ")

    # ------------------------------------------------------------------
    # PARTE 5 — Aptidão Técnica AT  (Seção 2.6)
    # ------------------------------------------------------------------
    print(f"\n[Parte 5] Aptidão Técnica AT = wmean(DOM, ECO, LING) = "
          f"({W_DOM}·DOM + {W_ECO}·ECO + {W_LING}·LING) / {W_DOM+W_ECO+W_LING}")

    soma_pesos = W_DOM + W_ECO + W_LING
    AT = (
        W_DOM  * DIM_vars["dominio"]
        + W_ECO  * DIM_vars["ecossistema"]
        + W_LING * DIM_vars["linguagens"]
    ) / soma_pesos

    # STFP-T: maximiza apenas AT (sem AC/AE — STFP completo é extensão futura)
    modelo += (AT, "objetivo_AT")

    print("\n[Modelo] Função objetivo: maximizar AT")
    print(f"[Modelo] Variáveis: {modelo.numVariables()} | "
          f"Restrições: {modelo.numConstraints()}")

    return modelo, x, info_dim, DIM_vars, req


# ---------------------------------------------------------------------------
# 3. EXIBIÇÃO DO RESULTADO
# ---------------------------------------------------------------------------

def exibir_resultado(
    modelo: pulp.LpProblem,
    x: Dict[int, pulp.LpVariable],
    devs: List[Dict],
    projeto: Dict,
    k: int,
    info_dim: Dict,
    DIM_vars: Dict,
    req: Dict,
    tempo_solver: float,
) -> None:
    """Exibe logs pós-solução: status, equipe, cobertura, escores."""

    DIMS = ["dominio", "ecossistema", "linguagens"]
    DIM_LABEL = {"dominio": "DOM", "ecossistema": "ECO", "linguagens": "LING"}

    print(f"\n{'='*70}")
    print("RESULTADO DO SOLVER")
    print(f"{'='*70}")

    status = pulp.LpStatus[modelo.status]
    at_val = pulp.value(modelo.objective)

    print(f"Status  : {status}")
    print(f"AT ótimo: {at_val:.6f}" if at_val is not None else "AT ótimo: N/A")
    print(f"Tempo   : {tempo_solver:.2f}s")

    if modelo.status != 1:
        print("⚠  Solver não encontrou solução ótima. Verifique os logs acima.")
        return

    # Equipe selecionada e descartados
    selecionados = [i for i in range(len(devs)) if pulp.value(x[i]) > 0.5]
    descartados  = [i for i in range(len(devs)) if pulp.value(x[i]) <= 0.5]

    print(f"\n{'─'*70}")
    print(f"EQUIPE SELECIONADA  (k={k})")
    print(f"{'─'*70}")
    print(f"{'ID':>5}  {'Nome':<35}  {'DOM':<25}  {'ECO':<25}  {'LING'}")
    print("-" * 120)
    for i in selecionados:
        d = devs[i]
        print(
            f"{d['id']:>5}  {d['nome']:<35}  "
            f"{str(d['dominio']):<25}  "
            f"{str(d['ecossistema']):<25}  "
            f"{d['linguagens']}"
        )

    print(f"\n{'─'*70}")
    print(f"DEVS DESCARTADOS  ({len(descartados)} dev(s))")
    print(f"{'─'*70}")
    for i in descartados:
        d = devs[i]
        print(f"  {d['id']:>5}  {d['nome']}")

    # Cobertura dos requisitos Must por dimensão
    print(f"\n{'─'*70}")
    print("COBERTURA DOS REQUISITOS MUST")
    print(f"{'─'*70}")
    for dim in DIMS:
        must = req[dim][0]
        label = DIM_LABEL[dim]
        if not must:
            print(f"  {label}: sem requisitos Must")
            continue
        for u in must:
            cobre = any(_a(devs[i], dim, u) == 1 for i in selecionados)
            simbolo = "✓" if cobre else "✗"
            print(f"  {label} | {simbolo}  {u}")

    # Escores por dimensão
    print(f"\n{'─'*70}")
    print("ESCORES POR DIMENSÃO")
    print(f"{'─'*70}")
    for dim in DIMS:
        label = DIM_LABEL[dim]
        caso  = info_dim[dim]["caso"]
        var   = DIM_vars[dim]
        val   = pulp.value(var) if not isinstance(var, (int, float)) else var
        val   = val if val is not None else 0.0
        print(f"  {label}  Caso {caso}  →  DIM = {val:.6f}")

    print(f"\n{'─'*70}")
    print(f"  AT (wmean) = {at_val:.6f}")
    print(f"{'─'*70}")


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'#'*70}")
    print("#  STFP-ILP  —  TeamPlus Baseline MILP (variante STFP-T)           #")
    print(f"{'#'*70}")
    print(f"Projeto     : {ID_PROJETO}")
    print(f"Equipe k    : {TAMANHO_EQUIPE}")
    print(f"Max devs    : {MAX_DEVS if MAX_DEVS else 'todos'}")
    print(f"Tempo limite: {TEMPO_LIMITE_SOLVER}s")

    # --- Carga
    devs     = carregar_base(CAMINHO_BASE, MAX_DEVS)
    projetos = carregar_projetos(CAMINHO_PROJETOS)

    if ID_PROJETO not in projetos:
        sys.exit(f"Projeto '{ID_PROJETO}' não encontrado. Disponíveis: {list(projetos.keys())}")

    projeto = projetos[ID_PROJETO]
    k       = projeto.get("team_size", TAMANHO_EQUIPE)

    print(f"\n{'='*70}")
    print("PROJETO-ALVO SELECIONADO")
    print(f"{'='*70}")
    print(f"ID   : {projeto['id']}")
    print(f"Nome : {projeto['name']}")
    for dim in ["dominio", "ecossistema", "linguagens"]:
        r = projeto.get(dim, {})
        print(f"  {dim.upper():<15} must={r.get('must',[])}  "
              f"should={r.get('should',[])}  could={r.get('could',[])}")

    # --- Construção do modelo
    modelo, x, info_dim, DIM_vars, req = construir_modelo_ilp(devs, projeto, k)

    # --- Solver
    print(f"\n{'='*70}")
    print("RODANDO SOLVER CBC (PuLP)")
    print(f"{'='*70}")
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=TEMPO_LIMITE_SOLVER)

    t0 = time.time()
    modelo.solve(solver)
    tempo_solver = time.time() - t0

    # --- Resultado
    exibir_resultado(
        modelo, x, devs, projeto, k,
        info_dim, DIM_vars, req, tempo_solver
    )


if __name__ == "__main__":
    main()
