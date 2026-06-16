"""
stfp_ilp.py — STFP: Software Team Formation Problem (MILP completo)
====================================================================
Modelo MILP para AT (aptidão técnica), AC (aptidão colaborativa) e
AE (aptidão global mixminmax), conforme formalização matemática em
formalizacao_STFP_linearizado.pdf.

Usado como baseline de comparação com o GA do sistema TeamPlus.
Tese de Doutorado — PPGI/UFPB.

Dependência única: pip install pulp
Python 3.8+

Nota de implementação:
  Desenvolvido com suporte de IA generativa (Claude, Anthropic, 2026)
  e verificado pelo co-autor Rian, especialista em Programação Linear
  Inteira.

IMPORTANTE SOBRE AC:
  A base real não possui métricas OSF/SLF. O AC é estimado a partir
  do grafo de colaboração real (Graph_DB_real.json), onde cada aresta
  contém weight = f(N) pré-calculado. A aproximação linear usada é:
    AC_aprox = (1/k) * sum_i AC_medio_i * x_i
  onde AC_medio_i é a média dos pesos de todas as arestas do dev i.
  Isso é completamente linear — sem produto de variáveis binárias.
  O experimento atual roda em modo AT_ONLY. O modo AE_FULL está
  implementado e pode ser ativado no bench quando houver dados mais
  completos de AC.

EXTENSÕES FUTURAS:
  TODO: quando houver OSF/SLF reais, substituir AC_aprox pela formulação
        exata da Seção 2.7: AC = sum_s cs * ps(T), onde ps(T) é a proporção
        de pares da equipe na faixa s. Isso requer produto x_i * x_j
        (não linear), que deve ser linearizado com variáveis binárias p_ij.
"""

# =============================================================================
# COEFICIENTES CALIBRADOS (iguais para DOM, ECO, LING)
# Calibrados por regressão linear contra cenários da Rede Bayesiana.
# =============================================================================

# Caso B (sit2 — exatamente 1 prioridade ativa)
B_INTERCEPT = -0.333333
B_A_FULL    =  0.166667
B_A_RED2    =  1.166667

# Caso C (sit1 — 2+ prioridades ativas)
C_INTERCEPT = 0.0
C_W_COVM    = 0.071429
C_W_RED2M   = 0.428571
C_W_INTMS   = 0.357143
C_W_HELPC   = 0.5

# Pesos AT — wmean espelhando Rede Bayesiana
W_DOM  = 3
W_ECO  = 1
W_LING = 5

# Pesos AE — mixminmax calibrado
W_MIN = 5   # efeito de gargalo (atributo mais fraco)
W_MAX = 1   # atributo mais forte

# Parâmetros adicionais
ETA       = 0.1   # floor da penalização Must
MU_NEUTRO = 0.5   # escore neutro para dimensão sem requisitos (Caso A)

# Centróides AC — faixas VL/L/M/H/VH (referência para futura extensão)
CENTROIDES_AC = [0.1, 0.3, 0.5, 0.7, 0.9]

# =============================================================================
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pulp

Dev     = Dict[str, Any]
Projeto = Dict[str, Any]
Grafo   = Dict[Tuple[int, int], float]


# ===========================================================================
# 1. CARREGAMENTO DE DADOS
# ===========================================================================

def carregar_base(caminho: str, max_devs: Optional[int] = None) -> List[Dev]:
    """
    Carrega e normaliza a base de desenvolvedores.
    Aceita 'user_id' (base nova) ou 'id' (base antiga).
    Filtra devs sem nenhuma competência técnica.
    Converte competências para lowercase.
    """
    with open(caminho, encoding="utf-8") as f:
        raw = json.load(f)

    devs: List[Dev] = []
    for obj in raw:
        uid = obj.get("user_id") or obj.get("id")
        if uid is None:
            continue
        dev: Dev = {
            "id":          int(uid),
            "nome":        obj.get("nomeCompleto", obj.get("nome", f"Dev#{uid}")),
            "dominio":     [c.lower() for c in (obj.get("dominio")     or [])],
            "ecossistema": [c.lower() for c in (obj.get("ecossistema") or [])],
            "linguagens":  [c.lower() for c in (obj.get("linguagens")  or [])],
        }
        if not (dev["dominio"] or dev["ecossistema"] or dev["linguagens"]):
            continue
        devs.append(dev)

    if max_devs is not None:
        devs = devs[:max_devs]

    return devs


def carregar_projetos(caminho: str) -> Dict[str, Projeto]:
    """Carrega projetos-alvo e retorna dict {id -> projeto}."""
    with open(caminho, encoding="utf-8") as f:
        raw = json.load(f)
    return {p["id"]: p for p in raw["projects"]}


def carregar_grafo(caminho: str) -> Grafo:
    """
    Carrega o grafo de colaboração.
    Retorna dict {(uid_i, uid_j): weight} com uid_i < uid_j (par canonico).
    weight = f(N) ja calculado na fonte.
    """
    with open(caminho, encoding="utf-8") as f:
        g = json.load(f)

    mapa: Grafo = {}
    for e in g["edges"]:
        a = int(e["source_user_id"])
        b = int(e["target_user_id"])
        w = float(e["weight"])
        par = (min(a, b), max(a, b))
        mapa[par] = max(mapa.get(par, 0.0), w)

    return mapa


def calcular_ac_medios(devs: List[Dev], grafo: Grafo) -> Dict[int, float]:
    """
    Calcula AC_medio_i para cada dev = media dos pesos de todas as
    arestas em que o dev participa. Devs sem arestas recebem 0.0.

    Usada na aproximacao linear de AC:
      AC_aprox = (1/k) * sum_i AC_medio_i * x_i
    """
    soma:  Dict[int, float] = {}
    conta: Dict[int, int]   = {}

    for (a, b), w in grafo.items():
        soma[a]  = soma.get(a, 0.0) + w
        conta[a] = conta.get(a, 0)  + 1
        soma[b]  = soma.get(b, 0.0) + w
        conta[b] = conta.get(b, 0)  + 1

    ac_medios: Dict[int, float] = {}
    for dev in devs:
        uid = dev["id"]
        ac_medios[uid] = soma[uid] / conta[uid] if uid in conta else 0.0

    return ac_medios


# ===========================================================================
# 2. CONSTRUCAO DO MODELO MILP
# ===========================================================================
# incluir remoção de hífen no _safe() e garantir deduplicação do universo
def _safe(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("-", "_").replace(
        "(", "").replace(")", "").replace(",", "").replace(".", "")[:50]


def _a(dev: Dev, dim: str, comp: str) -> int:
    """Parametro binario: 1 se dev possui comp na dimensao dim."""
    return 1 if comp in dev[dim] else 0


def construir_modelo(
    devs:      List[Dev],
    projeto:   Projeto,
    ac_medios: Dict[int, float],
    modo:      str,
    k:         int,
) -> Tuple[pulp.LpProblem, Dict, Dict]:
    """
    Constrói o modelo MILP conforme Secoes 2.2-2.9 da formalizacao.

    Retorna (modelo, x_vars, info) onde info contem variaveis auxiliares
    para extracao de resultados.
    """
    N    = len(devs)
    DIMS = ["dominio", "ecossistema", "linguagens"]
    LBL  = {"dominio": "DOM", "ecossistema": "ECO", "linguagens": "LING"}
    pid  = projeto.get("id", "?")

    print(f"[ILP] Construindo modelo {modo} para {pid} (k={k}, N={N})")

    # ------------------------------------------------------------------
    # Parte 1 — Variaveis de decisao x_i  (Secao 2.2)
    # ------------------------------------------------------------------
    modelo = pulp.LpProblem(f"STFP_{pid}_{modo}", pulp.LpMaximize)
    x: Dict[int, pulp.LpVariable] = {
        i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(N)
    }
    modelo += (pulp.lpSum(x[i] for i in range(N)) == k, "tamanho_equipe")
    print(f"  [Parte 1] Variaveis x_i: {N}")

    # ------------------------------------------------------------------
    # Parte 2 — Contagem h e coberturas y, z  (Secao 2.3)
    # ------------------------------------------------------------------
    universo: Dict[str, List[str]] = {}
    req:      Dict[str, Tuple[List, List, List]] = {}

    for dim in DIMS:
        r      = projeto.get(dim, {})
        must   = [c.lower() for c in r.get("must",   [])]
        should = [c.lower() for c in r.get("should", [])]
        could  = [c.lower() for c in r.get("could",  [])]
        req[dim] = (must, should, could)
        comp_proj = set(must + should + could)
        comp_devs = set(c for d in devs for c in d[dim])
        universo[dim] = sorted(set(comp_proj | comp_devs))#deduplicação explícita

    h: Dict[str, Dict[str, Any]] = {}
    y: Dict[str, Dict[str, pulp.LpVariable]] = {}
    z: Dict[str, Dict[str, pulp.LpVariable]] = {}

    for dim in DIMS:
        h[dim] = {}
        y[dim] = {}
        z[dim] = {}
        for u in universo[dim]:
            lbl = f"{LBL[dim]}_{_safe(u)}"
            h[dim][u] = pulp.lpSum(_a(devs[i], dim, u) * x[i] for i in range(N))

            y[dim][u] = pulp.LpVariable(f"y_{lbl}", cat="Binary")
            modelo += (y[dim][u] <= h[dim][u],          f"y_ub_{lbl}")
            modelo += (y[dim][u] >= (1/k) * h[dim][u], f"y_lb_{lbl}")

            z[dim][u] = pulp.LpVariable(f"z_{lbl}", cat="Binary")
            modelo += (2 * z[dim][u] <= h[dim][u],              f"z_ub_{lbl}")
            modelo += (z[dim][u] >= (1/k) * (h[dim][u] - 1),   f"z_lb_{lbl}")

    sizes = {LBL[d]: len(universo[d]) for d in DIMS}
    print(f"  [Parte 2] Cobertura y,z: DOM({sizes['DOM']}) "
          f"ECO({sizes['ECO']}) LING({sizes['LING']})")
    print(f"  [Parte 3] Features MoSCoW")

    # ------------------------------------------------------------------
    # Helpers para features full e red2  (Secao 2.4)
    # ------------------------------------------------------------------
    def _full(dim: str, prio: str) -> Any:
        idx  = {"must": 0, "should": 1, "could": 2}[prio]
        reqs = req[dim][idx]
        if not reqs:
            return pulp.lpSum([1.0])
        return (1.0 / len(reqs)) * pulp.lpSum(
            y[dim][u] for u in reqs if u in y[dim]
        )

    def _red2(dim: str, prio: str) -> Any:
        idx  = {"must": 0, "should": 1, "could": 2}[prio]
        reqs = req[dim][idx]
        if not reqs:
            return pulp.lpSum([0.0])
        return (1.0 / len(reqs)) * pulp.lpSum(
            z[dim][u] for u in reqs if u in z[dim]
        )

    # ------------------------------------------------------------------
    # Parte 4 — Escore tecnico por dimensao (Casos A/B/C)  (Secao 2.5)
    # ------------------------------------------------------------------
    dim_vars:  Dict[str, Any] = {}
    casos_dim: Dict[str, str] = {}
    casos_log = []

    for dim in DIMS:
        must, should, could = req[dim]
        lbl = LBL[dim]
        pis_ativas = [p for p, lst in
                      [("must", must), ("should", should), ("could", could)]
                      if lst]
        n_ativas = len(pis_ativas)

        # Caso A
        if n_ativas == 0:
            casos_dim[dim] = "A"
            dim_vars[dim]  = MU_NEUTRO
            casos_log.append(f"{lbl}->Caso A")

        # Caso B
        elif n_ativas == 1:
            casos_dim[dim] = "B"
            pi_star = pis_ativas[0]
            v_B     = B_INTERCEPT + B_A_FULL * _full(dim, pi_star) + B_A_RED2 * _red2(dim, pi_star)
            DIM_B   = pulp.LpVariable(f"DIM_{lbl}_B", lowBound=0.0, upBound=1.0)
            modelo += (DIM_B <= v_B, f"DIM_{lbl}_B_clamp")
            dim_vars[dim] = DIM_B
            casos_log.append(f"{lbl}->Caso B(pi={pi_star})")

        # Caso C
        else:
            casos_dim[dim] = "C"
            full_M = _full(dim, "must")
            red2_M = _red2(dim, "must")
            must_l   = [u for u in must   if u in y[dim]]
            should_l = [u for u in should if u in y[dim]]
            could_l  = [u for u in could  if u in y[dim]]

            # C.1: produto fullM * fullS (McCormick exato)
            w_MS: Dict[Tuple, pulp.LpVariable] = {}
            for ui in must_l:
                for uj in should_l:
                    vn = f"wMS_{lbl}_{_safe(ui)}_{_safe(uj)}"[:60]
                    w  = pulp.LpVariable(vn, cat="Binary")
                    w_MS[(ui, uj)] = w
                    modelo += (w <= y[dim][ui],                   f"{vn}_ub1")
                    modelo += (w <= y[dim][uj],                   f"{vn}_ub2")
                    modelo += (w >= y[dim][ui] + y[dim][uj] - 1, f"{vn}_lb")
            dMS = len(must_l) * len(should_l) if (must_l and should_l) else 1
            p_d = (1.0/dMS) * pulp.lpSum(w_MS.values()) if w_MS else pulp.lpSum([0.0])

            # C.2: produto triplo fullM * fullC * (1 - fullS)
            w_MC: Dict[Tuple, pulp.LpVariable] = {}
            for ui in must_l:
                for uj in could_l:
                    vn = f"wMC_{lbl}_{_safe(ui)}_{_safe(uj)}"[:60]
                    w  = pulp.LpVariable(vn, cat="Binary")
                    w_MC[(ui, uj)] = w
                    modelo += (w <= y[dim][ui],                   f"{vn}_ub1")
                    modelo += (w <= y[dim][uj],                   f"{vn}_ub2")
                    modelo += (w >= y[dim][ui] + y[dim][uj] - 1, f"{vn}_lb")
            v_MCS: Dict[Tuple, pulp.LpVariable] = {}
            for (ui, uj), w_mc in w_MC.items():
                for ul in should_l:
                    vn = f"vMCS_{lbl}_{_safe(ui)}_{_safe(uj)}_{_safe(ul)}"[:60]
                    v  = pulp.LpVariable(vn, cat="Binary")
                    v_MCS[(ui, uj, ul)] = v
                    modelo += (v <= w_mc,                         f"{vn}_ub1")
                    modelo += (v <= 1 - y[dim][ul],               f"{vn}_ub2")
                    modelo += (v >= w_mc + (1 - y[dim][ul]) - 1, f"{vn}_lb")
            dMCS = (len(must_l) * len(could_l) * len(should_l)
                    if (must_l and could_l and should_l) else 1)
            r_d = (1.0/dMCS) * pulp.lpSum(v_MCS.values()) if v_MCS else pulp.lpSum([0.0])

            # C.3: rho = max(eta, fullM)
            if not must_l:
                rho: Any = 1.0
            else:
                b_rho = pulp.LpVariable(f"b_rho_{lbl}", cat="Binary")
                rho   = pulp.LpVariable(f"rho_{lbl}", lowBound=ETA, upBound=1.0)
                modelo += (rho >= ETA,                   f"rho_{lbl}_eta")
                modelo += (rho >= full_M,                f"rho_{lbl}_full")
                modelo += (rho <= full_M + (1 - b_rho), f"rho_{lbl}_ub_full")
                modelo += (rho <= ETA + b_rho,           f"rho_{lbl}_ub_eta")
                modelo += (full_M >= ETA - (1 - b_rho), f"rho_{lbl}_b_cond")

            # C.4: clamp(v_C) * rho via McCormick continuo
            v_C = (C_INTERCEPT
                   + C_W_COVM  * full_M
                   + C_W_RED2M * red2_M
                   + C_W_INTMS * p_d
                   + C_W_HELPC * r_d)
            DIM_clamp = pulp.LpVariable(f"DIM_{lbl}_clamp", lowBound=0.0, upBound=1.0)
            modelo += (DIM_clamp <= v_C, f"DIM_{lbl}_clamp_ub")
            t = pulp.LpVariable(f"t_{lbl}", lowBound=0.0, upBound=1.0)
            modelo += (t <= DIM_clamp,            f"t_{lbl}_ub1")
            modelo += (t <= rho,                  f"t_{lbl}_ub2")
            modelo += (t >= DIM_clamp + rho - 1, f"t_{lbl}_lb")
            modelo += (t >= 0,                    f"t_{lbl}_nn")
            dim_vars[dim] = t
            casos_log.append(f"{lbl}->Caso C")

    print(f"  [Parte 4] {' | '.join(casos_log)}")

    # ------------------------------------------------------------------
    # Parte 5 — AT = wmean(DOM, ECO, LING)  (Secao 2.6)
    # ------------------------------------------------------------------
    soma_pesos = W_DOM + W_ECO + W_LING
    AT_expr    = (
        W_DOM  * dim_vars["dominio"]
        + W_ECO  * dim_vars["ecossistema"]
        + W_LING * dim_vars["linguagens"]
    ) / soma_pesos
    print(f"  [Parte 5] AT = ({W_DOM}*DOM + {W_ECO}*ECO + {W_LING}*LING) / {soma_pesos}")

    # ------------------------------------------------------------------
    # Partes 6 e 7 — AC e AE (somente modo AE_FULL)  (Secoes 2.7-2.8)
    # ------------------------------------------------------------------
    AC_expr: Any = None
    m_var:   Any = None
    M_var:   Any = None

    if modo == "AE_FULL":
        # Parte 6: AC_aprox linear
        AC_expr = (1.0 / k) * pulp.lpSum(
            ac_medios.get(devs[i]["id"], 0.0) * x[i] for i in range(N)
        )
        devs_com_ac = sum(1 for d in devs if ac_medios.get(d["id"], 0.0) > 0)
        print(f"  [Parte 6] AC_aprox linear | devs com AC>0: {devs_com_ac}/{N}")

        # Parte 7: AE = mixminmax(AT, AC) linearizado  (Secao 2.8)
        m_var = pulp.LpVariable("m_minAT_AC", lowBound=0.0, upBound=1.0)
        M_var = pulp.LpVariable("M_maxAT_AC", lowBound=0.0, upBound=1.0)
        b_ae  = pulp.LpVariable("b_ae", cat="Binary")

        modelo += (m_var <= AT_expr,               "m_ub_AT")
        modelo += (m_var <= AC_expr,               "m_ub_AC")
        modelo += (M_var >= AT_expr,               "M_lb_AT")
        modelo += (M_var >= AC_expr,               "M_lb_AC")
        modelo += (M_var <= AT_expr + (1 - b_ae),  "M_ub_AT")
        modelo += (M_var <= AC_expr + b_ae,         "M_ub_AC")

        AE_expr = (W_MIN * m_var + W_MAX * M_var) / (W_MIN + W_MAX)
        modelo += (AE_expr, "objetivo_AE")
        print(f"  [Parte 7] AE = ({W_MIN}*m + {W_MAX}*M) / {W_MIN+W_MAX} | mixminmax")
    else:
        modelo += (AT_expr, "objetivo_AT")
        print(f"  [Parte 5->obj] Maximizando AT diretamente (AT_ONLY)")

    print(f"  [Modelo] {modelo.numVariables()} variaveis | "
          f"{modelo.numConstraints()} restricoes")

    info = {
        "dim_vars":  dim_vars,
        "AT_expr":   AT_expr,
        "AC_expr":   AC_expr,
        "m_var":     m_var,
        "M_var":     M_var,
        "casos_dim": casos_dim,
        "req":       req,
        "x":         x,
    }
    return modelo, x, info


# ===========================================================================
# 3. RESOLVER
# ===========================================================================

def resolver_ilp(
    projeto:        Projeto,
    devs:           List[Dev],
    ac_medios:      Dict[int, float],
    modo:           str = "AT_ONLY",
    tamanho_equipe: int = 4,
    tempo_limite:   int = 120,
) -> Dict[str, Any]:
    """
    Constrói e resolve o modelo MILP para um projeto.

    Retorna:
      status, objetivo, AT, AC, AE, equipe, tempo_s,
      escores_dim, cobertura_must
    """
    k = projeto.get("team_size", tamanho_equipe)

    if len(devs) < k:
        return {
            "status": "Infeasible", "objetivo": None,
            "AT": None, "AC": None, "AE": None,
            "equipe": [], "tempo_s": 0.0,
            "escores_dim": {}, "cobertura_must": {},
        }

    modelo, x, info = construir_modelo(devs, projeto, ac_medios, modo, k)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=tempo_limite)
    t0 = time.time()
    try:
        modelo.solve(solver)
    except Exception as e:
        return {
            "status": f"Erro: {e}", "objetivo": None,
            "AT": None, "AC": None, "AE": None,
            "equipe": [], "tempo_s": round(time.time() - t0, 2),
            "escores_dim": {}, "cobertura_must": {},
        }
    tempo_s = time.time() - t0

    status  = pulp.LpStatus[modelo.status]
    obj_val = pulp.value(modelo.objective)

    if modelo.status != 1:
        return {
            "status": status, "objetivo": obj_val,
            "AT": None, "AC": None, "AE": None,
            "equipe": [], "tempo_s": round(tempo_s, 2),
            "escores_dim": {}, "cobertura_must": {},
        }

    selecionados = [i for i in range(len(devs)) if pulp.value(x[i]) > 0.5]
    equipe_ids   = [devs[i]["id"] for i in selecionados]

    # Escores por dimensao
    escores_dim: Dict[str, float] = {}
    for dim in ["dominio", "ecossistema", "linguagens"]:
        var = info["dim_vars"][dim]
        val = pulp.value(var) if not isinstance(var, (int, float)) else var
        escores_dim[dim] = round(val or 0.0, 6)

    soma_p = W_DOM + W_ECO + W_LING
    AT_val = (W_DOM  * escores_dim["dominio"]
            + W_ECO  * escores_dim["ecossistema"]
            + W_LING * escores_dim["linguagens"]) / soma_p

    AC_val: Optional[float] = None
    AE_val: Optional[float] = None
    if modo == "AE_FULL":
        m_v    = pulp.value(info["m_var"]) or 0.0
        M_v    = pulp.value(info["M_var"]) or 0.0
        AC_val = round(pulp.value(info["AC_expr"]) or 0.0, 6)
        AE_val = round((W_MIN * m_v + W_MAX * M_v) / (W_MIN + W_MAX), 6)

    # Cobertura Must
    req = info["req"]
    cobertura_must: Dict[str, bool] = {}
    for dim in ["dominio", "ecossistema", "linguagens"]:
        for u in req[dim][0]:
            cobre = any(_a(devs[i], dim, u) == 1 for i in selecionados)
            cobertura_must[f"{dim}.{u}"] = cobre

    return {
        "status":         status,
        "objetivo":       round(obj_val or 0.0, 6),
        "AT":             round(AT_val, 6),
        "AC":             AC_val,
        "AE":             AE_val,
        "equipe":         equipe_ids,
        "tempo_s":        round(tempo_s, 2),
        "escores_dim":    escores_dim,
        "cobertura_must": cobertura_must,
        "_selecionados":  selecionados,
        "_devs":          devs,
        "_info":          info,
    }


# ===========================================================================
# 4. EXIBICAO DE RESULTADO
# ===========================================================================

def exibir_resultado(resultado: Dict[str, Any], projeto: Projeto) -> None:
    """Imprime logs detalhados pos-solucao."""
    status = resultado["status"]
    print(f"\n[ILP] Solver: {status} em {resultado['tempo_s']}s")

    if status != "Optimal":
        print(f"  Solver nao encontrou solucao otima.")
        return

    if resultado["AE"] is not None:
        print(f"[ILP] AT={resultado['AT']:.4f}  "
              f"AC={resultado['AC']:.4f}  AE={resultado['AE']:.4f}")
    else:
        print(f"[ILP] AT={resultado['AT']:.4f}")

    devs         = resultado["_devs"]
    selecionados = resultado["_selecionados"]

    print(f"\n  Equipe selecionada:")
    for i in selecionados:
        d = devs[i]
        print(f"    Dev {d['id']:>4} | {d['nome']:<30} "
              f"DOM={d['dominio'][:3]} ECO={d['ecossistema'][:3]} "
              f"LING={d['linguagens'][:3]}")

    print(f"\n  Must cobertos:")
    for chave, ok in resultado["cobertura_must"].items():
        dim, comp = chave.split(".", 1)
        print(f"    {dim[:4].upper():<5} {'v' if ok else 'X'}  {comp}")

    print(f"\n  Escores: ", end="")
    casos = resultado["_info"]["casos_dim"]
    for dim, val in resultado["escores_dim"].items():
        print(f"{dim[:4].upper()}={val:.4f}(C{casos[dim]}) ", end="")
    print()


# ===========================================================================
# 5. EXEMPLO — use bench_stfp.py para rodar
# ===========================================================================
# if __name__ == "__main__":
#     devs      = carregar_base("base_final_com_rdas_anonimizado.json")
#     projetos  = carregar_projetos("target_projects.json")
#     grafo     = carregar_grafo("Graph_DB_real.json")
#     ac_medios = calcular_ac_medios(devs, grafo)
#     resultado = resolver_ilp(projetos["P1"], devs, ac_medios, modo="AT_ONLY")
#     exibir_resultado(resultado, projetos["P1"])
