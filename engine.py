# === GA4STF.py (TOP DO ARQUIVO) ===
import sys, os, re, json, time, copy, random, argparse
from datetime import datetime
from typing import Tuple



# cache de fitness e chave canônica de equipe
FITNESS_CACHE = {}  # chave: tuple(sorted(team_ids)) -> dict(res)

def _key(team) -> Tuple[int, ...]:
    """Chave canônica para identificar a equipe, independente de ordem."""
    return tuple(sorted(int(x) for x in team))

# -------------------- resolver caminho do projeto --------------------
# BASE_DIR = pasta deste arquivo ( ...\STFP\Algorithms\ga )
BASE_DIR = os.path.dirname(__file__)
# PROJECT_ROOT = pasta ACIMA de STFP ( ...\Works )
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# permitir importar evaluate_teams.py (via pacote STFP)
#from STFP.Pipeline.evaluate_teams import avaliar_equipe
#from Pipeline.evaluate_teams import avaliar_equipe
# usando surrogate
from Pipeline.evaluate_teams_sur import avaliar_equipe_surrogate as avaliar_equipe
# permitir importar evaluate_teams.py
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from STFP.Pipeline.evaluate_teams import avaliar_equipe

# -------------------- paths --------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, '..', '..', 'Data', 'base_final.json')
GRAFO_GRAPH_PATH = os.path.join(BASE_DIR, '..', '..', 'Data', 'Graph_DB_real.json')

# -------------------- REPORT / LOG (ok vir antes de qualquer print) --------------------
REPORT_DIR = os.path.join(BASE_DIR, "Reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_TEE_ORIG_STDOUT = None
_TEE_FILE_HANDLE = None

def enable_run_report(run_name="GA_run"):
    global _TEE_ORIG_STDOUT, _TEE_FILE_HANDLE
    latest_path  = os.path.join(REPORT_DIR, f"{run_name}_latest.txt")
    stamped_path = os.path.join(REPORT_DIR, f"{run_name}_{_timestamp()}.txt")
    fh_latest  = open(latest_path,  "w", encoding="utf-8")
    fh_stamped = open(stamped_path, "w", encoding="utf-8")
    _TEE_ORIG_STDOUT = sys.stdout
    sys.stdout = _TeeIO(sys.stdout, fh_latest, fh_stamped)
    _TEE_FILE_HANDLE = (fh_latest, fh_stamped)
    if False:
        print(f"[REPORT] Logging habilitado. latest='{latest_path}', stamped='{stamped_path}'")
    return latest_path, stamped_path

def disable_run_report():
    global _TEE_ORIG_STDOUT, _TEE_FILE_HANDLE
    if _TEE_ORIG_STDOUT is not None:
        sys.stdout.flush()
        sys.stdout = _TEE_ORIG_STDOUT
        _TEE_ORIG_STDOUT = None
    if _TEE_FILE_HANDLE is not None:
        for fh in _TEE_FILE_HANDLE:
            try: fh.close()
            except: pass
        _TEE_FILE_HANDLE = None

# artefatos estruturados
import csv
RUN_SUMMARY = []

def _append_run_row(gen, team, res_dict):
    row = {
        "generation": gen,
        "team": list(team),
        "fitness": res_dict.get("media_AE"),

        # distribuição AE
        "dist_VL": None, "dist_L": None, "dist_M": None, "dist_H": None, "dist_VH": None,

        # info técnica 
        "dom_score": res_dict.get("scores", {}).get("dominio", {}).get("score")
                      if isinstance(res_dict.get("scores"), dict) else
                      (res_dict.get("dominio", {}).get("score")
                       if isinstance(res_dict.get("dominio"), dict) else None),
        "dom_label": res_dict.get("scores", {}).get("dominio", {}).get("rotulo")
                      if isinstance(res_dict.get("scores"), dict) else
                      (res_dict.get("dominio", {}).get("rotulo")
                       if isinstance(res_dict.get("dominio"), dict) else None),
        "fullP": res_dict.get("dominio", {}).get("fullP")
                 if isinstance(res_dict.get("dominio"), dict) else None,

        # colaboração agregada original (se você já usa)
        "pc_percent": res_dict.get("pc_percent"),

        # NOVO: AT e AC contínuos (para os cenários)
        "AT_cont": res_dict.get("AT_cont"),
        "AC_cont": res_dict.get("AC_cont"),
    }

    dist = res_dict.get("distribuicao")
    if isinstance(dist, (list, tuple)) and len(dist) == 5:
        row["dist_VL"], row["dist_L"], row["dist_M"], row["dist_H"], row["dist_VH"] = dist

    RUN_SUMMARY.append(row)


def _save_jsonl_and_csv(run_name="GA_run"):
    # Paths
    jsonl_latest = os.path.join(REPORT_DIR, f"{run_name}_latest.jsonl")
    jsonl_stamp  = os.path.join(REPORT_DIR, f"{run_name}_{_timestamp()}.jsonl")

    # Salva JSONL (latest + stamped) com TODAS as linhas de RUN_SUMMARY
    with open(jsonl_latest, "w", encoding="utf-8") as f_latest, \
         open(jsonl_stamp,  "w", encoding="utf-8") as f_stamp:
        for r in RUN_SUMMARY:
            line = json.dumps(r, ensure_ascii=False)
            f_latest.write(line + "\n")
            f_stamp.write(line + "\n")

    # Monta best_by_gen para o CSV top por geração
    best_by_gen = {}
    for r in RUN_SUMMARY:
        g = r["generation"]
        cur = best_by_gen.get(g)
        if (cur is None) or (r.get("fitness", -1) > cur.get("fitness", -1)):
            best_by_gen[g] = r

    # IMPORTANTE: header inclui AT_cont e AC_cont
    header = [
        "generation",
        "fitness",
        "team",
        "dom_score",
        "dom_label",
        "fullP",
        "dist_VL",
        "dist_L",
        "dist_M",
        "dist_H",
        "dist_VH",
        "pc_percent",
        "AT_cont",
        "AC_cont",
    ]

    csv_top_path = os.path.join(REPORT_DIR, f"{run_name}_top_by_generation_latest.csv")
    with open(csv_top_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for g in sorted(best_by_gen):
            # garante que só escreve colunas do header
            row = {k: best_by_gen[g].get(k) for k in header}
            w.writerow(row)
    if False:
        print(f"[REPORT] JSONL latest: {jsonl_latest}")
        print(f"[REPORT] JSONL stamped: {jsonl_stamp}")
        print(f"[REPORT] CSV (top por geração) salvo em: {csv_top_path}")
    return jsonl_latest, csv_top_path


# -------------------- utils --------------------
def fmt_dur(segundos: float) -> str:
    m, s = divmod(segundos, 60)
    return f"{int(m)} min {s:05.2f} s"

def _csv_list(s: str):
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _carregar_devs(db_path):
    with open(db_path, encoding="utf-8") as f:
        raw = json.load(f)
    devs = raw if isinstance(raw, list) else raw.get("developers", [])
    norm = []
    for d in devs:
        d2 = dict(d)
        if "id" not in d2 and "user_id" in d2: #ler dois formatos de base de dados (antiga e nova)
            d2["id"] = d2["user_id"]
        d2["ecossistema"] = ([d2["ecossistema"]] if isinstance(d2.get("ecossistema"), str)
                             else (d2.get("ecossistema") or []))
        d2["linguagens"]  = ([d2["linguagens"]] if isinstance(d2.get("linguagens"), str)
                             else (d2.get("linguagens") or []))
        projs_in = d2.get("projects", []) or d2.get("projectHistory", [])
        d2["projects"] = [{"id": p.get("id") or p.get("projectId"),
                           "osf": p.get("osf") if "osf" in p else p.get("NPS"),
                           "slf": p.get("slf") if "slf" in p else p.get("SLF")} for p in projs_in]
        norm.append(d2)
    return norm

def _as_int(x):
    if x is None: return None
    if isinstance(x, int): return x
    m = re.search(r'(\d+)$', str(x))
    return int(m.group(1)) if m else None

def _load_grafo(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    edges = data.get('edges') or data.get('links') or (data if isinstance(data, list) else [])
    mapa, adj, ids = {}, {}, set()
    for e in edges:
        u = _as_int(e.get('source_user_id')) or _as_int(e.get('source'))
        v = _as_int(e.get('target_user_id')) or _as_int(e.get('target'))
        if u is None or v is None: continue
        w = float(e.get('weight', 0.0))  # mantém se você usar PCs
        a, b = (u, v) if u < v else (v, u)
        mapa[(a, b)] = max(mapa.get((a, b), 0.0), w)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        ids.update([u, v])
    return mapa, adj, ids

# >>> SOMENTE AGORA use as funções acima
MAPA_PESOS_GRAFO, ADJ, VALID_IDS = _load_grafo(GRAFO_GRAPH_PATH)
DEVS = [d for d in _carregar_devs(DB_PATH) 
        if (d.get("id") or d.get("user_id")) is not None
        and int(d.get("id") or d.get("user_id")) in VALID_IDS]
CANDIDATOS_IDS = [int(d.get("id") or d.get("user_id")) for d in DEVS]


# -------------------- args --------------------
parser = argparse.ArgumentParser(description="GA para formação de equipes.")
parser.add_argument("--team-size", type=int, default=4, help="Tamanho da equipe")
parser.add_argument("--dom-m",   type=str, default="Cloud", help="Domínio MUST, CSV")
parser.add_argument("--dom-s",   type=str, default="Web",   help="Domínio SHOULD, CSV")
parser.add_argument("--dom-c",   type=str, default="AI",    help="Domínio COULD, CSV")
parser.add_argument("--eco-s",   type=str, default="", help="Ecossistema SHOULD, CSV")
parser.add_argument("--eco-c",   type=str, default="", help="Ecossistema COULD, CSV")
parser.add_argument("--ling-s",  type=str, default="", help="Linguagens SHOULD, CSV")
parser.add_argument("--ling-c",  type=str, default="", help="Linguagens COULD, CSV")
args = parser.parse_args()

# -------------------- projeto-alvo --------------------
PROJETO_ALVO = {
    "dominio": {
        "must":   _csv_list(args.dom_m),
        "should": _csv_list(args.dom_s),
        "could":  _csv_list(args.dom_c),
    },
    "ecossistema": {
        "should": _csv_list(args.eco_s),
        "could":  _csv_list(args.eco_c),
    },
    "linguagens": {
        "should": _csv_list(args.ling_s),
        "could":  _csv_list(args.ling_c),
    },
    "tamanhoEquipe": args.team_size,
}

###################################################

def _as_int(x):
    if x is None: return None
    if isinstance(x, int): return x
    m = re.search(r'(\d+)$', str(x))
    return int(m.group(1)) if m else None

def _load_grafo(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    edges = data.get('edges') or data.get('links') or (data if isinstance(data, list) else [])
    mapa, adj, ids = {}, {}, set()
    for e in edges:
        u = _as_int(e.get('source_user_id')) or _as_int(e.get('source'))
        v = _as_int(e.get('target_user_id')) or _as_int(e.get('target'))
        if u is None or v is None: continue
        w = float(e.get('weight', 0.0))
        a, b = (u, v) if u < v else (v, u)
        mapa[(a, b)] = max(mapa.get((a, b), 0.0), w)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        ids.update([u, v])
    return mapa, adj, ids

MAPA_PESOS_GRAFO, ADJ, VALID_IDS = _load_grafo(GRAFO_GRAPH_PATH)

# filtrar DEVS por IDs presentes no grafo
DEVS = [d for d in DEVS if int(d["id"]) in VALID_IDS]
CANDIDATOS_IDS = [int(d["id"]) for d in DEVS]
if len(CANDIDATOS_IDS) < PROJETO_ALVO["tamanhoEquipe"]:
    raise RuntimeError("Candidatos insuficientes presentes no grafo para formar uma equipe.")


#novo trecho para convexidade
# ----------------------------------------------------------------------
# Fit técnico por desenvolvedor (proxy normalizada em [0,1])
# ----------------------------------------------------------------------
def build_dev_tech_fit(devs, default=0.5):
    """
    Constrói um score técnico escalar por desenvolvedor, normalizado em [0,1].
    Usa média de OSF/SLF (ou NPS/SLF) dos projetos como proxy.
    Se não houver histórico, usa 'default'.
    """
    raw = {}
    for d in devs:
        dev_id = int(d["id"])
        projs = d.get("projects", [])
        vals = []
        for p in projs:
            for k in ("osf", "OSF", "NPS", "slf", "SLF"):
                if p.get(k) is not None:
                    try:
                        vals.append(float(p[k]))
                    except (TypeError, ValueError):
                        pass
        raw[dev_id] = (sum(vals) / len(vals)) if vals else default

    if not raw:
        # fallback: todo mundo igual
        return {int(d["id"]): 0.5 for d in devs}

    vmin = min(raw.values())
    vmax = max(raw.values())
    if vmax > vmin:
        return {dev_id: (v - vmin) / (vmax - vmin) for dev_id, v in raw.items()}
    else:
        # todos os valores iguais → centraliza em 0.5
        return {dev_id: 0.5 for dev_id in raw.keys()}


DEV_TECH_FIT = build_dev_tech_fit(DEVS)

#########################33



def _mutacao_forcada_team(team):
    """
    Força a troca de 1 membro por outro para gerar uma equipe diferente.
    
    """
    t = list(team)
    if not t:
        return t
    idx = random.randrange(len(t))
    pool = [i for i in CANDIDATOS_IDS if i not in t]
    if not pool:
        return t
    t[idx] = random.choice(pool)
    return t

def _force_mutate_until_unique_team(team, seen_keys, max_tries=30):
    """
    Aplica mutações forçadas até gerar uma equipe com chave não vista.
    
    """
    t = list(team)
    for _ in range(max_tries):
        t = _mutacao_forcada_team(t)
        k = _key(t)
        if k not in seen_keys:
            return t
    return t


# -------------------- GA params --------------------
# Após fine tuning 24 de maio
TAM_POP       = 100
MAX_GERACOES  = 100

ELITISMO      = 1
TAXA_MUTACAO  = 0.05
TAXA_CROSSOVER = 1.0  # Valor default (comportamento atual)
CROSSOVER_OPERATOR = "ccc"   # "ccc" | "one_point"
CENTROIDES    = [0.1, 0.3, 0.5, 0.7, 0.9]

# -------------------- GA ops --------------------

def gerador_cromossomo():
    # equipe aleatória, sem checar arestas/clique
    eq = random.sample(CANDIDATOS_IDS, PROJETO_ALVO['tamanhoEquipe'])
    return {'equipe': eq, 'fitness': 0.0}

#def crossover(a, b):
#    filho1, filho2 = copy.deepcopy(a), copy.deepcopy(b)
#    if len(a['equipe']) > 1:
#        ponto = random.randint(0, len(a['equipe'])-1)
#        filho1['equipe'][:ponto], filho2['equipe'][:ponto] = (
#            b['equipe'][:ponto], a['equipe'][:ponto])
#    
#    return filho1, filho2

#novo crosover convexo
def _local_fit(dev_id: int) -> float:
    """Retorna o fit técnico escalar do dev (proxy), em [0,1]."""
    return DEV_TECH_FIT.get(dev_id, 0.5)


def _pick_replacement(target_fit: float, forbidden_ids):
    """
    Se o gene escolhido causar duplicata, escolhe outro dev em CANDIDATOS_IDS
    com fit técnico mais próximo de target_fit e que não esteja em forbidden_ids.
    """
    best_dev = None
    best_delta = None
    forb = set(forbidden_ids)
    for dev_id, val in DEV_TECH_FIT.items():
        if dev_id in forb:
            continue
        delta = abs(val - target_fit)
        if best_dev is None or delta < best_delta:
            best_dev = dev_id
            best_delta = delta

    if best_dev is not None:
        return best_dev

    # fallback extremo (quase nunca acontece)
    pool = [d for d in CANDIDATOS_IDS if d not in forb]
    return random.choice(pool) if pool else random.choice(CANDIDATOS_IDS)


def _make_convex_child(parent_a_ids, parent_b_ids):
    """
    Gera um filho aplicando combinação convexa gene a gene.
    parent_a_ids / parent_b_ids: listas de ids de devs na mesma posição.
    """
    size = len(parent_a_ids)
    child = []

    for i in range(size):
        dev_a = parent_a_ids[i]
        dev_b = parent_b_ids[i]

        fit_a = _local_fit(dev_a)
        fit_b = _local_fit(dev_b)

        alpha = random.random()  # α ∈ [0,1]
        fit_c = alpha * fit_a + (1.0 - alpha) * fit_b

        # escolhe o pai mais próximo de fit_c
        delta_a = abs(fit_c - fit_a)
        delta_b = abs(fit_c - fit_b)
        chosen = dev_a if delta_a <= delta_b else dev_b

        # garante que não haja duplicata no filho
        if chosen in child:
            chosen = _pick_replacement(fit_c, forbidden_ids=child)

        child.append(chosen)

    return child


def crossover(a, b):
    """
    Crossover baseado em combinação convexa:
    para cada posição i, gera α, combina fits locais dos devs dos pais
    e escolhe o dev cujo fit está mais próximo de fit_c.
    Retorna dois filhos no formato {'equipe': [...], 'fitness': 0.0}.
    """
    team_a = list(a['equipe'])
    team_b = list(b['equipe'])

    # Garante mesmo tamanho (por segurança)
    size = min(len(team_a), len(team_b))
    team_a = team_a[:size]
    team_b = team_b[:size]

    child1_ids = _make_convex_child(team_a, team_b)
    child2_ids = _make_convex_child(team_a, team_b)

    filho1 = {'equipe': child1_ids, 'fitness': 0.0}
    filho2 = {'equipe': child2_ids, 'fitness': 0.0}
    return filho1, filho2

def crossover_one_point(a, b):
    """
    One-point crossover: corta em ponto aleatório [1, size-1],
    child1 = a[:p] + b[p:], child2 = b[:p] + a[p:].
    Duplicatas resolvidas via _pick_replacement.
    """
    team_a = list(a['equipe'])
    team_b = list(b['equipe'])
    size = min(len(team_a), len(team_b))
    team_a, team_b = team_a[:size], team_b[:size]
    if size < 2:
        return ({'equipe': list(team_a), 'fitness': 0.0},
                {'equipe': list(team_b), 'fitness': 0.0})
    ponto = random.randint(1, size - 1)

    def _build_child(first_seg, second_seg):
        child = list(first_seg)
        seen  = set(child)
        for gene in second_seg:
            if gene not in seen:
                child.append(gene)
                seen.add(gene)
            else:
                repl = _pick_replacement(_local_fit(gene), forbidden_ids=seen)
                child.append(repl)
                seen.add(repl)
        return child

    child1 = _build_child(team_a[:ponto], team_b[ponto:])
    child2 = _build_child(team_b[:ponto], team_a[ponto:])
    return ({'equipe': child1, 'fitness': 0.0},
            {'equipe': child2, 'fitness': 0.0})

#####################

def mutacao(cromo):
    if random.random() < TAXA_MUTACAO:
        idx = random.randrange(len(cromo['equipe']))
        pool = [i for i in CANDIDATOS_IDS if i not in cromo['equipe']]
        if pool:
            cromo['equipe'][idx] = random.choice(pool)
    return cromo


def _build_next_population(pop_atual, pop_size, elitism_k=1):
    """
    Gera a próxima população sem duplicatas na MESMA geração:
      - Mantém 'elitism_k' elites únicos.
      - Cria filhos por seleção → crossover → mutação .
      - Rejeita duplicatas; se duplicado, força mutação até diferenciar.
    """
    
    # 1) Elites únicos (copiados sem fitness/dist para reavaliar)
    pop_ord = sorted(pop_atual, key=lambda c: c['fitness'], reverse=True)
    elites, seen = [], set()
    for c in pop_ord:
        k = _key(c['equipe'])
        if k in seen:
            continue
        elites.append({'equipe': list(c['equipe'])})
        seen.add(k)
        if len(elites) >= elitism_k:
            break

    # 2) Preencher com filhos únicos
    filhos = []
    # seleção: mesma lógica que você já usa (topo da população atual)
    k_pool = max(2, pop_size // 2)
    MAX_TRIES = 40 * (pop_size - len(elites))  # margem anti-loop
    tries = 0

    while len(elites) + len(filhos) < pop_size and tries < MAX_TRIES:
        tries += 1
        p1, p2 = random.sample(pop_ord[:k_pool], 2)

        # sua função crossover dá DOIS filhos em dicts {'equipe': [...]}
        if random.random() < TAXA_CROSSOVER:
            if CROSSOVER_OPERATOR == "one_point":
                f1, f2 = crossover_one_point(p1, p2)
            else:
                f1, f2 = crossover(p1, p2)
        else:
            # Filhos são cópias dos pais (sem recombinação)
            f1 = {'equipe': list(p1['equipe']), 'fitness': 0.0}
            f2 = {'equipe': list(p2['equipe']), 'fitness': 0.0}

        # mutação normal + repair (reutiliza suas funções)
        f1 = mutacao(f1)
        f2 = mutacao(f2)
        

        # tentar inserir f1
        if len(elites) + len(filhos) < pop_size:
            k1 = _key(f1['equipe'])
            if k1 in seen:
                # forçar mutação até ficar único (continua sendo operador genético)
                f1['equipe'] = _force_mutate_until_unique_team(f1['equipe'], seen, max_tries=30)
                k1 = _key(f1['equipe'])
            if k1 not in seen:
                filhos.append({'equipe': list(f1['equipe'])})
                seen.add(k1)

        # tentar inserir f2
        if len(elites) + len(filhos) < pop_size:
            k2 = _key(f2['equipe'])
            if k2 in seen:
                f2['equipe'] = _force_mutate_until_unique_team(f2['equipe'], seen, max_tries=30)
                k2 = _key(f2['equipe'])
            if k2 not in seen:
                filhos.append({'equipe': list(f2['equipe'])})
                seen.add(k2)

    # fallback (raro): se não preencher (espaço muito restrito), completa com elites diferentes
    nova_pop = elites + filhos
    if len(nova_pop) < pop_size:
        for c in pop_ord:
            k = _key(c['equipe'])
            if k in seen:
                continue
            nova_pop.append({'equipe': list(c['equipe'])})
            seen.add(k)
            if len(nova_pop) >= pop_size:
                break

    return nova_pop


#def avaliar_pop(pop, ger):
def avaliar_pop(pop, ger, log_eval=False, verbose=False):
    for i, c in enumerate(pop, 1):
        if False:
            print(f"\n[TEAM #{i:02d}] ids={c['equipe']}  (ger={ger})")

        k = _key(c['equipe'])
        if k in FITNESS_CACHE:
            res = FITNESS_CACHE[k]
        else:
            #res = avaliar_equipe(c['equipe'], PROJETO_ALVO, log=True)
            res = avaliar_equipe(c['equipe'], PROJETO_ALVO, log=log_eval)
            FITNESS_CACHE[k] = res

        c['fitness'] = res['media_AE']
        c['dist']    = res['distribuicao']
        _append_run_row(ger, c['equipe'], res)
    if False:
        print(f"\n==== GERAÇÃO {ger} avaliada ====")
        for j, c in enumerate(pop, 1):
            print(f" {j:02d}  Eq={c['equipe']}  Fit={c['fitness']:.4f}")
        _diagnostico_pop(pop)
    return pop

def _diagnostico_pop(pop):
    # 1) equipes únicas?
    teams = [tuple(sorted(c['equipe'])) for c in pop]
    if False:
        print("[CHK] equipes únicas na geração:", len(set(teams)) == len(teams))

    # 2) contagem de ocorrência por dev (mostra compartilhamento)
    from collections import Counter
    counts = Counter([m for c in pop for m in c['equipe']])
    top = counts.most_common(5)
    if False:
        print("[CHK] devs mais usados (dev, vezes):", top)
# -------------------- main loop --------------------

def run_ga_com_config(PROJETO_ALVO_EXTERNO, team_size, pop_size=100, geracoes=100, seed=123,
                      crossover_rate=1.0, mutation_rate=0.05,
                      verbose=False, report=False, log_eval=False, stable_gens=10,
                      crossover_operator="ccc"):
    # Habilita relatório
    if report:
        enable_run_report(run_name="GA_run")
    if False:
        print(f"[DEBUG] pop_size={pop_size} geracoes={geracoes} team_size={team_size}")
    """
    Executa o Algoritmo Genético com as configurações recebidas externamente.

    Args:
        PROJETO_ALVO_EXTERNO (dict): Dicionário com domínio, ecossistema e linguagens do projeto.
        team_size (int): Tamanho da equipe.
        pop_size (int): Tamanho da população.
        geracoes (int): Número de gerações.
        seed (int): Semente aleatória para reprodutibilidade.

    Returns:
        dict: {'best_team': [...], 'best_fitness': valor, 'duration_sec': segundos}
    """
    import time, random

    global PROJETO_ALVO, TAM_POP, MAX_GERACOES, TAXA_CROSSOVER, TAXA_MUTACAO, CROSSOVER_OPERATOR
    TAXA_CROSSOVER = crossover_rate
    TAXA_MUTACAO = mutation_rate
    CROSSOVER_OPERATOR = crossover_operator
    random.seed(seed)

    RUN_SUMMARY.clear()
    FITNESS_CACHE.clear()  # ← limpa cache entre execuções

    PROJETO_ALVO = dict(PROJETO_ALVO_EXTERNO)
    PROJETO_ALVO["tamanhoEquipe"] = team_size
    TAM_POP = pop_size
    MAX_GERACOES = int(geracoes)


    inicio = time.perf_counter()

    # Criação inicial da população
    pop = [gerador_cromossomo() for _ in range(TAM_POP)]
    pop = avaliar_pop(pop, 0, log_eval=log_eval, verbose=verbose)

    # --- early stop: N gerações estáveis sem melhorar o melhor AE_mean já visto ---
    STABLE_GENS = stable_gens
    MIN_DELTA = 1e-4  # ajuste se precisar (1e-6 mais sensível, 1e-3 mais rígido)

    best_ind = max(pop, key=lambda x: x['fitness'])
    best_ever = best_ind['fitness']
    best_team_ever = list(best_ind['equipe'])
    stable_count = 0

    stop_reason = "max_gens"
    last_gen = 0

    # Loop principal do GA
    #for g in range(1, MAX_GERACOES + 1):
    #    pop.sort(key=lambda x: x['fitness'], reverse=True)
    #    nova = pop[:ELITISMO]
    #    while len(nova) < TAM_POP:
    #        p1, p2 = random.sample(pop[:max(2, TAM_POP // 2)], 2)
    #        f1, f2 = crossover(p1, p2)
    #        nova.extend([mutacao(f1), mutacao(f2)])
    #    pop = avaliar_pop(nova[:TAM_POP], g)
    for g in range(1, MAX_GERACOES + 1):
        # NOVA POPULAÇÃO SEM DUPLICATAS NA MESMA GERAÇÃO
        pop = _build_next_population(pop, pop_size=TAM_POP, elitism_k=ELITISMO)

        # Avaliar (usa cache)
        pop = avaliar_pop(pop, g, log_eval=log_eval, verbose=verbose)

        # --- checar melhoria do melhor AE_mean (best fitness) ---
        best_gen = max(pop, key=lambda x: x['fitness'])
        if best_gen['fitness'] > best_ever + MIN_DELTA:
            best_ever = best_gen['fitness']
            best_team_ever = list(best_gen['equipe'])
            stable_count = 0
        else:
            stable_count += 1

        # --- parar se ficar 5 gerações seguidas sem melhorar ---
        if stable_count >= STABLE_GENS:
            stop_reason = "stable_gens_no_improve"
            last_gen = g
            if False:
                print(f"[EARLY STOP] {STABLE_GENS} gerações sem melhorar AE_mean. Parando em g={g}.")
            break
        last_gen = g



    dur = time.perf_counter() - inicio
    # usa o melhor global (best-so-far), não apenas o melhor da última população
    best = {"equipe": best_team_ever, "fitness": best_ever}

    if False:
        print(f"\nGA concluído em {fmt_dur(dur)}  (total {dur:.2f} s)")
        print(f"[STOP] reason={stop_reason} gens_executed={last_gen} stable_observed={stable_count}")

    # Salva artefatos estruturados (JSONL + CSV top por geração)
    if report:
        _save_jsonl_and_csv(run_name="GA_run")
        disable_run_report()

    # ── Seleciona top-2 com diversidade mínima ────────────────────────────
    def _jaccard_dist(a, b):
        A, B = set(a), set(b)
        return 1.0 - len(A & B) / len(A | B)

    # ordena pop final por fitness, remove duplicatas exatas
    pop_unica = []
    seen_keys = set()
    for ind in sorted(pop, key=lambda x: x["fitness"], reverse=True):
        k = tuple(sorted(int(x) for x in ind["equipe"]))
        if k not in seen_keys:
            seen_keys.add(k)
            pop_unica.append(ind)

    # seleciona 2 equipes com distância mínima, relaxando se necessário
    selecionadas = [{"equipe": best["equipe"], "fitness": best["fitness"]}]
    for min_dist in (0.5, 0.4, 0.3, 0.2, 0.1):
        for ind in pop_unica:
            if len(selecionadas) >= 4:
                break
            if all(_jaccard_dist(ind["equipe"], s["equipe"]) >= min_dist
                   for s in selecionadas):
                selecionadas.append({"equipe": list(ind["equipe"]), "fitness": ind["fitness"]})
        if len(selecionadas) >= 4:
            break

    # garante sempre 4 (caso população seja muito pequena)
    if len(selecionadas) < 4:
        selecionadas.append(selecionadas[0])

    top_teams = [
        {"team": s["equipe"], "fitness": s["fitness"]}
        for s in selecionadas
    ]

    return {
        "best_team":    top_teams[0]["team"],     # mantém compatibilidade
        "best_fitness": top_teams[0]["fitness"],
        "top_teams":    top_teams,                # NOVO: lista com 2
        "duration_sec": dur,
        "gens_executed": last_gen,
        "stable_gens_param": STABLE_GENS,
        "stable_gens_observed": stable_count,
        "stop_reason": stop_reason,
    }


# Protege a execução direta
if __name__ == "__main__":
    pass




