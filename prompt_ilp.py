"""
PROMPT  — Implementação ILP/MILP para STFP (AT + AC + AE)
=====================================================================

Este prompt descreve a implementação completa do modelo de Programação
Linear Inteira Mista (MILP) para o Problema de Formação de Equipes de
Software (STFP), incluindo os três modos de operação (AT_ONLY,
AE_FULL_APROX, AE_FULL_EXACT) e o script de bench que orquestra
os experimentos.

ARQUIVOS QUE DEVEM SER ANEXADOS (exatamente estes 4):
  1. formalizacao_STFP_linearizado.pdf  — formalização matemática linearizada
  2. base_final.json                    — base de desenvolvedores (507 devs)
  3. target_projects.json               — projetos-alvo P1-P12
  4. Graph_DB_real.json                 — grafo de colaboração real

Tudo mais está contido neste prompt.
=====================================================================
"""

PROMPT = """
Você é um assistente especializado em Pesquisa Operacional e otimização combinatória.
Implemente dois arquivos Python para resolver o Problema de Formação de Equipes de
Software (STFP) como modelo de Programação Linear Inteira Mista (MILP) usando PuLP:

  1. stfp_ilp.py   — modelo MILP completo (AT + AC + AE), recebe MODO como parâmetro
  2. bench_stfp.py — script de experimento que define MODO e orquestra os projetos

O experimento principal foi rodado no modo AE_FULL_EXACT (AC exato via
variáveis binárias p_ij por par de devs, com valor Low=0.3 para pares
sem colaboração prévia no grafo), cobrindo 12 projetos-alvo (P1-P12)
com a base real de 507 desenvolvedores do Virtus/UFPB.

=======================================================================
1. CONTEXTO
=======================================================================

O STFP seleciona uma equipe de tamanho k que maximize a aptidão global
(AE) em relação a um projeto-alvo com requisitos MoSCoW em três
dimensões técnicas (domínio, ecossistema, linguagens). AE combina
aptidão técnica (AT) e colaborativa (AC) via função mixminmax.

O modelo é usado como baseline de comparação com um Algoritmo Genético
(GA) na tese de doutorado sobre o sistema TeamPlus — PPGI/UFPB.

IMPORTANTE SOBRE OS DADOS DE AC:
A base de dados real não possui métricas de avaliação de desempenho
colaborativo (OSF/SLF). O AC é estimado a partir do grafo de colaboração
real (Graph_DB_real.json), que contém para cada par de devs o peso
PC_ij = f(N), onde N é o número de projetos em comum e f é a função de
saturação calibrada com especialista:
  f(N) = N/4                    se N ≤ 4  (N=1→0.25, N=4→1.00)
  f(N) = max(0, 1 - 0.1*(N-4)) se N > 4  (fadiga: N=5→0.90, N=14→0.00)

Pares sem colaboração (N=0, ausentes do grafo) recebem o centróide
"Low" = 0.3, seguindo regra do especialista de domínio:
ausência de colaboração prévia → faixa Low da escala VL/L/M/H/VH.

=======================================================================
2. FORMALIZAÇÃO MATEMÁTICA
=======================================================================

A formalização completa está no PDF anexado. Leia-o integralmente antes
de implementar. Seções relevantes:

  - 2.2: variáveis x_i, y_u^δ, z_u^δ
  - 2.3: contagem h_u^δ e cobertura simples (y) e redundante (z)
  - 2.4: features full e red2 por prioridade MoSCoW
  - 2.5: escore por dimensão — Caso A (neutro), Caso B (sit2, clamp
         linearizado), Caso C (sit1, McCormick exato, produto triplo,
         ρ = max(η,·), clamp + produto contínuo com ρ via McCormick)
  - 2.6: AT = wmean(DOM, ECO, LING) com pesos 3/1/5
  - 2.7: AC = valor esperado dos pesos PC_ij dos pares da equipe
  - 2.8: AE = mixminmax(AT, AC) linearizado com variáveis m, M, b
  - 2.9: modelo MILP consolidado

=======================================================================
3. COEFICIENTES CALIBRADOS (embutidos — não precisam de arquivo)
=======================================================================

Calibrados por regressão linear contra cenários rotulados pela Rede
Bayesiana. Iguais para as 3 dimensões (domínio, ecossistema, linguagens).

Caso B (sit2 — exatamente 1 prioridade ativa):
  b        = -0.333333
  a_full   =  0.166667
  a_red2   =  1.166667

Caso C (sit1 — 2+ prioridades ativas):
  b        =  0.0
  w_covM   =  0.071429
  w_red2M  =  0.428571
  w_intMS  =  0.357143
  w_helpC  =  0.5

Pesos AT (wmean — espelham Rede Bayesiana):
  W_DOM = 3, W_ECO = 1, W_LING = 5

Pesos AE (mixminmax — calibrados contra Rede Bayesiana):
  W_MIN = 5  (efeito de gargalo — peso do atributo mais fraco)
  W_MAX = 1  (peso do atributo mais forte)

Parâmetros adicionais:
  η = 0.1         (floor da penalização Must no Caso C)
  μ_neutro = 0.5  (escore neutro para dimensão sem requisitos, Caso A)

Centróides AC (faixas VL/L/M/H/VH):
  c = [0.1, 0.3, 0.5, 0.7, 0.9]
  AC_LOW_SEM_COLABORACAO = 0.3  (valor para pares sem aresta no grafo)

=======================================================================
4. FORMATO DA BASE DE DESENVOLVEDORES (base_final.json)
=======================================================================

Lista JSON. Campos relevantes por dev:
  user_id      (str/int) — id único (normalizar: aceitar também "id")
  nomeCompleto (str)     — nome
  dominio      (list)    — ex: ["Web", "Cloud"]
  ecossistema  (list)    — ex: ["AWS", "React"]
  linguagens   (list)    — ex: ["Python", "Java"]

Normalização obrigatória:
  - Converter user_id para int
  - Converter todas as competências para lowercase para comparação
  - Filtrar devs sem nenhuma competência (não contribuem para AT)

=======================================================================
5. FORMATO DOS PROJETOS-ALVO (target_projects.json)
=======================================================================

{
  "projects": [
    {
      "id": "P1", "name": "...", "team_size": 4,
      "dominio":     {"must": [...], "should": [...], "could": [...]},
      "ecossistema": {"must": [...], "should": [...], "could": [...]},
      "linguagens":  {"must": [...], "should": [...], "could": [...]}
    },
    ...  (P1 a P12)
  ]
}

=======================================================================
6. FORMATO DO GRAFO DE COLABORAÇÃO (Graph_DB_real.json)
=======================================================================

{
  "nodes": [{"id": "Dev2", "user_id": 2, ...}, ...],
  "edges": [
    {"source": "Dev77", "source_user_id": 77,
     "target": "Dev78", "target_user_id": 78,
     "weight": 0.25, "N": 1},
    ...
  ]
}

O campo weight = f(N) já está calculado na fonte.
Pares sem colaboração (N=0) não têm aresta no grafo — recebem
AC_LOW_SEM_COLABORACAO = 0.3 no cálculo do AC.

=======================================================================
7. MODOS DE OPERAÇÃO
=======================================================================

O modelo suporta três modos, configurável no bench:

AT_ONLY:
  Maximiza apenas AT. AC e AE ignorados. Rápido (~1-15s por projeto).
  Usado para comparação técnica pura ILP vs GA.

AE_FULL_APROX:
  AC aproximado: média individual de cada dev no grafo.
    AC_aprox = (1/k) * Σ_i AC_medio_i * x_i
  Completamente linear, sem variáveis extras. Rápido (~10-15s).
  Enviesado para devs com histórico colaborativo amplo.
  Devs sem nenhuma aresta no grafo recebem AC_LOW_SEM_COLABORACAO.

AE_FULL_EXACT (modo principal do experimento):
  AC exato: média dos pesos PC_ij APENAS entre os pares da equipe
  selecionada. Para cada par (i,j) com aresta no grafo, cria variável
  binária p_ij com restrições McCormick:
    p_ij ≤ x_i,  p_ij ≤ x_j,  p_ij ≥ x_i + x_j - 1
  Pares da equipe SEM aresta no grafo recebem AC_LOW_SEM_COLABORACAO.
  Fórmula:
    AC = (1/C(k,2)) * [Σ w_ij * p_ij  +  0.3 * (C(k,2) - Σ p_ij)]
  Cria ~8.216 variáveis p_ij (só arestas existentes entre candidatos).
  Total do modelo: ~9.250 variáveis | ~25.730 restrições.
  Tempo: 590-900s por projeto (TEMPO_LIMITE=600 recomendado).
  Fiel ao cálculo do GA/surrogate.

=======================================================================
8. ARQUIVOS A GERAR: stfp_ilp.py e bench_stfp.py
=======================================================================

─────────────────────────────────────────────────────────────────
ARQUIVO 1: stfp_ilp.py
─────────────────────────────────────────────────────────────────

Funções a implementar:

  carregar_base(caminho, max_devs=None)
    → lista de devs normalizados (filtrados, lowercase, int id)

  carregar_projetos(caminho)
    → dict {id → projeto}

  carregar_grafo(caminho)
    → dict {(uid_i, uid_j): weight}  com uid_i < uid_j (par canônico)

  calcular_ac_medios(devs, grafo)
    → dict {user_id: float}
    Média dos pesos de todas as arestas do dev no grafo.
    Devs sem nenhuma aresta recebem AC_LOW_SEM_COLABORACAO (não 0.0).

  preparar_grafo_ilp(devs, grafo)
    → dict {(idx_i, idx_j): weight}  com idx_i < idx_j (índice na lista)
    Filtra o grafo mantendo só pares onde AMBOS os devs estão na lista
    de candidatos. Converte user_id → índice da lista.

  construir_modelo(devs, projeto, ac_medios, grafo_ilp, modo, k)
    → (modelo, x_vars, info)
    Constrói o MILP passo a passo com logs por parte (1 a 7).
    Implementar EXATAMENTE a formalização do PDF.

  resolver_ilp(projeto, devs, ac_medios, grafo_ilp, modo,
               tamanho_equipe, tempo_limite)
    → dict com: status, objetivo, AT, AC, AE, equipe, tempo_s,
                escores_dim, cobertura_must

  exibir_resultado(resultado, projeto)
    → imprime logs detalhados pós-solução

LOGS obrigatórios dentro de construir_modelo:
  [Parte 1] Variáveis x_i: N
  [Parte 2] Cobertura y,z: DOM(X) ECO(X) LING(X)
  [Parte 3] Features MoSCoW
  [Parte 4] DOM->Caso X | ECO->Caso X | LING->Caso X
  [Parte 5] AT = (3*DOM + 1*ECO + 5*LING) / 9
  [Parte 6] AC_exato | arestas no grafo: X | C(k,2)=X | variaveis p_ij: X | Low p/ pares sem colaboracao: 0.3
  [Parte 7] AE = (5*m + 1*M) / 6 | mixminmax
  [Modelo] X variaveis | X restricoes

IMPORTANTE:
  - Universo de competências: usar sorted(set(comp_proj | comp_devs))
    com deduplicação explícita para evitar colisão de nomes de constraints
  - Função _safe(s): remove espaços, /, -, (, ), , e . (troca por _)
    e trunca em 50 chars — evita nomes inválidos no PuLP
  - O arquivo NÃO tem bloco if __name__ == "__main__" funcional.
    Quem orquestra é o bench.

─────────────────────────────────────────────────────────────────
ARQUIVO 2: bench_stfp.py
─────────────────────────────────────────────────────────────────

CONSTANTES CONFIGURÁVEIS NO TOPO (única parte que muda entre experimentos):

  CAMINHO_BASE     = "base_final.json"
  CAMINHO_PROJETOS = "target_projects.json"
  CAMINHO_GRAFO    = "Graph_DB_real.json"
  CAMINHO_SAIDA    = "resultados_ilp.csv"

  MODO         = "AE_FULL_EXACT"   # "AT_ONLY" | "AE_FULL_APROX" | "AE_FULL_EXACT"
  PROJETOS_IDS = ["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12"]
  MAX_DEVS     = None               # None = todos
  TEMPO_LIMITE = 600                # segundos por projeto

O bench:
  1. Carrega base, projetos e grafo
  2. Calcula ac_medios para todos os devs (1x, antes do loop)
  3. Prepara grafo_ilp (1x, antes do loop)
  4. Para cada projeto em PROJETOS_IDS:
     - Chama resolver_ilp() do stfp_ilp.py
     - Exibe resultado detalhado
  5. Imprime resumo final tabular
  6. Salva CSV com colunas:
     projeto, modo, status, AT, AC, AE, equipe, tempo_s,
     must_dom_ok, must_eco_ok, must_ling_ok

=======================================================================
9. REQUISITOS GERAIS
=======================================================================

- Python 3.8+, dependência única: pip install pulp (solver CBC gratuito)
- Nomes de variáveis em português
- Todos os coeficientes embutidos como constantes no topo do stfp_ilp.py
- Tratar exceções: projeto sem devs suficientes, solver timeout, erros
- Solver: pulp.PULP_CBC_CMD(msg=0, timeLimit=tempo_limite)

=======================================================================
10. EXEMPLO DE SAÍDA ESPERADA DO BENCH (modo AE_FULL_EXACT)
=======================================================================

######################################################################
#  BENCH STFP-ILP  |  Modo: AE_FULL_EXACT  |  Projetos: [P1..P12]
######################################################################

[BENCH] 507 devs com pelo menos 1 competencia carregados
[BENCH] 12809 arestas carregadas
[BENCH] 8216 pares com aresta entre candidatos

============================================================
  PROJETO P1 — Manutenção de Portal Corporativo .NET
============================================================
[ILP] Construindo modelo AE_FULL_EXACT para P1 (k=4, N=507)
  [Parte 1] Variaveis x_i: 507
  [Parte 2] Cobertura y,z: DOM(13) ECO(212) LING(22)
  [Parte 3] Features MoSCoW
  [Parte 4] DOM->Caso C | ECO->Caso C | LING->Caso C
  [Parte 5] AT = (3*DOM + 1*ECO + 5*LING) / 9
  [Parte 6] AC_exato | arestas no grafo: 8216 | C(k,2)=6 | variaveis p_ij: 8216 | Low p/ pares sem colaboracao: 0.3
  [Parte 7] AE = (5*m + 1*M) / 6 | mixminmax
  [Modelo] 9246 variaveis | 25715 restricoes

[ILP] Solver: Optimal em 598.70s
[ILP] AT=0.9524  AC=0.7250  AE=0.7629

  Equipe selecionada:
    Dev  282 | Dev_282  DOM=['cloud','web'] ECO=['angular','stripe'] LING=[]
    Dev  449 | Dev_449  DOM=['artificial intelligence','desktop standalone',...] ...
    Dev  467 | Dev_467  DOM=['desktop standalone','hardware','iot'] ...
    Dev  540 | Dev_540  DOM=['desktop standalone','embedded systems','hardware'] ...

  Must cobertos:
    DOMI  v  desktop standalone
    DOMI  v  web
    ECOS  v  .net
    LING  v  c#

  Escores: DOMI=0.8571(CC) ECOS=1.0000(CC) LING=1.0000(CC)

======================================================================
  RESUMO FINAL
======================================================================
projeto  modo          status   AT      AC      AE      tempo_s
P1       AE_FULL_EXACT Optimal  0.9524  0.7250  0.7629   598.70s
P2       AE_FULL_EXACT Optimal  0.7143  0.5667  0.5913   601.70s
...
P12      AE_FULL_EXACT Optimal  0.9444  0.6250  0.6782   597.87s

[BENCH] CSV salvo em: resultados_ilp.csv

=======================================================================
11. EXTENSÕES FUTURAS (documentar no código, não implementar)
=======================================================================

- Quando houver dados OSF/SLF reais: substituir AC_aprox pela formulação
  exata da Seção 2.7 com pesos por faixa (VL/L/M/H/VH). Requer produto
  x_i * x_j linearizado via variáveis binárias q_ij para todos os pares.
- Rodar benchmark batch com múltiplos projetos e salvar comparativo GA vs ILP.

=======================================================================
12. NOTA DE IMPLEMENTAÇÃO
=======================================================================

Este código foi desenvolvido com suporte de IA generativa (Claude,
Anthropic, 2026) e verificado pelo co-autor Rian, especialista em
Programação Linear Inteira.

Para citação na tese e no artigo:
  "A implementação do modelo MILP foi desenvolvida com auxílio de
  ferramenta de IA generativa (Claude, Anthropic, 2026) e verificada
  pelo co-autor Rian, especialista em Programação Linear Inteira."
"""
