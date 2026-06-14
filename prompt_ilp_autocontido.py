"""
PROMPT — Implementação ILP para STFP-T
================================================

ARQUIVOS QUE DEVEM SER ANEXADOS (exatamente estes 3):
  1. formalizacao_STFP_linearizado.pdf    — formalização matemática linearizada
  2. base_final_com_rdas.json             — base de desenvolvedores
  3. target_projects.json                 — projetos-alvo P1-P6

Tudo mais está contido neste prompt.
================================================
"""

PROMPT = """
Você é um assistente especializado em Pesquisa Operacional e otimização combinatória.
Implemente um modelo de Programação Linear Inteira (ILP/MILP) em Python usando PuLP
para resolver o Problema de Formação de Equipes de Software — variante STFP-T.

=======================================================================
1. CONTEXTO
=======================================================================

O STFP-T é a variante técnica do Software Team Formation Problem. Dado
um conjunto de desenvolvedores e um projeto-alvo com requisitos MoSCoW
(Must/Should/Could) em três dimensões (domínio, ecossistema, linguagens),
o objetivo é selecionar uma equipe de tamanho k que maximize a Aptidão
Técnica (AT). Este modelo é usado como baseline de comparação com um
Algoritmo Genético (GA).

=======================================================================
2. FORMALIZAÇÃO MATEMÁTICA
=======================================================================

A formalização completa está no PDF anexado
(formalizacao_STFP_linearizado.pdf). Leia-o integralmente antes de
implementar. Ele contém:

  - Seção 2.2: variáveis de decisão x_i, y_u^δ, z_u^δ e suas restrições
  - Seção 2.3: contagem e cobertura de competências (h_u^δ, y, z)
  - Seção 2.4: features de cobertura MoSCoW (full, red2)
  - Seção 2.5: escore por dimensão — Caso A (neutro), Caso B (sit2,
               clamp linearizado), Caso C (sit1, McCormick exato,
               produto triplo, ρ = max(η,·), clamp + produto com ρ)
  - Seção 2.6: aptidão técnica AT (wmean 3/1/5)
  - Seção 2.9: modelo MILP completo consolidado

Implemente EXATAMENTE a formalização do PDF, incluindo todas as
linearizações descritas nas subseções C.1 a C.4.

=======================================================================
3. COEFICIENTES CALIBRADOS (embutidos aqui — não precisam de arquivo)
=======================================================================

Os coeficientes abaixo foram calibrados por regressão linear contra
cenários rotulados pela Rede Bayesiana, modelo probabilistico criado para representar o conhecimento de um especialista em formação de equipes de Software. São iguais para as 3 dimensões.

Caso B (sit2):
  b        = -0.333333
  a_full   =  0.166667
  a_red2   =  1.166667

Caso C (sit1):
  b        =  0.0
  w_covM   =  0.071429
  w_red2M  =  0.428571
  w_intMS  =  0.357143
  w_helpC  =  0.5

Pesos do AT (wmean — espelham a Rede Bayesiana):
  W_DOM = 3, W_ECO = 1, W_LING = 5

Parâmetros adicionais:
  η (eta, floor da penalização Must) = 0.1
  μ_neutro (dimensão sem requisitos)  = 0.5

=======================================================================
4. FORMATO DA BASE DE DESENVOLVEDORES (base_final_com_rdas.json — arquivo anexado)
=======================================================================

Lista JSON de objetos com os campos relevantes:
  user_id      (int)   — identificador único
  nomeCompleto (str)   — nome completo
  dominio      (list)  — competências de domínio ex: ["Web", "Cloud"]
  ecossistema  (list)  — competências de ecossistema ex: ["AWS", "React"]
  linguagens   (list)  — competências de linguagens ex: ["Python", "Java"]

Normalização necessária:
  - Aceitar tanto "id" (base antiga) quanto "user_id" (base nova)
  - Converter todas as competências para lowercase para comparação

=======================================================================
5. FORMATO DOS PROJETOS-ALVO (target_projects.json — arquivo anexado)
=======================================================================

{
  "projects": [
    {
      "id": "P1",
      "name": "...",
      "team_size": 4,
      "dominio":     {"must": [...], "should": [...], "could": [...]},
      "ecossistema": {"must": [...], "should": [...], "could": [...]},
      "linguagens":  {"must": [...], "should": [...], "could": [...]}
    },
    ...  (P1 a P6)
  ]
}

=======================================================================
6. REQUISITOS DE IMPLEMENTAÇÃO
=======================================================================

Arquivo: stfp_ilp.py
Dependência única: pip install pulp
Python: 3.8+

CONSTANTES CONFIGURÁVEIS NO TOPO DO ARQUIVO (sem tocar no resto):
  CAMINHO_BASE        = "base_final_com_rdas.json"
  CAMINHO_PROJETOS    = "target_projects.json"
  ID_PROJETO          = "P1"          # qual projeto rodar
  TAMANHO_EQUIPE      = 4
  MAX_DEVS            = None          # None = todos; número = limitar p/ teste
  TEMPO_LIMITE_SOLVER = 60            # segundos

LOGS DETALHADOS (obrigatório):
  - Devs carregados com suas competências
  - Projeto-alvo selecionado e seus requisitos
  - Construção do modelo passo a passo (Parte 1 a 5)
  - Qual caso foi ativado por dimensão (A, B ou C)
  - Equipe selecionada: id, nome, competências
  - Devs descartados: id, nome
  - Cobertura dos requisitos Must (✓ ou ✗)
  - Escores por dimensão
  - Status e valor AT ótimo do solver

NOMES DE VARIÁVEIS EM PORTUGUÊS para facilitar leitura.

ESTRUTURA DO CÓDIGO (use funções separadas):
  carregar_base(caminho, max_devs)        → lista de devs normalizados
  carregar_projetos(caminho)              → dict de projetos
  construir_modelo_ilp(devs, projeto, k)  → (modelo, x, escores_dim)
  exibir_resultado(modelo, x, ...)        → logs pós-solução
  main()                                  → orquestra tudo

PESOS CALIBRADOS: embutidos no código como constantes (não carregar de arquivo).

=======================================================================
7. EXEMPLO MÍNIMO PARA VALIDAÇÃO RÁPIDA
=======================================================================

Para testar antes de rodar o projeto completo, configure:
  MAX_DEVS   = 10
  ID_PROJETO = "P4"  (projeto com só 1 must por dimensão — mais simples)

Resultado esperado:
  - Status: Optimal
  - Equipe com exatamente 4 devs
  - Todos os requisitos Must cobertos (✓)
  - AT entre 0 e 1


=======================================================================
9. EXTENSÕES FUTURAS (não implementar agora, só documentar no código)
=======================================================================

- Rodar todos os 6 projetos em sequência e salvar CSV comparativo com GA
- Incluir AC e AE (STFP completo) com variáveis m, M, b para min/max:
    m ≤ AT, m ≤ AC
    M ≥ AT, M ≥ AC, M ≤ AT + (1-b), M ≤ AC + b
    AE = (5*m + 1*M) / 6
  Requer dados de compatibilidade PC_ij entre pares de devs.
"""

print(PROMPT)
