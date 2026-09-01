# LangChain & LangGraph — Construire des systèmes agentiques en production

> Manuel d’ingénierie, avec du code, des architectures, des exercices et un projet fil rouge.

> ℹ️ Le contenu du livre est découpé par partie dans le dossier [`book/`](./book/) afin que GitHub puisse afficher le rendu riche (texte et diagrammes Mermaid) de chaque fichier. Utilise le sommaire ci-dessous pour naviguer.

## Sommaire

- [Préface](#préface)
- [Partie I — Comprendre les applications LLM](book/01a-partie-i.md#partie-i--comprendre-les-applications-llm)
  - [Chapitre 1 — Des LLM aux systèmes agentiques](book/01a-partie-i.md#chapitre-1--des-llm-aux-systèmes-agentiques)
    - [1.1 Les limites d'un appel LLM classique](book/01a-partie-i.md#11-les-limites-dun-appel-llm-classique)
    - [1.2 Le modèle comme moteur de raisonnement](book/01a-partie-i.md#12-le-modèle-comme-moteur-de-raisonnement)
    - [1.3 Le contexte](book/01a-partie-i.md#13-le-contexte)
    - [1.4 Les messages](book/01a-partie-i.md#14-les-messages)
    - [1.5 Les entrées et sorties structurées](book/01b-partie-i-suite.md#15-les-entrées-et-sorties-structurées)
    - [1.6 Pourquoi les LLM ont besoin d'outils](book/01b-partie-i-suite.md#16-pourquoi-les-llm-ont-besoin-doutils)
    - [1.7 Workflow déterministe vs agent](book/01b-partie-i-suite.md#17-workflow-déterministe-vs-agent)
    - [1.8 Le concept de boucle agentique](book/01b-partie-i-suite.md#18-le-concept-de-boucle-agentique)
    - [1.9 Agentic AI : définition et limites](book/01b-partie-i-suite.md#19-agentic-ai--définition-et-limites)
- [Partie II — Fondamentaux de LangChain](book/02-partie-ii.md#partie-ii--fondamentaux-de-langchain)
  - [Chapitre 2 — Architecture de LangChain](book/02-partie-ii.md#chapitre-2--architecture-de-langchain)
- [Partie III — Models, Messages et Prompts](book/03-partie-iii.md#partie-iii--models-messages-et-prompts)
  - [Chapitre 3 — Interagir avec les modèles](book/03-partie-iii.md#chapitre-3--interagir-avec-les-modèles)
  - [Chapitre 4 — Prompt Engineering avec LangChain](book/03-partie-iii.md#chapitre-4--prompt-engineering-avec-langchain)
- [Partie IV — Structured Output](book/04-partie-iv.md#partie-iv--structured-output)
  - [Chapitre 5 — Faire produire des données fiables au LLM](book/04-partie-iv.md#chapitre-5--faire-produire-des-données-fiables-au-llm)
- [Partie V — Tools et Tool Calling](book/05-partie-v.md#partie-v--tools-et-tool-calling)
  - [Chapitre 6 — Comprendre les Tools](book/05-partie-v.md#chapitre-6--comprendre-les-tools)
  - [Chapitre 7 — Tool Calling](book/05-partie-v.md#chapitre-7--tool-calling)
- [Partie VI — Construire des Agents](book/06-partie-vi.md#partie-vi--construire-des-agents)
  - [Chapitre 8 — Premier agent](book/06-partie-vi.md#chapitre-8--premier-agent)
  - [Chapitre 9 — Concevoir un agent robuste](book/06-partie-vi.md#chapitre-9--concevoir-un-agent-robuste)
- [Partie VII — RAG avec LangChain](book/07-partie-vii.md#partie-vii--rag-avec-langchain)
  - [Chapitre 10 — Comprendre le RAG](book/07-partie-vii.md#chapitre-10--comprendre-le-rag)
  - [Chapitre 11 — Construire un RAG complet](book/07-partie-vii.md#chapitre-11--construire-un-rag-complet)
  - [Chapitre 12 — RAG avancé](book/07-partie-vii.md#chapitre-12--rag-avancé)
- [Partie VIII — LangGraph](book/08-partie-viii.md#partie-viii--langgraph)
  - [Chapitre 13 — Pourquoi LangGraph ?](book/08-partie-viii.md#chapitre-13--pourquoi-langgraph-)
  - [Chapitre 14 — Les fondamentaux de LangGraph](book/08-partie-viii.md#chapitre-14--les-fondamentaux-de-langgraph)
- [Partie IX — Construire des Agents avec LangGraph](book/09-partie-ix.md#partie-ix--construire-des-agents-avec-langgraph)
  - [Chapitre 15 — Agent LangGraph](book/09-partie-ix.md#chapitre-15--agent-langgraph)
- [Partie X — State, Mémoire et Persistence](book/10-partie-x.md#partie-x--state-mémoire-et-persistence)
  - [Chapitre 16 — Concevoir le State](book/10-partie-x.md#chapitre-16--concevoir-le-state)
  - [Chapitre 17 — Persistence et Checkpoints](book/10-partie-x.md#chapitre-17--persistence-et-checkpoints)
- [Partie XI — Human-in-the-Loop](book/11-partie-xi.md#partie-xi--human-in-the-loop)
  - [Chapitre 18 — Ajouter un humain dans la boucle](book/11-partie-xi.md#chapitre-18--ajouter-un-humain-dans-la-boucle)
- [Partie XII — Agents spécialisés et architectures complexes](book/12-partie-xii.md#partie-xii--agents-spécialisés-et-architectures-complexes)
  - [Chapitre 19 — Routing et orchestration](book/12-partie-xii.md#chapitre-19--routing-et-orchestration)
  - [Chapitre 20 — Multi-Agent Systems](book/12-partie-xii.md#chapitre-20--multi-agent-systems)
- [Partie XIII — Agents multimodaux](book/13-partie-xiii.md#partie-xiii--agents-multimodaux)
  - [Chapitre 21 — Vision, audio et données structurées](book/13-partie-xiii.md#chapitre-21--vision-audio-et-données-structurées)
- [Partie XIV — Observabilité et Evaluation](book/14-partie-xiv.md#partie-xiv--observabilité-et-evaluation)
  - [Chapitre 22 — Observer un système agentique](book/14-partie-xiv.md#chapitre-22--observer-un-système-agentique)
  - [Chapitre 23 — Évaluer un agent](book/14-partie-xiv.md#chapitre-23--évaluer-un-agent)
- [Partie XV — Production](book/15-partie-xv.md#partie-xv--production)
  - [Chapitre 24 — Transformer un prototype en service](book/15-partie-xv.md#chapitre-24--transformer-un-prototype-en-service)
  - [Chapitre 25 — Performance et coûts](book/15-partie-xv.md#chapitre-25--performance-et-coûts)
  - [Chapitre 26 — Sécurité des agents](book/15-partie-xv.md#chapitre-26--sécurité-des-agents)
- [Partie XVI — Projet final](book/16-partie-xvi.md#partie-xvi--projet-final)
  - [Chapitre 27 — Architecture du projet](book/16-partie-xvi.md#chapitre-27--architecture-du-projet)
- [Partie XVII — Projet avancé : Agent autonome multimodal](book/17-partie-xvii.md#partie-xvii--projet-avancé--agent-autonome-multimodal)
  - [Chapitre 28 — Construire un agent de Spatial Intelligence](book/17-partie-xvii.md#chapitre-28--construire-un-agent-de-spatial-intelligence)
- [Annexes](book/18-annexes.md#annexes)
  - [Annexe A — Référence Python](book/18-annexes.md#annexe-a--référence-python)
  - [Annexe B — Référence LangChain](book/18-annexes.md#annexe-b--référence-langchain)
  - [Annexe C — Référence LangGraph](book/18-annexes.md#annexe-c--référence-langgraph)
  - [Annexe D — Patterns agentiques](book/18-annexes.md#annexe-d--patterns-agentiques)
  - [Annexe E — Checklist production](book/18-annexes.md#annexe-e--checklist-production)
  - [Annexe F — Glossaire](book/18-annexes.md#annexe-f--glossaire)
  - [Annexe G — Références](book/18-annexes.md#annexe-g--références)

## Préface

**LangChain**, **LangGraph** et les systèmes agentiques sont passés en peu de temps du stade d’expérimentation à celui de brique logicielle stratégique. Ce livre a été conçu comme un manuel d’ingénierie : tu y trouveras du code, des architectures, des exercices et surtout un projet fil rouge orienté production.

L’objectif n’est pas seulement de t’apprendre à “parler” à un modèle, mais de t’aider à construire des applications fiables, traçables et contrôlables. Tu verras pourquoi une simple interaction avec un **LLM** ne suffit plus dès qu’on doit intégrer des outils, du contexte métier, du **RAG**, des contraintes de sécurité ou une boucle de décision.

Le livre s’adresse en priorité aux :

- développeurs Python ;
- **Machine Learning Engineers** ;
- **AI Engineers** ;
- data scientists ;
- développeurs qui souhaitent construire des agents IA ;
- ingénieurs qui veulent intégrer des LLM dans des applications existantes.

Prérequis recommandés :

- **Python** ;
- APIs REST ;
- **JSON** ;
- notions de programmation orientée objet ;
- notions de machine learning recommandées ;
- notions de LLM utiles mais non indispensables.

Le principe directeur du livre est simple : **ne pas utiliser un agent parce qu’un agent est possible**. Utilise toujours l’architecture la plus simple capable de résoudre le problème de manière fiable — qu’il s’agisse d’une LLM application, d’un workflow, d’un système RAG ou d’un agent plus autonome.

