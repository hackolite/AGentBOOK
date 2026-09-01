# AGentBOOK

Manuel d'ingénierie LangChain & LangGraph — systèmes agentiques en production.

Le sommaire et la préface se trouvent dans [`AGentBOOK.md`](./AGentBOOK.md). Le contenu est découpé par partie dans le dossier [`book/`](./book/) : GitHub refuse d'afficher le rendu riche (« Unable to render rich display ») des fichiers Markdown trop volumineux, le livre est donc réparti en fichiers plus petits pour que le texte et les diagrammes Mermaid s'affichent correctement.

## Générer un PDF

### Option 1 — Pandoc (recommandé)

Installer les dépendances :

```bash
# Debian / Ubuntu
sudo apt-get install pandoc texlive-xetex

# macOS
brew install pandoc
brew install --cask mactex-no-gui
```

Le livre contient des diagrammes Mermaid : installer le filtre qui les convertit en images lors de l'export :

```bash
npm install -g mermaid-filter
```

Générer le PDF :

```bash
pandoc AGentBOOK.md book/*.md \
  -o AGentBOOK.pdf \
  --pdf-engine=xelatex \
  -F mermaid-filter \
  --toc \
  --toc-depth=3 \
  -V geometry:margin=2.5cm \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  --highlight-style=tango
```

### Option 2 — md-to-pdf (Node.js, rendu HTML/CSS)

```bash
npm install -g md-to-pdf
cat AGentBOOK.md book/*.md > AGentBOOK-complet.md
md-to-pdf AGentBOOK-complet.md
```

> Note : `md-to-pdf` rend nativement les blocs Mermaid via Chromium.

### Option 3 — VS Code

Installer l'extension **Markdown PDF** (yzane), concaténer le livre (`cat AGentBOOK.md book/*.md > AGentBOOK-complet.md`), ouvrir `AGentBOOK-complet.md`, puis `Ctrl+Shift+P` → « Markdown PDF: Export (pdf) ». Activer `markdown-pdf.mermaidServer` dans les réglages pour le rendu des diagrammes Mermaid.