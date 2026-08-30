# AGentBOOK

Manuel d'ingénierie LangChain & LangGraph — systèmes agentiques en production.

Le contenu complet se trouve dans [`AGentBOOK.md`](./AGentBOOK.md).

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
pandoc AGentBOOK.md \
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
md-to-pdf AGentBOOK.md
```

> Note : `md-to-pdf` rend nativement les blocs Mermaid via Chromium.

### Option 3 — VS Code

Installer l'extension **Markdown PDF** (yzane), ouvrir `AGentBOOK.md`, puis `Ctrl+Shift+P` → « Markdown PDF: Export (pdf) ». Activer `markdown-pdf.mermaidServer` dans les réglages pour le rendu des diagrammes Mermaid.