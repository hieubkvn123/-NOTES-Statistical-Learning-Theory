# NOTE-template
---
Latex template for note-taking

# 1. Install MikTex + Perl (for Windows)
---
- Install MikTex from [Link (MikTex)](https://miktex.org/download).
- Install Perl from [Link (Perl)](https://strawberryperl.com/).

# 2. Install Texlive (for Ubuntu)
---
```bash
sudo apt install texlive-science texlive-latex-extra texlive-extra-utils latexmk texlive-publishers texlive-science
```

# 3. Setting up VS-Code
---

## 3.1. Setting up output directory
- 1. Install Latex workshop plug-in.
- 2. Press `Ctrl` + `Shift` + `P` and search for `Open User Settings (UI)`.
- 3. Search for `latex OutDir`.
- 4. Key in `%DIR%/output` so that output is stored in `./output`.
- 5. Search for `Auto Clean`.
- 6. Set to `onBuilt`. 

## 3.2. Plugins
Install the following plugins:
- 1. `Latex Refs`.
- 2. `Latex Workshop`.
