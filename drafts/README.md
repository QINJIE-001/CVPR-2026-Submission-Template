# Drafts

- `paper_en.md`: English draft of the Introduction, Related Work and Method sections (source of `../sec/1_intro.tex`, `2_related.tex`, `3_method.tex`).
- `paper_zh.md`: literal Chinese translation of `paper_en.md`, paragraph by paragraph.

Workflow: edit `paper_en.md` and `paper_zh.md` together, then run `python3 tools/md2tex.py` from the repository root to regenerate `sec/0_abstract.tex`, `sec/1_intro.tex`, `sec/2_related.tex` and `sec/3_method.tex`. The method figure is `fig/overview.tex` (TikZ) and is placed by the `[[figure:overview]]` line of the draft.

Citation keys in both files are placeholders until `main.bib` is filled.
