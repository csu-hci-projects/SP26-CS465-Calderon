# LaTeX Source

This directory contains the ACM manuscript source for the AirDesk paper.

The paper uses the one-column ACM manuscript format:

```tex
\documentclass[manuscript]{acmart}
```

The local ACM files are included here so the source directory is self-contained:

- `acmart.cls`
- `ACM-Reference-Format.bst`
- `acmart.bib`

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If the system has `latexmk`:

```bash
latexmk -pdf main.tex
```

## Notes

- `main.tex` is the paper source.
- `references.bib` contains the bibliography entries used by the paper.
- The final rendered PDF is stored at the repository root as `Final_Report.pdf`.
