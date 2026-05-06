# LaTeX Source

This directory contains the ACM manuscript source for the AirDesk paper.

The teacher's screenshot says to use:

```tex
\documentclass[manuscript]{acmart}
```

That is what `main.tex` uses. The local ACM files copied from `/home/caden/Downloads/acmart-primary.zip` are included here so the final submission source directory is self-contained:

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

## TODO Before Final

- Replace placeholder email and participant/acknowledgment fields.
- Decide whether the roommate should be an author, participant, or acknowledgment.
- Add final pilot results.
- Replace placeholder related-work prose with citations after the PDFs are collected.
- Add figures for the pipeline, gesture vocabulary, and evaluation artifacts.

