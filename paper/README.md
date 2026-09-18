# AstroCLIMB system paper

This directory contains the ACL-formatted system-description paper for the solo team **Syed Mohaiminul Hoque**.

## Before submission

1. Confirm the title, author spelling, affiliation, and whether the workshop requires anonymous review. For anonymous review, change `\usepackage{acl}` to `\usepackage[review]{acl}` and remove identifying information as required by the current call.
2. Replace the provisional website citation for the shared task with the official overview-paper citation if the organizers release one.
3. Confirm the venue's current page limit and mandatory sections before uploading.
4. Compile and inspect the PDF for overfull boxes, table placement, and page count.

## Compile

The official ACL style and bibliography style are included locally. Run:

```bash
latexmk -pdf paper.tex
```

or:

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

The source intentionally distinguishes hidden-test Kaggle scores from fixed-split validation results. The reported final standing is third place among 11 teams.
