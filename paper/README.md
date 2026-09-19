# AstroCLIMB system paper

This directory contains the ACL-formatted system-description paper for the solo team **Syed Mohaiminul Hoque**.

## Before submission

1. Confirm the title, author spelling, affiliation, and whether the workshop requires anonymous review. For anonymous review, change `\usepackage{acl}` to `\usepackage[review]{acl}` and remove identifying information as required by the current call.
2. Replace the provisional website citation for the shared task with the official overview-paper citation if the organizers release one.
3. Confirm the venue's mandatory sections and whether appendices are permitted in the submission package.
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

The target venue is the online **Fourth Workshop on Artificial Intelligence for Scientific Publications (WASP 2026)** on November 9, 2026. The submission window is June 29, 2026 at 8:59:07 PM through September 22, 2026 at 9:59:00 AM.

The paper has four content pages, one references page, and two appendix pages. It distinguishes final Kaggle leaderboard results from fixed-split validation results. The documented best system scores **0.73725 macro-F1** and ranks **third among 11 teams**. The associated code and artifacts are at <https://github.com/SyedT1/Astroclimb-sharedtask>.
