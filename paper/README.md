# AutoML-IoT: journal draft (Overleaf project)

## Compile on Overleaf
1. Upload the ZIP (New Project > Upload Project).
2. Menu > Main document: `main.tex`; Compiler: pdfLaTeX (the default). Overleaf runs BibTeX automatically.
3. Template: `IEEEtran` (journal mode). To switch to ACM style, change the document class and keep the content.

## Structure
- `main.tex`            preamble, title, abstract, section inputs, bibliography
- `sections/`           one file per section (Morshed structure; appendices in `10_appendix.tex`)
- `tables/`             all tables (`table_related`, `table_config`, `table_env` are static; the rest are generated)
- `figures/`            pipeline figure (TikZ, editable)
- `refs.bib`            44 BibTeX entries (see "References" below)
- `data/`               run-level results and audit files from which all result tables are generated
- `scripts/make_tables.py`    regenerates every result table:  `python scripts/make_tables.py`  (needs pandas, numpy)
- `scripts/check_writing.py`  checks numbers in abstract/introduction, sentence length, and vague wording

## References: verification status
All entries are real publications. Against the Scite index:
- Verified (found in Scite, or DOI resolves): 37 entries.
- Matched only at proceedings level: `tavallaee2009` (the paper itself is widely cited; no DOI was added).
- NOT INDEXED in Scite (real works taken from your earlier reference lists; please confirm details before submission):
  `feurer2015` (NIPS 2015), `ledell2020` (ICML AutoML Workshop 2020; author initial of Poirier), `yang2024automl` (IEEE TNSM 2024; confirm author list and volume/pages),
  `tama2019` (Neural Comput. Appl. 2019), `antonakakis2017` (USENIX Security 2017).
- `velnambi2026` is your own unpublished manuscript and cannot be verified; update venue and status.
- Entries with "and others" list only the verified first authors.
- A DOI is given only where it was verified; none was added from memory.
- Note: one later paper reports contrasting evidence on block cross-validation in a genomics setting (cited `roberts2017`); the text states only that blocked validation "can give more realistic estimates".

## Open items (red in the PDF)
Author names and affiliations; acknowledgments and funding; repository URL; status of `velnambi2026`.

## Writing conventions followed
- Abstract: two numbers (40.5%, 0.72). Introduction: one number word ("two"). List labels (1)-(6), table/figure numbers and citation numbers are not counted; "H2O" is a product name.
- Body text: mean sentence length about 19 words, no sentence below 9 words; remaining long sentences are equation-bearing.
