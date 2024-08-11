PROJECT_NAME = "666"

SOURCES := $(wildcard *.tex)

MAIN_DOCUMENT_BASE_NAME = main

PDFLATEX_CMD = pdflatex
BIBTEX_CMD = bibtex


.PHONY: pdf
pdf: $(MAIN_DOCUMENT_BASE_NAME).pdf

$(MAIN_DOCUMENT_BASE_NAME).pdf: $(SOURCES) $(BIBFILES)
	@echo "Compiling LaTeX using the pdf target..."
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN_DOCUMENT_BASE_NAME).tex
	@if grep -q "Rerun to get cross-references right" $(MAIN_DOCUMENT_BASE_NAME).log; then pdflatex -interaction=nonstopmode -halt-on-error $(MAIN_DOCUMENT_BASE_NAME).tex; fi

.PHONY: clean
clean:
	-rm -r *.aux *.toc *.log *.nav *.out *.snm -f
