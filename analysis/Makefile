# Render every plot into plots/ and show it inline in the terminal.
#
#   make icat     render what changed, then display
#   make plots    render only
#   make clean    drop the rendered plots
#
# One script per plot: plot_<name>.py writes plots/<name>.png, and is picked up here with no further wiring.

PYTHON ?= ../.venv/bin/python
# icat.py, not imgcat: imgcat speaks the iTerm2 protocol, which kitty renders as a wall of escape codes.
ICAT ?= $(shell command -v kitten >/dev/null 2>&1 && echo "kitten icat" || echo "$(PYTHON) icat.py")

SCRIPTS := $(wildcard plot_*.py)
IMAGES := $(patsubst plot_%.py,plots/%.png,$(SCRIPTS))

.PHONY: icat plots clean FORCE

plots: $(IMAGES)

# FORCE, not .PHONY: make skips pattern rules for phony targets, so marking the images phony would stop them
# being built at all. They must rerun every time regardless, since the data lives in wandb and the file
# timestamps know nothing about it.
plots/%.png: plot_%.py common.py FORCE
	$(PYTHON) $< $@

FORCE:

icat: plots
	@for f in $(IMAGES); do echo "== $$f"; $(ICAT) "$$f"; done

clean:
	rm -f plots/*.png
