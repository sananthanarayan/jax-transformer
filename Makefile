PYTHON ?= python
SWEEP_JSON := results/sweep.json
FIGURE := results/length_gen.png

.PHONY: help install smoke test train sweep sweep-phase1 quick plot figures paper paper-html clean

help:
	@echo "Targets:"
	@echo "  install      Install Python dependencies into the active environment"
	@echo "  smoke        Run scripts/smoke_test.py (no training)"
	@echo "  test         Run tests/ unit tests"
	@echo "  train        Train the default addition model (single variant, CLI)"
	@echo "  sweep        Run the full length-generalization sweep (Phases 1+2, 6 variants)"
	@echo "  sweep-phase1 Run only the Phase 1 variants (baseline/reversed/nope)"
	@echo "  quick        Run a fast sanity sweep (2 epochs, fewer eval samples)"
	@echo "  plot         Rebuild figures from results/sweep.json"
	@echo "  figures      sweep + plot (the headline command)"
	@echo "  paper        Render paper/whitepaper.md -> paper/whitepaper.pdf (the headline artifact)"
	@echo "  paper-html   Render paper/whitepaper.md -> paper/whitepaper.html only"
	@echo "  clean        Delete results/*.json and results/*.png"

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) tests/test_digit_positions.py

sweep-phase1:
	$(PYTHON) scripts/run_sweep.py --phase 1

train:
	$(PYTHON) -m addition_transformer.train --op addition

$(SWEEP_JSON):
	$(PYTHON) scripts/run_sweep.py

sweep: $(SWEEP_JSON)

quick:
	$(PYTHON) scripts/run_sweep.py --epochs 2 --eval-samples 100

plot: $(SWEEP_JSON)
	$(PYTHON) scripts/plot.py

figures: sweep plot
	@echo "Wrote $(FIGURE)"

CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

paper-html:
	pandoc paper/whitepaper.md \
	  --standalone \
	  --metadata pagetitle="Length generalization, measured in the weights" \
	  --css=whitepaper.css \
	  --include-in-header=paper/header.html \
	  -o paper/whitepaper.html
	@echo "Wrote paper/whitepaper.html"

paper: paper-html
	@TMPDIR_CHROME=$$(mktemp -d) && \
	"$(CHROME)" \
	  --headless=new \
	  --disable-gpu \
	  --no-sandbox \
	  --allow-file-access-from-files \
	  --user-data-dir="$$TMPDIR_CHROME" \
	  --virtual-time-budget=15000 \
	  --run-all-compositor-stages-before-draw \
	  --print-to-pdf-no-header \
	  --print-to-pdf="$$PWD/paper/whitepaper.pdf" \
	  "file://$$PWD/paper/whitepaper.html" 2>&1 | tail -1 && \
	rm -rf "$$TMPDIR_CHROME"
	@echo "Wrote paper/whitepaper.pdf  (open paper/whitepaper.pdf to read)"

clean:
	rm -f results/*.json results/*.png
