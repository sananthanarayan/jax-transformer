PYTHON ?= python
SWEEP_JSON := results/sweep.json
FIGURE := results/length_gen.png

.PHONY: help install smoke test train sweep sweep-phase1 quick plot figures paper clean

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
	@echo "  paper        Render paper/whitepaper.md -> paper/whitepaper.html (needs pandoc)"
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

paper:
	pandoc paper/whitepaper.md \
	  --standalone \
	  --metadata pagetitle="Length generalization, measured in the weights" \
	  --css=whitepaper.css \
	  --include-in-header=paper/header.html \
	  -o paper/whitepaper.html
	@echo "Wrote paper/whitepaper.html (open in browser; print-to-PDF if desired)"

clean:
	rm -f results/*.json results/*.png
