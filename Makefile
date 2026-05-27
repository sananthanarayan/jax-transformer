PYTHON ?= python
SWEEP_JSON := results/sweep.json
FIGURE := results/length_gen.png

.PHONY: help install smoke train sweep quick plot figures clean

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies into the active environment"
	@echo "  smoke     Run scripts/smoke_test.py (no training)"
	@echo "  train     Train the default addition model (single variant, CLI)"
	@echo "  sweep     Run the full Phase 1 length-generalization sweep"
	@echo "  quick     Run a fast sanity sweep (2 epochs, fewer eval samples)"
	@echo "  plot      Rebuild the headline figure from results/sweep.json"
	@echo "  figures   sweep + plot (the headline command)"
	@echo "  clean     Delete results/*.json and results/*.png"

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) scripts/smoke_test.py

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

clean:
	rm -f results/*.json results/*.png
