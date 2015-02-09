PYTHON ?= python

all: fetch train leaderboard

setup:
	$(PYTHON) scripts/setup_databoard.py

fetch:
	$(PYTHON) scripts/repos2databoard.py

train:
	$(PYTHON) scripts/train_model.py

server:
	$(PYTHON) server.py

leaderboard:
	$(PYTHON) scripts/make_leaderboards.py

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-ctags:
	rm -f tags
