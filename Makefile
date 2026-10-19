PY := .venv/bin/python

.PHONY: help venv data enrich run cli test lint fmt all

help:
	@echo "make venv    — создать .venv и установить зависимости"
	@echo "make data    — собрать POI из Overpass и построить data/pois.json"
	@echo "make enrich  — добавить описания из Wikipedia/Wikidata"
	@echo "make run     — запустить Streamlit-приложение"
	@echo "make cli     — пример запроса через CLI (demo-режим)"
	@echo "make test    — pytest (офлайн, без модели)"
	@echo "make lint    — ruff check"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m seruen.collect.overpass
	$(PY) -m seruen.collect.normalize

enrich:
	$(PY) -m seruen.collect.enrich_wiki

run:
	.venv/bin/streamlit run seruen/app/streamlit_app.py

cli:
	SERUEN_DEMO=1 $(PY) -m seruen.cli --preset mega --query "Я у Mega Silk Way, хочу погулять 45 минут"

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

fmt:
	.venv/bin/ruff check --fix .

all: data test lint
