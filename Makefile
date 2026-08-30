PYTHON ?= python3
APP_ENV ?= .modeldebugger-app
APP_PYTHON := $(APP_ENV)/bin/python
APP_READY := $(APP_ENV)/.dependencies-ready
LEGACY_WORKER_ENV := .refusalscope-worker
WORKER_ENV ?= $(if $(wildcard $(LEGACY_WORKER_ENV)/.dependencies-ready),$(LEGACY_WORKER_ENV),.modeldebugger-worker)
WORKER_PYTHON := $(WORKER_ENV)/bin/python
WORKER_READY := $(WORKER_ENV)/.dependencies-ready

.PHONY: all build run app-setup worker worker-setup test clean

all: build

build:
	$(PYTHON) -m compileall -q refusalscope workers tests

$(APP_READY): pyproject.toml
	$(PYTHON) -m venv $(APP_ENV)
	$(APP_PYTHON) -m pip install --upgrade pip
	$(APP_PYTHON) -m pip install -e .
	touch $(APP_READY)

app-setup: $(APP_READY)

run: build app-setup
	$(APP_PYTHON) -m refusalscope

$(WORKER_READY):
	$(PYTHON) -m venv $(WORKER_ENV)
	$(WORKER_PYTHON) -m pip install --upgrade pip
	$(WORKER_PYTHON) -m pip install torch transformers accelerate safetensors
	touch $(WORKER_READY)

worker-setup: $(WORKER_READY)

worker: worker-setup
	$(WORKER_PYTHON) workers/modeldebugger_worker.py --port 8765 --session-file .modeldebugger/local-worker.json

test: build
	node --test src/benchmark.test.js src/debugger.test.js src/graph-routing.test.js src/dom-contract.test.js
	$(PYTHON) -m unittest discover -s tests -v

clean:
	find refusalscope workers tests -type d -name __pycache__ -prune -exec rm -rf {} +
