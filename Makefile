.PHONY: install test demo serve clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

demo:
	python examples/run_demo.py

serve:
	mlgates serve --host 127.0.0.1 --port 8000

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist
