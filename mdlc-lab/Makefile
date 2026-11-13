.PHONY: install data notebooks test check

install:
	pip install -e ".[dev]"

data:            ## download Yelp (~200 MB) and build every lab dataset
	python -c "from mdlc.data import build_all; print(build_all())"

notebooks:       ## regenerate student + solution notebooks from notebooks/src
	python notebooks/build_notebooks.py

test:
	python -m pytest -q

check: data      ## run every solution notebook end to end
	for nb in solutions/*.ipynb; do \
	  jupyter nbconvert --to notebook --execute $$nb --output-dir /tmp/mdlc-check || exit 1; done
