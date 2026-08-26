.PHONY: install test build clean

install:
	python -m pip install -e .

test:
	python -m pytest

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
