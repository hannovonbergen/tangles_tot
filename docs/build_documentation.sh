#!/bin/bash

black .
jupyter nbconvert examples/*.ipynb --to markdown --output-dir docs/converted_notebooks --execute
pdoc --math -d google tangles_tot examples -o docs/