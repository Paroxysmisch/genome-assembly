#!/bin/bash

if [ -d ".venv" ]; then
	echo "Virtual environment already exists."
else
	echo "Creating new virtual environment."
	python3 -m venv ".venv"
fi

. ".venv/bin/activate"

pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install -r requirements
