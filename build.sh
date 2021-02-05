rm -rf dist/*
python setup.py sdist bdist_wheel
python -m pip install .