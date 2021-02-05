./build.sh

if [ "$1" == "test" ]; then
    python -m twine upload --repository testpypi dist/*
else
    python -m twine upload dist/*
fi
