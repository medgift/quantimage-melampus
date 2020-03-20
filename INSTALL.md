Generate distribution:
```
python3 setup.py sdist bdist_wheel
```
Upload:
```
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

