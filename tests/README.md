# Automated Testing

## Running tests locally

Additional requirements for running tests:
```
pip install .[dev]
```
Run inference tests:
```
pytest tests/inference
```

## Quality regression test
Compares images in 2 directories to ensure they are the same

1) Run an inference test to save a directory of "ground truth" images
```
    pytest tests/inference --output_dir tests/inference/baseline
```
2) Make code edits

3) Run inference and quality comparison tests
```
pytest
```