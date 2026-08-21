# tests

Unit tests for the VLSI SSTA framework.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Test Modules

| File | Description |
|------|-------------|
| `test_variation.py` | Tests for variation sampling and analytical moments |
| `test_timing.py` | Tests for timing graph and delay models |
| `test_analysis.py` | Tests for MC and analytical SSTA engines |
| `test_clark_max.py` | Tests for Clark MAX approximation |

All four modules are implemented: 11 tests, all passing (`python -m pytest tests/ -q`).

## Naming Convention

- `test_*.py` — Test modules
- Tests use `pytest` conventions
- Each test file covers one module from `variation/`, `timing/`, or `ssta/`
