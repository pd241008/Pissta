# Postmortem: numpy.erf AttributeError

> **Date:** August 15, 2026  
> **Severity:** Low — runtime error, fixed within minutes  
> **Status:** Fixed

## What Happened

`stage4b_hybrid_run.py` crashed with:
```
AttributeError: module 'numpy' has no attribute 'erf'
```

## Root Cause

NumPy 2.0+ moved `numpy.erf` to `scipy.special.erf` (or requires `from numpy.lib.scimath import erf` in some versions). The code used `np.erf()` directly, which worked in older NumPy but not in the installed version (Python 3.13, NumPy 2.x).

## Detection

Immediate crash on first run. No data corruption.

## Fix

Replaced `np.erf()` with `math.erf()` from the standard library. The `math` module has provided `erf` since Python 3.2 and has no external dependencies.

## Lessons Learned

- **Prefer stdlib over NumPy for special functions when possible.** `math.erf`, `math.exp`, `math.sqrt` are always available and often faster.
- **Pin NumPy major versions** or test against both NumPy 1.x and 2.x if the code is meant to be portable.
