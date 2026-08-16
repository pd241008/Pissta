# ADR-004: Monorepo Structure per Design Dungeons Standards

> **Status:** Decided  
> **Date:** August 2026

## Context

We need a repository structure that separates concerns, enables independent module development, and follows documented engineering standards.

## Options Considered

1. **Flat structure** — All files at root. Simple but unmaintainable as the project grows.
2. **Multiple repos** — One repo per module. Clean separation but hard to keep versions in sync.
3. **Numbered monorepo** — Follow Design Dungeons convention: `foundations/`, `variation/`, etc.

## Decision

We use a **numbered monorepo structure** following the Design Dungeons playbook.

## Reasoning

- Numbered directories enforce a logical dependency order (foundations → variation → timing → analysis → experiments).
- Each module has its own README documenting interfaces and dependencies.
- Root-level README provides navigation without requiring deep knowledge of internals.
- ADRs and postmortems live in `docs/` for traceability.

## Consequences

- All imports must use the new package paths (e.g., `from variation.sampler import ...`).
- Config is centralized in `foundations/stage3_config.json`.
- Results are isolated in `results/` with stage-specific files.
