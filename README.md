# attention-factors-europe

Code for the MSc thesis: an out-of-sample test in European equities of the attention
factor model of Epstein, Wang, Choi & Pelger (2025).

`CLAUDE.md` holds the project decisions and the invariants. Read it first.
`docs/schemas.md` is the interface contract between the data layer and everything else.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"        # add ",data" on the machine that talks to WRDS
uv run pytest                     # should be green on a fresh clone
uv run nbstripout --install       # once per clone, keeps notebook output out of git
```

Copy `.env.example` to `.env` and fill in WRDS credentials. `.env` is gitignored and
must stay that way.

## Layout

```
src/afe/
  schemas.py         column and dtype contracts, plus validators
  runs.py            run directories: config, commit hash, seed, versions, metrics
  data/synthetic.py  fixture panel conforming to the schema, for downstream work
  data/              universe construction and feature builds (US and EU adapters)
  model/             attention factor model
  policy/            trading policy and the cost term
  evaluation/        performance, break-even cost curve, statistical tests
configs/             one YAML per specification
scripts/             entry points
tests/               invariants: no lookahead, universe, schema
```

## Day to day with git

Nobody here is a git veteran. The short version, all of which VS Code's Source Control
panel does with buttons:

1. `git pull` before you start. Always.
2. Make a branch for the task: `git switch -c universe-construction`.
3. Commit as you go. Small commits with a sentence saying what changed.
4. `git push -u origin universe-construction`, then open a pull request on GitHub.
5. The other person reads the diff and comments. Then merge, and delete the branch.

Rules that keep this painless:

- `main` always runs end to end. Nothing merges that has not been run once.
- Branches live days, not weeks. A stale branch against a moving schema is the one
  thing that will genuinely hurt.
- Stay inside your own part of the tree where you can. Two people editing one file is
  what makes conflicts.
- If a merge conflict appears, do not guess. Ask the other person whose version is right.

## Runs

Training runs happen on the shared cloud box. Two separate clones, one per person,
pointing at one read-only shared data directory. Never two people in one working tree.