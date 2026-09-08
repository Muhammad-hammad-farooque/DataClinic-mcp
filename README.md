# DataClinic MCP

An MCP server that lets an AI assistant run a full exploratory data analysis
workflow — inspect a dataset, diagnose what is wrong with it, fix it, and write
the result back out.

> **Status: early alpha.** Phase 1 of 6 is complete. Two tools work today
> (`load_dataset`, `manage_sources`). The analysis, cleaning and database tools
> described in the roadmap are **not implemented yet**. See
> [Current state](#current-state) for exactly what runs.

---

## Why another EDA server

Most data tools built for AI assistants return *data*. `df.describe()` on fifty
columns, a thirty-by-thirty correlation matrix, a hundred raw rows. The
assistant then has to work out what any of it means, and the numbers stay in
the conversation being re-sent on every later turn.

DataClinic returns *findings*:

```
10 columns, 5,150 rows. 4 column(s) need attention before analysis.

HIGH notes: 100% missing -> add a missingness flag rather than imputing (5,150 rows)
HIGH notes: entirely empty -> drop
MED  country: 27% missing -> decide between imputation and a flag (1,412 rows)
MED  age: 25% missing -> decide between imputation and a flag (1,277 rows)
MED  3% of rows are exact duplicates -> drop_duplicates unless repetition is meaningful
MED  region_code: single value throughout -> drop; it carries no signal
```

Every response carries a token budget and stops when it reaches it, saying what
it left out.

### Statistics cover every row

Some tools profile a file by reading its first hundred rows. On a sorted file
that produces answers that are simply wrong, without saying so. Measured on this
project's own test fixture:

| | mean of `price` |
|---|---|
| first 100 rows | 2.81 |
| full population | 36.04 |
| **error** | **92%** |

DataClinic reads the whole file. A test asserts that a sorted and an unsorted
copy of the same data profile identically.

---

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Muhammad-hammad-farooque/DataClinic-mcp.git
cd DataClinic-mcp
uv sync
```

Optional format support:

```bash
uv sync --extra excel      # .xlsx, .xls
uv sync --extra parquet    # .parquet
```

## Connect it

**Claude Code**

```bash
claude mcp add dataclinic -- uv --directory /path/to/DataClinic-mcp run dataclinic-mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dataclinic": {
      "command": "uv",
      "args": ["--directory", "/path/to/DataClinic-mcp", "run", "dataclinic-mcp"]
    }
  }
}
```

Restart the client and the tools appear.

---

## Current state

### Working

| Tool | What it does |
|---|---|
| `load_dataset` | Reads CSV, TSV, Excel, Parquet, JSON or NDJSON into the session and returns shape, column classifications, missing-data summary, duplicate count and ranked findings |
| `manage_sources` | Lists what is open in the session, or closes one to free memory |

`load_dataset` returns a profile in its first response, so there is no need for
a follow-up call to describe what was just loaded.

**On the way in, it handles:**

- encoding detection (UTF-8, UTF-8-BOM, cp1252, latin-1)
- delimiter sniffing (`,` `;` tab `|`)
- nulls disguised as `N/A`, `-`, `?`, `unknown`, `#N/A` and similar
- numbers stored as text, including thousands separators (`"20,281.33"`)
- dates stored as text, including mixed formats

Every conversion is reported in a `read.coerced` field rather than done
silently.

**Column classification:** numeric, categorical, datetime, boolean, text,
identifier, constant, empty. Identifiers are detected by a consecutive-run
signature rather than uniqueness alone, so a column of unique prices is not
mistaken for a key.

### Not built yet

`profile` · `analyze_column` · `find_issues` · `check_relationships` ·
`analyze_target` · `query` · `validate_rules` · `clean_data` ·
`transform_data` · `reshape_data` · `history` · `plot` · `generate` · `export`
· database connectivity

These are specified in [`speckit.md`](speckit.md) but not implemented. The
roadmap below gives the order.

---

## Example

```
> Load tests/fixtures/messy.csv and tell me what is wrong with it
```

```json
{
  "shape": [5150, 10],
  "memory_mb": 1.72,
  "columns": {
    "numeric": ["age", "churn_score", "churned", "price", "revenue"],
    "categorical": ["country"],
    "datetime": ["signup_date"],
    "text": ["customer_id"],
    "constant": ["region_code"],
    "empty": ["notes"]
  },
  "missing_cells_pct": 16,
  "duplicate_rows": 150,
  "read": {
    "format": "csv",
    "encoding": "utf-8",
    "delimiter": ",",
    "coerced": {
      "revenue": "numeric (thousands separators removed)",
      "signup_date": "datetime"
    }
  },
  "summary": "10 columns, 5,150 rows. 4 column(s) need attention before analysis.",
  "findings": ["..."]
}
```

---

## Design

Three ideas drive the implementation. The full reasoning is in
[`speckit.md`](speckit.md).

**Findings, not data.** Each tool has a token budget (800 for `load_dataset`,
1,500 for a full profile). Findings are ranked by severity and emitted until the
budget is reached, then `truncated` reports what was omitted and how to get it.
Floats are rounded to three significant figures; absent fields are dropped
entirely.

**Correct beats fast.** Statistics are computed over the whole dataset. When
sampling eventually becomes necessary for large database tables, the response
will carry a `sampled` field stating the size and method — never silently.

**Nothing is modified in place.** Data is loaded into an in-memory session. The
source file is never written to; a test asserts it is byte-identical after a
full session. Cleaning operations, when they arrive, will mutate only the
session and be reversible.

**Errors are structured.** Every failure returns a stable code, a message, the
remedy to try next, and whether a retry could help:

```json
{
  "error": {
    "code": "SOURCE_TOO_LARGE",
    "class": "user",
    "message": "orders has ~41,238,904 rows, above max_load_rows (5,000,000)",
    "retryable": false,
    "remedy": "use profile() for push-down analysis, or pass limit="
  }
}
```

---

## Configuration

Environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `EDA_MCP_MAX_LOAD_ROWS` | 5,000,000 | Refuse to load more rows than this |
| `EDA_MCP_MAX_MEMORY_MB` | 4,096 | Memory ceiling |
| `EDA_MCP_ALLOWED_PATHS` | working directory | Roots the server may read and write |
| `EDA_MCP_SEED` | 42 | Seed for any sampling, for reproducibility |
| `EDA_MCP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Paths are resolved before checking, so `../` cannot escape an allowed root.
Logs are JSON on **stderr** — stdout carries the MCP protocol — and never
contain values from your data.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Config, errors, logging, registry, loaders, budgeting, `load_dataset` | **done** |
| 2 | `profile`, `analyze_column`, `find_issues`, `check_relationships`, `analyze_target`, `query` | next |
| 3 | Database read path — PostgreSQL, MySQL, SQLite, DuckDB, push-down profiling | planned |
| 4 | `clean_data`, `transform_data`, `reshape_data`, undo, `validate_rules` | planned |
| 5 | `plot`, `generate`, `export`, MCP resources | planned |
| 6 | Cost benchmark, performance gates, docs | planned |

---

## Development

```bash
uv sync --group dev
uv run pytest                       # 38 tests
uv run pytest --cov=eda_mcp         # coverage
uv run ruff check src tests         # lint
uv run ruff format src tests        # format
uv run mypy src/eda_mcp             # strict type check
```

Regenerate the test fixtures — deliberately broken files covering mixed types,
sorted data, disguised nulls, duplicates and cp1252 encoding:

```bash
uv run python tests/fixtures/make_fixtures.py
```

All three gates must pass before a change lands: `ruff`, `mypy --strict`, and
the test suite.

---

## Licence

Not yet licensed. The code is readable here, but until a licence is added it is
"all rights reserved" and cannot be reused. A licence will be chosen before the
first release.
