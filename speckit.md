# EDA MCP Server — Engineering Specification

| | |
|---|---|
| **Status** | Draft v5 — production design |
| **Date** | 2026-09-05 |
| **Author** | Muhammad Hammad Farooque |
| **Licence** | MIT (code) |
| **Target runtime** | Python 3.11–3.13, stdio MCP transport |
| **Protocol** | MCP 2026-07-28, `mcp` Python SDK 2.x |

---

# Part I — Product

## 1. Overview

An MCP server that performs a complete exploratory data analysis workflow
against **files or live databases** — inspecting data, diagnosing quality
problems, repairing them, engineering features, and writing results back to a
file or a table.

The assistant should carry out either of these end to end:

> "Load `sales.csv`, tell me what's wrong with it, fix the obvious problems,
> and save a cleaned copy."

> "Connect to the warehouse, profile the `orders` table, find the quality
> problems, and write a cleaned version to `orders_clean`."

## 2. Problem and market

### 2.1 Competitive position (verified 2026-09-04, GitHub API)

| Search | Repos | Best result |
|---|---|---|
| `exploratory data analysis mcp` | 4 | 1★ |
| `csv analysis mcp server` | 24 | `databeak` 2★ |
| `pandas mcp server` | 52 | `pandas-mcp-server` 45★ |
| `data analysis mcp server` | 477 | finance/charting, not EDA |

The 477 figure is misleading: the leaders (`mcp-server-chart` 4,352★,
`tradingview-mcp` 4,351★, `alpaca-mcp-server` 945★) are charting and trading
tools. No server owns "explain and repair this dataset."

Database MCP servers exist but are **query executors** — they run SQL and
return rows. None profile a table, diagnose its quality, or clean it.

### 2.2 Failure modes in existing tools

| Project | Defect | Consequence |
|---|---|---|
| `pandas-mcp-server` (45★) | Metadata computed from the **first 100 rows** | Silently wrong statistics on any sorted or grouped file |
| `databeak` (2★) | 40+ tool schemas loaded every turn | High fixed context cost before any work happens |
| Database MCP servers | Return raw rows | `SELECT * LIMIT 100` ≈ 6,000 tokens, says nothing about 40M rows |
| All of the above | No token budget | Context exhaustion mid-session |

### 2.3 The two constraints

**Context.** Raw statistical output is enormous: `describe()` on 50 columns,
a 30×30 correlation matrix (900 cells), a wide `SELECT *`. Tool results are
re-sent as input on **every subsequent turn**, so cost is quadratic in session
length (§10.1).

**Memory.** A 40M-row table does not fit in RAM, and transferring it to compute
a mean is absurd when the database can compute it in place.

**Consequence:** return *conclusions, not tables*; push computation *into the
source* wherever possible.

## 3. Goals and non-goals

### 3.1 Goals

- Correct statistics on the full population, or an explicit statement of the
  sample used
- Complete EDA lifecycle: inspect → diagnose → clean → transform → export
- Files and databases behind one interface
- Bounded, predictable token cost per call
- Safe by default: no in-place mutation of any source

### 3.2 Non-goals

- Model training or evaluation — this is EDA, not AutoML
- Dashboards or a web UI
- A general SQL client; `query` supports analysis, it does not replace a
  database IDE
- Out-of-core datasets beyond RAM — push-down is the answer for large tables
- Replacing a notebook for bespoke work; `query` and `generate` are the
  handoff points

## 4. Design principles

1. **Return findings, not data.** Every response is a budgeted digest.
2. **Push computation to the source.** Never `SELECT *` to compute a mean.
3. **Full-population statistics, or say so.** Never sample silently.
4. **Decisive over exhaustive.** One recommendation, not a menu — output
   tokens cost 5× input (§10.6).
5. **Few tools, batched operations.**
6. **Read-only by default.** Writes require explicit opt-in at two levels.
7. **Sources are never modified in place.**
8. **Every mutation is reversible and logged.**
9. **Fail loudly and structurally.** Errors are typed data, not prose (§9.2).
10. **Honest about uncertainty.** Report sample sizes; flag heuristics as
    heuristics.

---

# Part II — Architecture

## 5. System architecture

```
┌──────────────┐  stdio/MCP  ┌──────────────────────────────────────┐
│ MCP client   │ ──────────► │           EDA MCP Server             │
└──────────────┘             │                                      │
                             │  ┌────────────────────────────────┐  │
                             │  │       Tool layer (18)          │  │
                             │  │  validation · dispatch · errors│  │
                             │  └───────────────┬────────────────┘  │
                             │  ┌───────────────▼────────────────┐  │
                             │  │        Source Registry         │  │
                             │  │  connections · datasets · undo │  │
                             │  └───────┬────────────────┬───────┘  │
                             │  ┌───────▼──────┐  ┌──────▼───────┐  │
                             │  │  SQL engine  │  │ pandas engine│  │
                             │  │ (push-down)  │  │ (in-memory)  │  │
                             │  └───────┬──────┘  └──────┬───────┘  │
                             │          └────────┬───────┘          │
                             │            ┌──────▼──────┐           │
                             │            │   digest    │           │
                             │            │  budgeting  │           │
                             │            └─────────────┘           │
                             └──────┬──────────────────┬────────────┘
                          ┌─────────▼────────┐  ┌──────▼──────┐
                          │    databases     │  │ local files │
                          └──────────────────┘  └─────────────┘
```

### 5.1 Execution engines

| | SQL engine | pandas engine |
|---|---|---|
| Used for | live tables | files, query results, loaded data |
| Computation | aggregate SQL, in-database | in-memory DataFrame |
| Size limit | unbounded | available RAM |
| Mutation | never | in-session, undoable |

Dispatch is automatic: an unloaded table is profiled with SQL; once
`load_dataset` pulls it in, pandas takes over.

### 5.2 Source registry

**Connection:** `alias`, SQLAlchemy `engine`, `dialect`, `read_only`,
`opened_at`, `pool_stats`.

**Dataset:** `alias`, `df`, `origin` (path or `{connection, query}`),
`history`, `snapshots`, `loaded_at`, `bytes`.

## 6. Database connectivity

### 6.1 Dialects

| Dialect | Driver | Extra | Notes |
|---|---|---|---|
| PostgreSQL | `psycopg[binary]` | `[postgres]` | Primary target |
| MySQL / MariaDB | `pymysql` | `[mysql]` | |
| SQLite | stdlib | — | Zero-config; CI default |
| SQL Server | `pyodbc` | `[mssql]` | Needs system ODBC driver |
| DuckDB | `duckdb` | `[duckdb]` | Also reads Parquet/CSV directly |

All access is via **SQLAlchemy 2.x**, with dialect-specific SQL where
profiling requires it. Drivers are optional extras so a CSV-only user does not
pull an ODBC stack.

### 6.2 Credentials

Resolution order:

1. `EDA_MCP_DSN_<ALIAS>` environment variable
2. `DATABASE_URL`
3. `~/.eda-mcp/config.toml`, refused if group- or world-readable
4. Explicit `dsn` argument — permitted, but the server returns a warning that
   the value now resides in conversation history

Secrets are redacted in every response, log record, error, and generated
artefact. Generated code emits `os.environ[...]`, never a literal.

### 6.3 Read-only enforcement

Model-composed SQL is untrusted input. Four independent layers:

1. **Session** — `SET TRANSACTION READ ONLY` (PostgreSQL),
   `SET SESSION TRANSACTION READ ONLY` (MySQL), `file:...?mode=ro` (SQLite)
2. **Statement** — parsed with `sqlglot`; anything but `SELECT`/`WITH` is
   refused before reaching the driver. Blocks stacked statements, comment
   smuggling, CTE-wrapped DML, and `SELECT ... INTO`
3. **Binding** — all user values are bound parameters, never interpolated
4. **Limits** — `statement_timeout` (default 30 s) and a mandatory row cap on
   non-aggregate queries

Writes require `read_only=False` at connect **and** `confirm=True` per call.

### 6.4 Push-down profiling

For tables of unknown size, one aggregate query per column batch replaces row
transfer:

```sql
SELECT
  COUNT(*)                        AS n_rows,
  COUNT(price)                    AS n_present,
  COUNT(DISTINCT price)           AS n_distinct,
  MIN(price), MAX(price), AVG(price),
  STDDEV_POP(price)               AS sd,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS p75
FROM orders;
```

Yields distribution shape, missingness, cardinality and outlier bounds on 40M
rows without moving a row. Categorical columns use a bounded
`GROUP BY … ORDER BY count DESC LIMIT 20` plus a distinct count.

### 6.5 Sampling and estimation

When a statistic cannot be pushed down (skew, correlation, model-based outlier
detection), the server samples and **always reports that it did**.

| Dialect | Method |
|---|---|
| PostgreSQL | `TABLESAMPLE BERNOULLI (p)` |
| SQL Server | `TABLESAMPLE (n ROWS)` |
| MySQL | `ORDER BY RAND() LIMIT n` (warned: full scan) |
| SQLite / DuckDB | `USING SAMPLE n` / `ORDER BY random() LIMIT n` |

Default 100,000 rows, seeded for reproducibility (§13.4). Row counts come from
planner estimates (`pg_class.reltuples`) and are marked
`row_count_exact: false`; exact counts are computed only on request.

## 7. Tool specification

**18 tools.** Consolidated from a 25-tool draft on cost grounds (§10.2), under
one rule: **merge tools that differ only in source, never tools that differ in
meaning.** A polymorphic schema the model must disambiguate causes wrong calls,
and one retry costs more than the schema saved.

`source` accepts a dataset alias or a `connection.table` reference and
dispatches per §5.1.

### 7.1 Sources

| Tool | Signature | Notes |
|---|---|---|
| `connect_database` | `(alias, dsn?, env_var?, read_only=True)` | Returns dialect, version, schemas, table count |
| `explore_schema` | `(connection, schema?, table?)` | Progressive: schemas → tables → columns/keys/indexes |
| `load_dataset` | `(source, alias, query?, limit?, options={})` | Returns orientation digest **including top findings** |
| `manage_sources` | `(action="list"\|"close", alias?)` | Listing and release |

`load_dataset` refuses tables above `max_load_rows` (default 5M) without
`limit`/`sample`, and points at `profile`.

### 7.2 Analysis

| Tool | Signature | Returns |
|---|---|---|
| `profile` | `(source, detail="standard", columns?)` | Type classification, missingness, duplicates, distributions, ranked findings |
| `analyze_column` | `(source, column)` | Type-adapted deep dive |
| `find_issues` | `(source, severity="all")` | Ranked problems **with recommended fixes** |
| `check_relationships` | `(source, target?, group_by?)` | Ranked strong pairs, collinear clusters, group comparison with effect sizes |
| `analyze_target` | `(source, target)` | Class balance, feature strength, **leakage detection** |
| `query` | `(source, expression, limit=20)` | pandas expression or read-only SQL |
| `validate_rules` | `(source, rules=[...])` | Pass/fail with violating counts and examples |

`profile` returns problem columns in full plus a one-line roll-up of clean
ones (§10.5) — critical on wide tables.

`check_relationships` never returns a full correlation matrix; only ranked
pairs above a threshold.

### 7.3 Mutation **[W]**

| Tool | Signature |
|---|---|
| `clean_data` | `(alias, operations=[...])` |
| `transform_data` | `(alias, operations=[...])` |
| `reshape_data` | `(alias, operation, ...)` |
| `history` | `(alias, action="list"\|"undo", steps=1)` |

**`clean_data` operations:** `fill_missing` (mean/median/mode/constant/ffill/
bfill/interpolate/knn), `drop_missing`, `drop_duplicates`, `drop_columns`,
`drop_rows`, `remove_outliers` (iqr/zscore/modified_zscore/percentile ×
drop/clip/flag), `replace_values`, `rename_columns`, `cast_type`,
`strip_whitespace`, `standardize_case`, `parse_dates`.

**`transform_data` operations:** `encode_categorical` (onehot/ordinal/target/
frequency), `scale` (standard/minmax/robust), `bin` (cut/qcut),
`log_transform`, `power_transform` (boxcox/yeo-johnson), `extract_datetime`
(incl. cyclical), `create_feature`, `lag`, `rolling`.

**`reshape_data` operations:** `pivot`, `melt`, `merge`, `concat`, `sort`,
`sample`, `split`, `groupby_agg`. Merges return join diagnostics — unmatched
counts and fan-out detection.

Mutating tools return a **delta** — operations applied, rows/columns affected,
before/after shape, and any operation refused with its reason. Never a
re-profile.

### 7.4 Output

| Tool | Signature | Notes |
|---|---|---|
| `plot` | `(alias, kind, columns, save_path?, options={})` | Returns **path + description**, never image bytes |
| `generate` | `(alias, kind="code"\|"report", path?, style="pandas")` | Script or narrative report |
| `export` | `(alias, destination, format, mode, confirm, overwrite)` | File path **or** `connection.table` |

`plot` kinds: `histogram`, `box`, `violin`, `scatter`, `hexbin`, `bar`, `line`,
`heatmap`, `pairplot`, `missingness`, `qq`, `ecdf`, `crosstab_heatmap`.

`export` refuses to overwrite the origin file without `overwrite=True`, and
requires `read_only=False` plus `confirm=True` for table writes. `mode=replace`
on a source table additionally needs `allow_source_overwrite=True`.

### 7.5 Tool groups

The loaded set is fixed at **startup** by `EDA_MCP_TOOLS`:

| Group | Tools | Schema cost/turn |
|---|---|---|
| `core` | 10 — files, analysis, cleaning, export | ~1,000 tok |
| `full` (default) | 18 — adds database, plots, reports | ~1,800 tok |
| `readonly` | 11 — analysis only | ~1,100 tok |

Fixed at startup, never mid-session: dynamic registration invalidates the
prompt cache and costs more than it saves (§10.3).

### 7.6 Server instructions

The MCP `instructions` field is sent **once** at initialize and then costs
nothing for the rest of the session. It is the cheapest available defence
against tool misuse, and one prevented wrong call repays it many times over
(§10.4).

Kept under ~200 tokens, byte-stable for cache safety (§10.3):

```text
This server performs exploratory data analysis on files and databases.

Choosing a tool:
- load_dataset already returns a profile and top findings. Do not call
  profile immediately after it.
- For database tables, call profile directly — it computes in SQL without
  transferring rows. Only load_dataset when you need row-level operations.
- find_issues returns a recommended fix with each problem. Do not ask which
  fix to apply; apply it or explain why not.
- clean_data and transform_data take a list of operations. Batch them into
  one call rather than calling repeatedly.

Reading results:
- Statistics cover the full dataset unless a "sampled" field is present.
- "resources" lists URIs holding full detail; read one only if the digest is
  insufficient.
- "truncated" states what was omitted and how to retrieve it.

Writing:
- Sources are never modified in place. export writes elsewhere.
- Mutations are reversible with history(action="undo").
```

The instructions state **behaviour the response format cannot express** —
notably that `load_dataset` already profiles, which removes the most common
redundant round trip in the whole workflow.

### 7.7 Tool annotations

Every tool carries MCP behavioural hints. Clients use them to skip
confirmation prompts on safe calls and to gate risky ones — fewer
interruptions, fewer wasted turns.

| Tool | `readOnly` | `destructive` | `idempotent` | `openWorld` |
|---|---|---|---|---|
| `connect_database` | ✓ | | ✓ | ✓ |
| `explore_schema` | ✓ | | ✓ | ✓ |
| `load_dataset` | ✓ | | ✓ | ✓ |
| `manage_sources` | | | | |
| `profile` | ✓ | | ✓ | ✓ |
| `analyze_column` | ✓ | | ✓ | |
| `find_issues` | ✓ | | ✓ | |
| `check_relationships` | ✓ | | ✓ | |
| `analyze_target` | ✓ | | ✓ | |
| `query` | ✓ | | ✓ | ✓ |
| `validate_rules` | ✓ | | ✓ | |
| `clean_data` | | | | |
| `transform_data` | | | | |
| `reshape_data` | | | | |
| `history` | | | | |
| `plot` | | | ✓ | |
| `generate` | | | ✓ | |
| `export` | | **✓** | | ✓ |

Notes on the judgements:

- **`readOnly`** means *no source is modified*. Analysis tools qualify;
  mutation tools do not, even though they only alter an in-memory session.
- **`destructive`** is reserved for `export`, the one tool that can overwrite
  a file or replace a table. Session mutations are not destructive because
  they are reversible via `history` (§13.2).
- **`idempotent`** marks calls safe to repeat. `clean_data` is not — applying
  `drop_duplicates` twice is harmless, but `remove_outliers` twice removes a
  second tranche.
- **`openWorld`** marks tools touching state outside the process — the
  filesystem or a database.

Annotations are hints, not enforcement. The refusal policy (§12.2) and the
two-level write gate (§7.4) remain the actual controls.


## 8. Resources

Large artefacts are exposed as **MCP resources**, not tool results — the client
fetches them only if needed, so they cost zero tokens otherwise.

| URI | Content |
|---|---|
| `eda://{alias}/profile/full` | Complete per-column profile |
| `eda://{alias}/correlations` | Full correlation matrix |
| `eda://{alias}/history` | Complete audit trail |
| `eda://{alias}/report` | Generated narrative report |
| `eda://{alias}/sample?n=` | Row sample |

Tool responses reference the URI; the model retrieves it only when the digest
is insufficient. This is the single largest structural saving available
(§10.5).

## 9. Response and error model

### 9.1 Success envelope

```json
{
  "source": {"kind": "database", "connection": "warehouse", "table": "orders"},
  "shape": [41238904, 24],
  "row_count_exact": false,
  "sampled": {"n": 100000, "of": 41238904, "method": "bernoulli", "seed": 42},
  "findings": [
    "HIGH  region: 43% null, correlated with signup_date -> add missingness flag (17.7M rows)",
    "MED   price: right-skewed (2.3), 47 outliers beyond 1.5 IQR -> consider log transform"
  ],
  "summary": "24 columns, ~41M rows. 3 columns need attention before modelling.",
  "resources": ["eda://orders/profile/full"],
  "truncated": {"omitted": 6, "reason": "token_budget"}
}
```

Conventions (§10.5): findings are ranked terse lines, not nested objects;
floats are rounded to 3 significant figures; null and absent fields are
omitted entirely; JSON structure is reserved for values a caller must parse.

### 9.2 Error taxonomy

Errors are structured data with stable codes, never prose. Every error states
what failed, why, and the next action.

```json
{
  "error": {
    "code": "SOURCE_TOO_LARGE",
    "message": "orders has ~41.2M rows, above max_load_rows (5M)",
    "remedy": "use profile() for push-down analysis, or pass limit=",
    "retryable": false
  }
}
```

| Code | Class | Retryable |
|---|---|---|
| `SOURCE_NOT_FOUND` | user | no |
| `SOURCE_TOO_LARGE` | user | no |
| `COLUMN_NOT_FOUND` | user | no |
| `INVALID_OPERATION` | user | no |
| `OPERATION_REFUSED` | policy | no |
| `WRITE_NOT_PERMITTED` | policy | no |
| `STATEMENT_REJECTED` | policy | no |
| `CONNECTION_FAILED` | infra | yes |
| `QUERY_TIMEOUT` | infra | yes |
| `MEMORY_LIMIT_EXCEEDED` | infra | no |
| `DEPENDENCY_MISSING` | config | no |
| `INTERNAL_ERROR` | bug | no |

**Policy** errors are deliberate refusals (§12.2) and are returned as findings
with reasons, not exceptions. **Infra** errors carry `retryable: true` and a
suggested backoff. `INTERNAL_ERROR` includes a correlation id matching a log
record; it never leaks a stack trace or a DSN.

---

# Part III — Engineering

## 10. Cost and performance design

### 10.1 Cost model

A tool result is not billed once — it enters conversation history and is
re-sent as input every subsequent turn:

```
total_input ≈ r × n(n+1)/2  +  s × n  +  conversation
              ↑ results        ↑ schemas
```

Quadratic in turns `n`, linear in schema size `s`. A 4,000-token result on turn
3 of a 15-turn session is billed 13 more times. Three levers follow: shrink
`s`, shrink `r`, and — worth most — shrink `n`.

### 10.2 Schema size (`s`)

25 → 18 tools by source-polymorphic merging only. Descriptions target ≤ 100
tokens: enums over prose, no restated parameter names, no inline examples.
Under-describing is a false economy — one misuse costs a whole round trip.
Tool groups (§7.5) let a CSV user load 10 instead of 18.

Result: `s` ≈ 1,800 tok/turn, from ~3,750.

### 10.3 Cache stability

Schemas are identical every turn, so they cache at a fraction of input price.
This imposes a hard rule:

> **Tool definitions must be byte-stable for the entire session.**

No session state in descriptions, no timestamps, no dynamic enums of current
aliases. Any of these invalidate the cache and re-bill the whole schema block
every turn — far more than the dynamic hint is worth. This is why tool groups
are chosen at startup (§7.5).

### 10.4 Round trips (`n`)

Because cost is quadratic in `n`, eliminating one round trip beats shrinking
several results.

1. `load_dataset` returns findings, so the routine follow-up `profile` never
   happens
2. `find_issues` ships the fix with the diagnosis
3. `clean_data` applies twelve operations in one turn
4. Mutations return deltas, not restatements
5. Truncation states what was omitted and how to get it — never a cursor that
   guarantees another call
6. Server `instructions` (§7.6) state behaviour the schema cannot — chiefly
   that `load_dataset` already profiles, removing the most common redundant
   call. Sent once; free thereafter
7. Tool annotations (§7.7) let clients skip confirmation on read-only calls,
   removing interruption turns

### 10.5 Result size (`r`)

**Budgets**, enforced by severity-ranked emission; never truncate
mid-structure.

| Tool | Cap |
|---|---|
| `profile` | 1,500 |
| `find_issues` | 1,200 |
| `explore_schema` | 1,000 |
| `analyze_target`, `check_relationships`, `load_dataset` | 800 |
| `analyze_column` | 600 |
| all others | 500 |

**Encoding rules**, each measured against a raw baseline:

| Rule | Saving |
|---|---|
| Round floats to 3 significant figures | ~5 tok → ~2 tok per number |
| Omit null and absent fields entirely | ~15 tok per omitted field |
| Findings as ranked terse lines, not nested JSON | ~15 tok per finding |
| Column subsetting: problem columns in full, clean ones rolled up | dominant on wide tables |
| Handles not payloads — paths and resource URIs | ~1,500 tok per image avoided |
| No raw dumps: no `describe()`, no correlation matrices, no `SELECT *` | — |

### 10.6 Output tokens

Output costs **5× input** ($25/M vs $5/M on Opus 5). A response that invites
deliberation — "here are three options with tradeoffs" — burns the expensive
side of the ledger. Every tool therefore returns **one recommendation, not a
menu**, and states findings as conclusions. Decisiveness is a cost
optimisation, and it is also better analysis.

### 10.7 Projected cost

15-turn session, priced at Sonnet 5 ($2/M) and Opus 5 ($5/M) input:

| Design | Results | Schemas | Total | Sonnet 5 | Opus 5 |
|---|---|---|---|---|---|
| Naive (raw output, 40 tools) | 480,000 | 90,000 | 570,000 | $1.14 | $2.85 |
| v2 (25 tools, budgeted) | 72,000 | 56,250 | 128,000 | $0.26 | $0.64 |
| **v4 (18 tools, encoded)** | **48,000** | **27,000** | **75,000** | **$0.15** | **$0.38** |
| v4 + prompt caching | 48,000 | ~2,700 | ~51,000 | $0.10 | $0.26 |
| + round-trip reduction (15→10) | 22,000 | ~1,800 | ~24,000 | **$0.05** | **$0.12** |

~**8× cheaper than naive at parity; ~24× with caching and fewer round trips.**

> **These are projections from estimated per-call sizes, not measurements.**
> §16.5 defines the benchmark that must replace them before v1.0. If measured
> figures differ materially, this table is corrected, not the benchmark.

### 10.8 Rejected optimisations

| Rejected | Reason |
|---|---|
| Merging clean/transform/reshape | Saves ~100 tok; yields a 35-value enum the model gets wrong |
| Dynamic tool registration | Destroys cache stability (§10.3) |
| Descriptions below ~100 tok | Misuse costs more than the savings |
| Sampling by default | Silently wrong statistics defeat the purpose |
| Dropping fixes from `find_issues` | Smaller responses, more round trips — net loss |
| Declaring `outputSchema` | Re-sent every turn like a tool schema; net loss at 18 tools |
| Server-side result caching | Saves latency and compute, saves zero tokens |
| Composite `run_workflow` tool | Deferred, not rejected: collapses n but its value depends on usage we have not observed. Revisit after §16.5 measures real sessions |
| Elicitation on ambiguity | Picking the obvious reading and stating it costs the same one turn |
| Differential responses across turns | Requires the model to recall prior output reliably; silent wrongness violates §4.10 |

## 11. Performance targets

Measured on the reference dataset (§16.4) on a 4-core/16 GB runner.

| Operation | Target (p95) |
|---|---|
| `load_dataset`, 100 MB CSV | < 8 s |
| `profile`, 1M rows × 50 cols | < 5 s |
| `profile` push-down, 40M-row table | < 15 s |
| `analyze_column`, 1M rows | < 1 s |
| `check_relationships`, 50 numeric cols | < 3 s |
| `clean_data`, 10 operations, 1M rows | < 10 s |
| Server cold start | < 1.5 s |

Memory: peak RSS ≤ 3× the loaded dataset size. Exceeding `max_memory_mb`
(default 4,096) raises `MEMORY_LIMIT_EXCEEDED` before the allocation, not
after.

## 12. Security

### 12.1 Threat model

| Threat | Vector | Mitigation |
|---|---|---|
| Destructive SQL | Model composes `DROP`/`DELETE` | Four-layer read-only enforcement (§6.3) |
| SQL injection | User value interpolated | Bound parameters only; `sqlglot` validation |
| Credential leakage | DSN in response, log, or generated code | Redaction at the serialisation boundary; `os.environ` in codegen |
| Path traversal | `export(destination="../../etc/passwd")` | Path resolution confined to `allowed_paths` |
| Arbitrary code execution | `create_feature`, `query` expressions | pandas restricted `eval`; no `__import__`, attribute access, or I/O |
| Resource exhaustion | Unbounded query or load | Row caps, `statement_timeout`, memory ceiling |
| Data exfiltration | Server reaching the network | No outbound network calls; stdio transport only |
| Supply chain | Compromised dependency | Pinned lockfile, `pip-audit` in CI, SBOM published per release |

### 12.2 Refusal policy

Returned as findings with reasons, never silent, never exceptions:

- dropping > 50% of rows in one operation
- dropping a column carrying > 90% of remaining information
- imputing a column > 60% missing (recommend a flag instead)
- any non-`SELECT` on a read-only connection
- unbounded queries above the row threshold
- writing to a path outside `allowed_paths`

### 12.3 Boundaries

The server makes **no outbound network connections** other than to explicitly
configured databases. It reads only paths under `allowed_paths` (default: the
working directory) and writes only where told. There is no telemetry.

## 13. Reliability

### 13.1 Resource management

Connections use a bounded SQLAlchemy pool (default `pool_size=2`,
`max_overflow=3`, `pool_pre_ping=True`, `pool_recycle=1800`). All connections
and cursors close on server shutdown, including on `SIGTERM`. Every session
runs under `statement_timeout`.

### 13.2 Undo

Snapshots for the last 10 mutations, capped at `max_snapshot_mb` (default 500)
in aggregate. Beyond the cap the oldest is evicted and its history entry marked
`undoable: false` — degradation is visible, never silent.

### 13.3 Degradation

A missing optional driver yields `DEPENDENCY_MISSING` naming the extra to
install, not an ImportError traceback. A failure on one column during a
multi-column profile marks that column `failed` with a reason and completes the
rest. Partial results are always labelled.

### 13.4 Determinism

Every sampling operation takes a seed (default 42) and reports it. Two runs of
the same call on unchanged data return identical results. `generate(kind=
"code")` output is deterministic and diffable.

### 13.5 Concurrency

The server is single-session and processes tool calls serially; the registry is
not shared across processes. Blocking pandas and database work runs in a
thread executor so the MCP event loop stays responsive. Mutation of a dataset
takes a per-alias lock.

## 14. Observability

Structured JSON logs to **stderr** (stdout is the MCP transport — writing there
corrupts the protocol).

```json
{"ts":"2026-09-05T10:31:02Z","level":"INFO","event":"tool_call",
 "tool":"profile","source":"warehouse.orders","engine":"sql",
 "duration_ms":4820,"rows":41238904,"result_tokens":1420,
 "truncated":true,"correlation_id":"01JC..."}
```

Every call logs tool, source, engine, duration, rows touched, result token
count, truncation, and a correlation id. Errors log the same plus code and
class. **No values from the data and no credentials are ever logged.**

`EDA_MCP_LOG_LEVEL` controls verbosity; `DEBUG` adds generated SQL with
parameters redacted. Per-session counters (calls, tokens emitted, cache
evictions, refusals) are retrievable via `manage_sources(action="list")`.

## 15. Configuration

Precedence: CLI flag → environment variable → `~/.eda-mcp/config.toml` →
default. Validated at startup with **pydantic-settings**; invalid configuration
fails fast with the offending key named.

| Key | Env | Default |
|---|---|---|
| tool group | `EDA_MCP_TOOLS` | `full` |
| max rows to load | `EDA_MCP_MAX_LOAD_ROWS` | 5,000,000 |
| memory ceiling (MB) | `EDA_MCP_MAX_MEMORY_MB` | 4,096 |
| snapshot cap (MB) | `EDA_MCP_MAX_SNAPSHOT_MB` | 500 |
| statement timeout (s) | `EDA_MCP_STATEMENT_TIMEOUT` | 30 |
| sample size | `EDA_MCP_SAMPLE_SIZE` | 100,000 |
| random seed | `EDA_MCP_SEED` | 42 |
| allowed paths | `EDA_MCP_ALLOWED_PATHS` | cwd |
| log level | `EDA_MCP_LOG_LEVEL` | `INFO` |

## 16. Quality engineering

### 16.1 Type safety

`mypy --strict` across `src/`, no `Any` in public signatures, no unchecked
`type: ignore`. Every tool's arguments and response are **pydantic models** —
validation happens at the boundary, so tool bodies never defend against
malformed input. `ruff` for lint and format, `ruff --select S` (bandit rules)
for security lint.

### 16.2 Test strategy

| Layer | Scope | Target |
|---|---|---|
| Unit | Pure functions: statistics, digests, type inference | ≥ 90% line coverage |
| Property-based | `hypothesis` — invariants over generated frames | — |
| Integration | Full tool calls against fixtures | every tool, every error code |
| Adversarial | SQL guard bypass attempts | 100% must fail closed |
| Performance | Benchmarks against §11 targets | regression gate |
| Contract | MCP protocol conformance | schema validity, no stdout writes |

### 16.3 Invariants

Property-based tests assert, over arbitrary generated frames:

- statistics computed on the **full** dataset match pandas exactly
- SQL push-down results match pandas results on identical data
- every mutation is exactly reversible by `history(action="undo")`
- the source file is byte-identical after any session
- every response is within its token budget
- no response contains a credential
- generated code re-executed reproduces the final frame exactly

### 16.4 Fixtures

**Files** — deliberately broken: mixed types in one column; a **sorted** file
(catches the first-100-rows class of bug); 60% missing; duplicates;
inconsistent casing; an identifier column; a leaked target; three date
formats; 100 MB for performance; UTF-8, latin-1 and cp1252 encodings.

**Databases** — a seeded SQLite schema for offline CI, and a docker-compose
PostgreSQL with a 5M-row table exercising push-down, sampling and estimation.

### 16.5 Cost benchmark

A required suite that **measures** what §10.7 projects: it runs a scripted
15-turn session against the reference dataset, counts tokens with the
Anthropic tokenizer, and emits per-tool actuals.

Gates: no tool may exceed its §10.5 budget; total session input must be within
20% of the projection. **The projections in §10.7 are provisional until this
suite runs; measured values replace them before v1.0.**

### 16.6 CI

GitHub Actions on push and PR:

```
lint      ruff check · ruff format --check
types     mypy --strict
test      pytest, matrix: {3.11, 3.12, 3.13} × {ubuntu, macos, windows}
db        pytest -m integration (docker-compose postgres, mysql)
security  pip-audit · ruff --select S
bench     pytest-benchmark vs §11 · cost benchmark vs §10.5
```

Merge is blocked on any failure. Coverage may not drop. Benchmarks may not
regress more than 10% without an explicit waiver in the PR.

---

# Part IV — Delivery

## 17. Project structure

```
src/eda_mcp/
  __init__.py         entry point, version
  server.py           MCPServer, tool registration, groups
  config.py           pydantic-settings, validation
  errors.py           error taxonomy, codes, redaction
  logging.py          structured stderr logging
  registry.py         connections, datasets, history, snapshots
  models.py           pydantic request/response models
  db/
    connect.py        engine creation, DSN resolution, pooling
    introspect.py     schema/table/column discovery
    sqlprofile.py     push-down aggregate profiling
    guard.py          sqlglot validation, read-only enforcement
    sampling.py       dialect-specific sampling
  loaders.py          file reading, encoding/delimiter detection
  profiling.py        type inference, distributions
  issues.py           quality detection, severity ranking
  relations.py        correlation, collinearity
  target.py           balance, feature strength, leakage
  rules.py            business-rule validation
  mutations.py        clean, transform, reshape
  codegen.py          pandas/SQL script generation
  plots.py            chart rendering
  report.py           narrative report
  resources.py        MCP resource handlers (§8)
  digest.py           token budgeting, encoding rules
tests/
  unit/ integration/ adversarial/ bench/
  fixtures/
docs/
  README.md  ARCHITECTURE.md  SECURITY.md  CONTRIBUTING.md  CHANGELOG.md
```

## 18. Build roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **1** Foundation | `config`, `errors`, `logging`, `registry`, `loaders`, `digest`, `server` + `load_dataset`, `manage_sources` | Server connects; a messy CSV loads; budgets enforced; CI green |
| **2** Analysis | `profiling`, `issues`, `relations`, `target` + 7 analysis tools | Correct on sorted fixtures; property tests pass; **this is where the value is** |
| **3** Database read | `db/*` + `connect_database`, `explore_schema`, push-down `profile`, `query` | SQLite then PostgreSQL; adversarial SQL suite 100% closed |
| **4** Mutation | `mutations`, undo, `validate_rules` | Every mutation reversible; refusal policy enforced |
| **5** Output | `codegen`, `plots`, `report`, `resources`, `export` | Generated code reproduces session exactly |
| **6** Hardening | Cost benchmark, performance gates, docs, SBOM | §10.7 measured; §11 met; v1.0.0 tagged |

*Deferred:* time-series tooling (§A.21), Snowflake/BigQuery, out-of-core
datasets.

## 19. Versioning and release

**Semantic versioning.** The public API is the tool surface: names, parameters,
response field names, and error codes.

- **Major** — removing or renaming a tool, parameter, or error code; changing a
  response field's meaning
- **Minor** — new tools, new optional parameters, new operations, new fields
- **Patch** — fixes that do not change the contract

Deprecation runs one minor release with a `deprecated` flag in the response
before removal. Every release publishes a signed sdist and wheel to PyPI, a
CycloneDX SBOM, and a `CHANGELOG.md` entry.

## 20. Documentation

`README.md` — install, configure, three worked examples (CSV, database, full
clean-and-export). `ARCHITECTURE.md` — engines, registry, budgeting.
`SECURITY.md` — threat model and reporting policy. `CONTRIBUTING.md` — setup,
test invocation, review expectations. Every tool documents parameters,
response shape, error codes, and token budget.

## 21. Success criteria

**Functional**

1. A messy CSV is profiled, diagnosed, cleaned and exported in one conversation
   without hand-written pandas
2. A 40M-row table is profiled without transferring rows
3. Statistics are correct on sorted and grouped data, where `pandas-mcp-server`
   is wrong
4. Generated code reproduces the session exactly

**Non-functional**

5. Full profile of a 50-column dataset under 1,500 measured tokens
6. §11 performance targets met at p95
7. No model-generated SQL writes to a read-only connection (adversarial suite)
8. No credential appears in any response, log, or artefact
9. `mypy --strict` clean; ≥ 90% unit coverage
10. Installable in under two minutes with `uv`

## 22. Open questions

- Should `export` support `UPSERT`/merge, or is create/append enough for v1?
- Should DuckDB back large **file** analysis, not just be a connectable
  dialect? It would lift the RAM ceiling and unify the engines.
- Do MCP resources (§8) warrant carrying the full profile, or does the digest
  make them redundant in practice?
- How much merge capability belongs in `reshape_data` before it becomes a join
  tool in its own right?
- Is a hosted HTTP deployment ever in scope, or is stdio-only a permanent
  boundary? It changes the auth and multi-tenancy story entirely.

---
## Appendix A — Full EDA process reference

The complete catalogue of EDA steps and the code that implements them. This is
the source material for the tool design above: every section maps to a tool in
§6, and the mapping is given at the end of this appendix.


Every step, check, and operation in a thorough exploratory data analysis, with
the code that implements it. This is the source material for the EDA MCP
server's tool design: each section maps to something the server must be able
to do.

**Legend:** 🔍 analysis (read-only) · ✏️ mutation (write) · 📊 visualisation

---

### 0. Setup

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
```

---

### 1. Loading 🔍

#### 1.1 Formats

```python
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json", lines=True)
df = pd.read_sql("SELECT * FROM t", con)
df = pd.read_csv("data.tsv", sep="\t")
df = pd.read_clipboard()
```

#### 1.2 Reading awkward files

```python
# Encoding problems
df = pd.read_csv(f, encoding="utf-8")        # try first
df = pd.read_csv(f, encoding="latin-1")      # fallback
df = pd.read_csv(f, encoding="cp1252")       # Windows exports

# Detect it instead of guessing
import chardet
enc = chardet.detect(open(f, "rb").read(100_000))["encoding"]

# Unknown delimiter
df = pd.read_csv(f, sep=None, engine="python")   # sniff

# Messy structure
df = pd.read_csv(f, skiprows=3)                  # junk header rows
df = pd.read_csv(f, header=None, names=cols)     # no header
df = pd.read_csv(f, skipfooter=2, engine="python")
df = pd.read_csv(f, thousands=",", decimal=".")
df = pd.read_csv(f, na_values=["", "NA", "N/A", "null", "-", "?", "missing"])
df = pd.read_csv(f, true_values=["yes","Y"], false_values=["no","N"])
df = pd.read_csv(f, parse_dates=["date"], dayfirst=True)
df = pd.read_csv(f, dtype={"zip": str})          # preserve leading zeros
df = pd.read_csv(f, usecols=["a","b"])           # subset for speed
df = pd.read_csv(f, nrows=1000)                  # peek
```

#### 1.3 Large files

```python
# Chunked
for chunk in pd.read_csv(f, chunksize=100_000):
    process(chunk)

# Memory-efficient dtypes on load
df = pd.read_csv(f, dtype={"category_col": "category"}, engine="pyarrow")
```

---

### 2. First look 🔍

```python
df.shape                    # (rows, cols)
df.head(10); df.tail(10); df.sample(10)
df.info(memory_usage="deep")
df.dtypes
df.columns.tolist()
df.index
len(df); df.size
df.memory_usage(deep=True).sum() / 1024**2      # MB
df.describe()                                    # numeric
df.describe(include="object")                    # categorical
df.describe(include="all")
df.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
```

**Questions to answer here:** What does one row represent? What is the grain?
Is there a natural key? Which column is the target?

---

### 3. Types 🔍 ✏️

#### 3.1 Classify every column

```python
numeric  = df.select_dtypes(include=np.number).columns.tolist()
cats     = df.select_dtypes(include=["object","category"]).columns.tolist()
dates    = df.select_dtypes(include="datetime").columns.tolist()
bools    = df.select_dtypes(include="bool").columns.tolist()
```

#### 3.2 Detect wrong types

```python
# Numbers stored as text
for c in cats:
    converted = pd.to_numeric(df[c], errors="coerce")
    if converted.notna().mean() > 0.9:
        print(f"{c}: {converted.notna().mean():.0%} numeric — likely mistyped")

# Dates stored as text
for c in cats:
    parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        print(f"{c}: parses as datetime")

# Mixed types in one column
for c in cats:
    kinds = df[c].dropna().map(type).value_counts()
    if len(kinds) > 1:
        print(f"{c}: mixed types {dict(kinds)}")

# Identifier columns (unique per row)
for c in df.columns:
    if df[c].nunique() == len(df):
        print(f"{c}: unique per row — identifier, not a feature")

# Constant / near-constant
for c in df.columns:
    top = df[c].value_counts(normalize=True, dropna=False).iloc[0]
    if top > 0.99:
        print(f"{c}: {top:.1%} single value — no signal")
```

#### 3.3 Convert ✏️

```python
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["d"] = pd.to_datetime(df["d"], errors="coerce", format="%Y-%m-%d")
df["c"] = df["c"].astype("category")
df["i"] = df["i"].astype("Int64")          # nullable integer
df["b"] = df["b"].astype(bool)
df["s"] = df["s"].astype("string")         # nullable string

# Downcast to save memory
df["n"] = pd.to_numeric(df["n"], downcast="integer")
df["f"] = pd.to_numeric(df["f"], downcast="float")
```

---

### 4. Missing data 🔍 ✏️

#### 4.1 Detect

```python
df.isna().sum()
df.isna().mean().sort_values(ascending=False)      # proportion
df.isna().sum().sum()                              # total
df.isna().any(axis=1).sum()                        # rows with any NA
df[df.isna().any(axis=1)]                          # inspect them
df.notna().sum()

# Disguised missing values
for c in cats:
    suspicious = df[c].isin(["", " ", "NA", "N/A", "null", "None",
                             "-", "?", "unknown", "missing", "nan"])
    if suspicious.any():
        print(f"{c}: {suspicious.sum()} disguised missing")

# Sentinel values in numerics
for c in numeric:
    for sentinel in (-1, -999, 9999, 0):
        n = (df[c] == sentinel).sum()
        if n > len(df) * 0.05:
            print(f"{c}: {n} occurrences of {sentinel} — possible sentinel")
```

#### 4.2 Understand the pattern

```python
# Is missingness related to other columns? (MAR vs MCAR)
for c in df.columns[df.isna().any()]:
    flag = df[c].isna()
    for other in numeric:
        if other == c: continue
        t, p = stats.ttest_ind(df.loc[flag, other].dropna(),
                               df.loc[~flag, other].dropna(),
                               equal_var=False)
        if p < 0.01:
            print(f"{c} missingness relates to {other} (p={p:.4f}) — MAR, not MCAR")

# Do columns go missing together?
df.isna().corr()

# Missingness by group
df.groupby("segment").apply(lambda g: g.isna().mean())
```

📊 `sns.heatmap(df.isna(), cbar=False)` — the missingness matrix.

#### 4.3 Handle ✏️

```python
# Drop
df.dropna()                                  # any NA
df.dropna(how="all")                         # entirely empty rows
df.dropna(subset=["important"])              # only where key col missing
df.dropna(thresh=int(0.8*df.shape[1]))       # keep rows ≥80% complete
df.dropna(axis=1, thresh=int(0.6*len(df)))   # drop sparse columns

# Impute
df["x"].fillna(df["x"].mean())
df["x"].fillna(df["x"].median())             # safer with outliers
df["c"].fillna(df["c"].mode()[0])
df["x"].fillna(0)
df["x"].ffill(); df["x"].bfill()             # time series
df["x"].interpolate(method="linear")
df["x"].fillna(df.groupby("g")["x"].transform("median"))   # group-wise

# Model-based
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
df[numeric] = KNNImputer(n_neighbors=5).fit_transform(df[numeric])
df[numeric] = IterativeImputer(random_state=0).fit_transform(df[numeric])

# Flag instead of impute — preferred when missingness is informative
df["x_was_missing"] = df["x"].isna().astype(int)
df["x"] = df["x"].fillna(df["x"].median())
```

**Rule of thumb:** >60% missing → drop or flag, don't impute. Missing not at
random → always add a flag.

---

### 5. Duplicates 🔍 ✏️

```python
df.duplicated().sum()
df[df.duplicated(keep=False)].sort_values(by=cols)     # see all copies
df.duplicated(subset=["id"]).sum()                     # key duplicates
df.drop_duplicates()                                   # ✏️
df.drop_duplicates(subset=["id"], keep="last")         # ✏️

# Near-duplicates (fuzzy)
from difflib import SequenceMatcher
# or: rapidfuzz, recordlinkage for scale
```

---

### 6. Univariate — numeric 🔍

```python
s = df["x"]
s.mean(); s.median(); s.mode()
s.std(); s.var(); s.min(); s.max()
s.quantile([.25,.5,.75])
s.max() - s.min()                             # range
s.quantile(.75) - s.quantile(.25)             # IQR
s.skew()                                      # >1 right, <-1 left
s.kurtosis()                                  # >3 heavy tails
stats.variation(s.dropna())                   # coefficient of variation
s.nunique(); s.value_counts()
(s == 0).sum(); (s < 0).sum()                 # zeros, negatives
s.is_monotonic_increasing
```

#### Normality

```python
stats.shapiro(s.sample(min(5000, len(s))))    # n < 5000
stats.normaltest(s.dropna())                  # D'Agostino
stats.anderson(s.dropna())
stats.jarque_bera(s.dropna())
```

📊 histogram, KDE, boxplot, violin, Q-Q plot (`stats.probplot`), ECDF.

---

### 7. Univariate — categorical 🔍

```python
s = df["c"]
s.value_counts()
s.value_counts(normalize=True)
s.value_counts(dropna=False)
s.nunique()
s.nunique() / len(s)                          # cardinality ratio
s.mode()

# Rare categories
rare = s.value_counts(normalize=True)
rare[rare < 0.01].index.tolist()

# Inconsistent encoding — the classic USA/usa/U.S.A. problem
norm = s.astype(str).str.strip().str.lower()
collisions = norm.value_counts()[norm.value_counts() != s.value_counts().reindex(norm.unique()).fillna(0)]
s.str.strip().str.lower().nunique() < s.nunique()     # True = inconsistent

# High cardinality warning
if s.nunique() > 50:
    print("high cardinality — one-hot will explode; consider target/frequency encoding")
```

📊 bar chart, count plot, pie (sparingly), treemap.

---

### 8. Univariate — datetime 🔍

```python
s = df["date"]
s.min(); s.max(); s.max() - s.min()           # range
s.dt.year.value_counts().sort_index()
s.dt.month.value_counts().sort_index()
s.dt.dayofweek.value_counts()
s.dt.hour.value_counts()
s.is_monotonic_increasing                     # sorted?
s.diff().value_counts()                       # granularity / regularity
s.diff().max()                                # largest gap

# Sanity
(s > pd.Timestamp.now()).sum()                # future dates
(s < pd.Timestamp("1900-01-01")).sum()        # impossible
s.dt.date.duplicated().sum()

# Gaps in a supposedly complete series
full = pd.date_range(s.min(), s.max(), freq="D")
missing_days = full.difference(s.dt.normalize().unique())
```

📊 line plot over time, seasonal decomposition, gap chart.

---

### 9. Univariate — text 🔍

```python
s = df["text"]
s.str.len().describe()
s.str.split().str.len().describe()             # word count
s.str.isupper().sum(); s.str.islower().sum()
s.str.contains(r"\d").sum()
s.str.strip().ne(s).sum()                      # stray whitespace
s.str.match(r"^[\w.]+@[\w.]+$").sum()          # emails
s.str.match(r"^https?://").sum()               # URLs
s.str.extract(r"(\d{4}-\d{2}-\d{2})")          # embedded dates
s.value_counts().head(20)
```

---

### 10. Outliers 🔍 ✏️

```python
s = df["x"]

# IQR
q1, q3 = s.quantile([.25,.75]); iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = s[(s < lo) | (s > hi)]

# Z-score
z = np.abs(stats.zscore(s.dropna()))
outliers = s.dropna()[z > 3]

# Modified z-score (robust — use with skewed data)
med = s.median(); mad = stats.median_abs_deviation(s.dropna())
mz = 0.6745 * (s - med) / mad
outliers = s[np.abs(mz) > 3.5]

# Percentile
lo, hi = s.quantile([.01,.99])

# Multivariate
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(df[numeric])
flags = LocalOutlierFactor(n_neighbors=20).fit_predict(df[numeric])

# Mahalanobis distance
```

Handling ✏️:
```python
df = df[(s >= lo) & (s <= hi)]                # remove
df["x"] = s.clip(lo, hi)                      # winsorise
df["x"] = np.log1p(s)                         # transform
df["x_outlier"] = ((s < lo) | (s > hi)).astype(int)   # flag
```

**Always ask whether an outlier is an error or a real extreme value.**

---

### 11. Bivariate — numeric ↔ numeric 🔍

```python
df[numeric].corr()                             # Pearson (linear)
df[numeric].corr(method="spearman")            # monotonic, rank-based
df[numeric].corr(method="kendall")
stats.pearsonr(df.x, df.y)                     # with p-value
stats.spearmanr(df.x, df.y)

# Strong pairs only — never dump the whole matrix
c = df[numeric].corr().abs()
pairs = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
pairs[pairs > 0.7].sort_values(ascending=False)

df.x.cov(df.y)
```

📊 scatter, hexbin, pairplot, correlation heatmap, regression plot.

---

### 12. Bivariate — categorical ↔ numeric 🔍

```python
df.groupby("c")["x"].agg(["count","mean","median","std","min","max"])
df.groupby("c")["x"].describe()

# Significance
groups = [g["x"].dropna() for _, g in df.groupby("c")]
stats.f_oneway(*groups)                        # ANOVA
stats.kruskal(*groups)                         # non-parametric
stats.ttest_ind(a, b, equal_var=False)         # two groups
stats.mannwhitneyu(a, b)

# Effect size (matters more than the p-value)
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var() + (nb-1)*b.var()) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled

# Correlation ratio (eta squared)
```

📊 grouped boxplot, violin, bar with error bars, strip/swarm.

---

### 13. Bivariate — categorical ↔ categorical 🔍

```python
pd.crosstab(df.a, df.b)
pd.crosstab(df.a, df.b, normalize="index")
pd.crosstab(df.a, df.b, margins=True)

chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df.a, df.b))

# Cramér's V — effect size for categorical association
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
```

📊 stacked bar, grouped bar, mosaic plot, heatmap of the crosstab.

---

### 14. Multivariate 🔍

```python
# Collinearity — VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[numeric].dropna()
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
# VIF > 10 → serious multicollinearity

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
p = PCA().fit(Xs)
p.explained_variance_ratio_.cumsum()

# Clustering structure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

📊 pairplot, PCA scatter, parallel coordinates, andrews curves, clustermap.

---

### 15. Target analysis 🔍

```python
y = df["target"]

# Classification
y.value_counts()
y.value_counts(normalize=True)
imbalance = y.value_counts(normalize=True).max()
if imbalance > 0.9:
    print(f"severe imbalance: {imbalance:.1%} majority class")

# Regression
y.describe(); y.skew()
np.log1p(y).skew()                             # would a transform help?

# Feature → target strength
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
mi = mutual_info_classif(X, y)

# LEAKAGE — a feature that is suspiciously predictive
for c in numeric:
    r = abs(df[c].corr(y))
    if r > 0.95:
        print(f"{c}: r={r:.3f} with target — probable leakage")

# Categorical leakage: does one category map perfectly to one class?
for c in cats:
    purity = df.groupby(c)["target"].nunique()
    if (purity == 1).mean() > 0.9:
        print(f"{c}: categories map 1:1 to target — leakage")
```

---

### 16. Distributions and transforms ✏️

```python
np.log1p(s)                                    # right skew, zeros ok
np.sqrt(s)                                     # mild right skew
np.square(s)                                   # left skew
stats.boxcox(s[s>0])                           # positive only
stats.yeojohnson(s)                            # handles zero/negative
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
PowerTransformer(method="yeo-johnson").fit_transform(X)
QuantileTransformer(output_distribution="normal").fit_transform(X)
```

Check improvement: `s.skew()` before vs after.

---

### 17. Visualisation catalogue 📊

| Purpose | Plot |
|---|---|
| Numeric distribution | histogram, KDE, boxplot, violin, ECDF |
| Normality | Q-Q plot |
| Categorical frequency | bar, count plot |
| Numeric ↔ numeric | scatter, hexbin, regplot |
| Cat ↔ numeric | grouped box, violin, bar+CI |
| Cat ↔ cat | stacked bar, mosaic, crosstab heatmap |
| Correlation | heatmap, clustermap |
| All pairs | pairplot, scatter matrix |
| Missingness | `sns.heatmap(df.isna())`, missingno matrix/bar/dendrogram |
| Time | line, seasonal decomposition, lag plot, autocorrelation |
| Outliers | boxplot, scatter with flags |
| High dimensions | PCA scatter, t-SNE, UMAP, parallel coordinates |

```python
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))
for ax, c in zip(axes.flat, numeric):
    df[c].hist(ax=ax, bins=30); ax.set_title(c)
plt.tight_layout(); plt.savefig("dist.png", dpi=120, bbox_inches="tight")
```

---

### 18. Cleaning operations ✏️

```python
# Column hygiene
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.rename(columns={"old":"new"})
df = df.drop(columns=["unused"])

# String hygiene
df["c"] = df["c"].str.strip().str.lower()
df["c"] = df["c"].str.replace(r"\s+", " ", regex=True)
df["c"] = df["c"].replace({"u.s.a.":"usa", "united states":"usa"})
df["c"] = df["c"].str.normalize("NFKD")

# Value repair
df["age"] = df["age"].where(df["age"].between(0, 120))
df = df[df["price"] > 0]
df["pct"] = df["pct"].clip(0, 100)

# Structural
df = df.reset_index(drop=True)
df = df.sort_values(["date","id"])
df = df.set_index("date")
```

---

### 19. Feature engineering ✏️

#### Encoding

```python
pd.get_dummies(df, columns=["c"], drop_first=True)          # one-hot
df["c"].map({"low":0,"med":1,"high":2})                     # ordinal
df["c"].map(df["c"].value_counts(normalize=True))           # frequency
df.groupby("c")["target"].transform("mean")                 # target (leaky — use CV folds)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import category_encoders as ce                              # binary, hashing, WOE
```

#### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
StandardScaler()      # mean 0, sd 1
MinMaxScaler()        # [0,1]
RobustScaler()        # median/IQR — outlier resistant
```

#### Binning

```python
pd.cut(df.age, bins=[0,18,35,60,120], labels=["child","young","adult","senior"])
pd.qcut(df.income, q=4, labels=["Q1","Q2","Q3","Q4"])       # equal frequency
```

#### Dates

```python
d = df["date"].dt
df["year"], df["month"], df["day"] = d.year, d.month, d.day
df["dow"], df["week"], df["quarter"] = d.dayofweek, d.isocalendar().week, d.quarter
df["is_weekend"] = d.dayofweek >= 5
df["days_since"] = (pd.Timestamp.now() - df["date"]).dt.days
df["month_sin"] = np.sin(2*np.pi*d.month/12)                # cyclical
df["month_cos"] = np.cos(2*np.pi*d.month/12)
```

#### Derived

```python
df["ratio"] = df.a / df.b.replace(0, np.nan)
df["total"] = df[["a","b","c"]].sum(axis=1)
df["a_x_b"] = df.a * df.b                                   # interaction
from sklearn.preprocessing import PolynomialFeatures

# Time series
df["lag_1"] = df.x.shift(1)
df["roll_7"] = df.x.rolling(7).mean()
df["expanding"] = df.x.expanding().mean()
df["pct_change"] = df.x.pct_change()
df["cumsum"] = df.x.cumsum()

# Group aggregates
df["mean_by_g"] = df.groupby("g")["x"].transform("mean")
df["rank_in_g"] = df.groupby("g")["x"].rank()
```

---

### 20. Reshaping ✏️

```python
df.pivot_table(index="a", columns="b", values="x", aggfunc="mean")
df.melt(id_vars=["id"], value_vars=["x","y"])
pd.merge(a, b, on="id", how="left", indicator=True)      # check merge quality
pd.concat([a, b], axis=0, ignore_index=True)
df.groupby("g").agg({"x":["mean","sum"], "y":"count"})
df.stack(); df.unstack()
df.transpose()
df.explode("list_col")
```

Post-merge validation:
```python
merged["_merge"].value_counts()      # left_only rows = failed joins
assert len(merged) == len(a)         # no unexpected fan-out
```

---

### 21. Time series specifics 🔍

```python
df = df.set_index("date").sort_index()
df.resample("M").mean()
df.asfreq("D")                                  # expose gaps
df.x.rolling(30).mean()
from statsmodels.tsa.seasonal import seasonal_decompose, STL
seasonal_decompose(df.x, model="additive", period=12).plot()
from statsmodels.tsa.stattools import adfuller, acf, pacf
adfuller(df.x.dropna())                         # stationarity
pd.plotting.autocorrelation_plot(df.x)
```

---

### 22. Sanity checks 🔍

```python
assert df["id"].is_unique
assert df["age"].between(0,120).all()
assert (df["end"] >= df["start"]).all()
assert df.groupby("id").size().max() == 1
assert df["pct"].between(0,100).all()
assert abs(df["parts"].sum() - df["total"].iloc[0]) < 1e-6

# Business rules
assert (df["discount"] <= df["price"]).all()
assert df["quantity"].ge(0).all()
```

---

### 23. Reporting and export

```python
# Automated profiling
from ydata_profiling import ProfileReport
ProfileReport(df, explorative=True).to_file("report.html")
import sweetviz; sweetviz.analyze(df).show_html("sv.html")
from dtale import show; show(df)
import autoviz

# Export
df.to_csv("clean.csv", index=False)
df.to_parquet("clean.parquet", compression="snappy")
df.to_excel("clean.xlsx", sheet_name="data", index=False)
df.to_json("clean.json", orient="records", lines=True)
```

---


---

### A.25 Mapping — reference section to tool

| Appendix section | Tool(s) |
|---|---|
| A.1 Loading, A.2 First look | `load_dataset` |
| A.1 (database sources) | `connect_database`, `explore_schema`, `query` |
| A.2, A.3, A.4.1, A.5 | `profile` |
| A.6, A.7, A.8, A.9, A.10 | `analyze_column` |
| A.3.2, A.4.1–4.2, A.5, A.7, A.10 | `find_issues` |
| A.11, A.12, A.13, A.14 | `check_relationships` (incl. group comparison) |
| A.15 Target analysis | `analyze_target` |
| A.17 Visualisation | `plot` |
| A.22 Sanity checks | `validate_rules` |
| A.4.3, A.5, A.10, A.18 | `clean_data` |
| A.16, A.19 | `transform_data` |
| A.20, A.21 | `reshape_data` |
| A.23 Reporting | `generate` |
| A.23 Export | `export` |

### A.26 Coverage notes

**Fully covered by v1.** Sections A.1–A.20, A.22, A.23.

**Deferred to a later phase.** Section A.21 (time series) — resampling,
seasonal decomposition, stationarity testing and lag features are a separate
discipline and are not in the v1 tool set. `transform_data` does provide
`lag` and `rolling` for ordered data, which covers the common feature-
engineering case without the full time-series apparatus.

**Deliberately out of scope.** The automated profiling libraries in A.23
(`ydata_profiling`, `sweetviz`, `dtale`) are alternatives to this server, not
dependencies of it. They produce large HTML artefacts for human reading;
`generate(kind="report")` produces a token-budgeted narrative for a model to reason
over, and writes the detail to a file.
