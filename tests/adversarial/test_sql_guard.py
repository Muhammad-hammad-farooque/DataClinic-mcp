"""Adversarial tests for the SQL statement guard (spec 6.3, layer 2).

Phase 3's exit criterion is this suite at 100%: every statement below must
be rejected in every dialect, and every legitimate query must still pass.
Rejection must come from the guard -- a ``StatementRejectedError`` -- never
an unhandled exception.
"""

from __future__ import annotations

import pytest

from eda_mcp.db.guard import check
from eda_mcp.errors import ErrorCode, StatementRejectedError

DIALECTS = ["postgres", "sqlite", "duckdb", "mysql", "tsql"]

ATTACKS = {
    # stacked statements
    "stacked drop": "SELECT 1; DROP TABLE t",
    "stacked delete no space": "SELECT 1;DELETE FROM t",
    "stacked select": "SELECT 1; SELECT 2",
    "trailing statement after comment": "SELECT 1 --\n; DROP TABLE t",
    "leading comment": "/* report */ DROP TABLE t",
    # plain writes and DDL
    "insert": "INSERT INTO t VALUES (1)",
    "insert select": "INSERT INTO t SELECT * FROM s",
    "update": "UPDATE t SET a = 1",
    "delete": "DELETE FROM t",
    "merge": "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
    "drop": "DROP TABLE t",
    "create": "CREATE TABLE t (a INT)",
    "create as select": "CREATE TABLE t2 AS SELECT * FROM t",
    "create view": "CREATE VIEW v AS SELECT 1",
    "alter": "ALTER TABLE t ADD COLUMN b INT",
    "truncate": "TRUNCATE TABLE t",
    "grant": "GRANT ALL ON t TO public",
    "revoke": "REVOKE ALL ON t FROM public",
    # writes disguised as queries
    "cte delete": "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
    "cte insert": "WITH i AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM i",
    "cte update": "WITH u AS (UPDATE t SET a = 1 RETURNING *) SELECT 1",
    "select into": "SELECT * INTO t2 FROM t",
    "for update": "SELECT * FROM t FOR UPDATE",
    "for share": "SELECT * FROM t FOR SHARE",
    # session and transaction control
    "set": "SET search_path TO evil",
    "set role": "SET ROLE admin",
    "begin": "BEGIN",
    "commit": "COMMIT",
    "rollback": "ROLLBACK",
    "read write transaction": "START TRANSACTION READ WRITE",
    "set transaction": "SET TRANSACTION READ WRITE",
    "pragma": "PRAGMA writable_schema = 1",
    "attach": "ATTACH DATABASE 'other.db' AS o",
    "detach": "DETACH DATABASE o",
    "use": "USE other",
    # maintenance, procedures and dynamic SQL
    "vacuum": "VACUUM",
    "analyze": "ANALYZE t",
    "reindex": "REINDEX TABLE t",
    "call": "CALL do_something()",
    "exec cmdshell": "EXEC xp_cmdshell 'dir'",
    "execute": "EXECUTE stmt",
    "prepare": "PREPARE stmt AS DELETE FROM t",
    "do block": "DO $$ BEGIN DELETE FROM t; END $$",
    "lock": "LOCK TABLE t",
    "declare": "DECLARE c CURSOR FOR SELECT 1",
    "listen": "LISTEN channel",
    "notify": "NOTIFY channel",
    "checkpoint": "CHECKPOINT",
    "kill": "KILL 1",
    # files, extensions, networks
    "copy to": "COPY t TO '/tmp/out.csv'",
    "copy from program": "COPY t FROM PROGRAM 'rm -rf /'",
    "load data": "LOAD DATA INFILE '/etc/passwd' INTO TABLE t",
    "install": "INSTALL httpfs",
    "load extension statement": "LOAD httpfs",
    # dangerous functions, top level and nested
    "pg_read_file": "SELECT pg_read_file('/etc/passwd')",
    "pg_read_file upper": "SELECT PG_READ_FILE('/etc/passwd')",
    "pg_read_file qualified": "SELECT pg_catalog.pg_read_file('/etc/passwd')",
    "pg_ls_dir": "SELECT * FROM pg_ls_dir('.')",
    "lo_import": "SELECT lo_import('/etc/passwd')",
    "pg_sleep": "SELECT pg_sleep(100)",
    "sleep in where": "SELECT * FROM t WHERE a = (SELECT pg_sleep(10))",
    "sleep in order by": "SELECT * FROM t ORDER BY pg_sleep(1)",
    "sleep in union": "SELECT 1 UNION SELECT pg_sleep(1)",
    "sleep in cte": "WITH s AS (SELECT pg_sleep(1)) SELECT * FROM s",
    "mysql sleep": "SELECT SLEEP(100)",
    "benchmark": "SELECT BENCHMARK(100000000, MD5('x'))",
    "load_file": "SELECT LOAD_FILE('/etc/passwd')",
    "load_extension": "SELECT load_extension('evil.so')",
    "readfile": "SELECT readfile('/etc/passwd')",
    "writefile": "SELECT writefile('/tmp/x', 'data')",
    "duckdb read_csv": "SELECT * FROM read_csv('/etc/passwd')",
    "duckdb read_csv_auto": "SELECT * FROM read_csv_auto('/etc/passwd')",
    "duckdb read_parquet": "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
    "duckdb glob": "SELECT * FROM glob('/home/*')",
    "duckdb query string": "SELECT * FROM query('DELETE FROM t')",
    "duckdb getenv": "SELECT getenv('AWS_SECRET_ACCESS_KEY')",
    "query_to_xml": "SELECT query_to_xml('DELETE FROM t', true, false, '')",
    "dblink": "SELECT * FROM dblink('host=evil', 'SELECT 1') AS x(a int)",
    "dblink_exec": "SELECT dblink_exec('host=evil', 'DROP TABLE t')",
    "nextval": "SELECT nextval('seq')",
    "setval": "SELECT setval('seq', 1)",
    "set_config": "SELECT set_config('default_transaction_read_only', 'off', false)",
    "terminate": "SELECT pg_terminate_backend(1)",
    "advisory lock": "SELECT pg_advisory_lock(1)",
    "openrowset": "SELECT * FROM OPENROWSET('SQLNCLI', 'server=evil', 'SELECT 1')",
    # degenerate input
    "empty": "",
    "whitespace": "   ",
    "semicolon only": ";",
    "comment only": "-- nothing here",
    "garbage": "SELEC * FRM t",
    "too long": "SELECT " + ", ".join(["1"] * 10_000),
    "deep nesting": "SELECT " + "(" * 5_000 + "1" + ")" * 5_000,
}

LEGITIMATE = [
    "SELECT 1",
    "SELECT a, b FROM t WHERE a > 1 AND b IN ('x', 'y') ORDER BY a DESC LIMIT 10",
    "SELECT country, COUNT(*) AS n, AVG(price) FROM orders GROUP BY country HAVING COUNT(*) > 5",
    "WITH recent AS (SELECT * FROM orders WHERE amount > 0) SELECT COUNT(*) FROM recent",
    "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id",
    "SELECT * FROM t LEFT JOIN s ON t.id = s.id WHERE s.id IS NULL",
    "SELECT a, ROW_NUMBER() OVER (PARTITION BY b ORDER BY c) FROM t",
    "SELECT CASE WHEN a > 1 THEN 'big' ELSE 'small' END FROM t",
    "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.id = t.id)",
    "SELECT a FROM t UNION ALL SELECT a FROM s",
    "SELECT a FROM t EXCEPT SELECT a FROM s",
    "SELECT UPPER(name), LENGTH(name), COALESCE(a, 0), ROUND(price, 2) FROM t",
    "SELECT * FROM t WHERE name LIKE 'A%'",
    "SELECT MIN(a), MAX(a), SUM(a) FROM t",
    "SELECT COUNT(DISTINCT a) FROM t",
    "SELECT a /* inline note */ FROM t -- trailing note",
    "SELECT 'DROP TABLE t' AS text_that_looks_dangerous",
    "SELECT * FROM t WHERE note = 'x; DELETE FROM t'",
]


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("name", list(ATTACKS))
def test_attack_is_rejected(name: str, dialect: str) -> None:
    with pytest.raises(StatementRejectedError) as caught:
        check(ATTACKS[name], dialect)
    assert caught.value.code is ErrorCode.STATEMENT_REJECTED


@pytest.mark.parametrize("dialect", ["postgres", "sqlite", "duckdb", "mysql"])
@pytest.mark.parametrize("sql", LEGITIMATE)
def test_legitimate_query_passes(sql: str, dialect: str) -> None:
    checked = check(sql, dialect)
    assert checked.sql


@pytest.mark.parametrize("dialect", DIALECTS)
def test_comment_wrapped_statement_never_runs(dialect: str) -> None:
    # Whether "/*/ ... /*/" is one comment depends on the dialect: SQLite and
    # MySQL close it at the first "*/", Postgres nests it and never closes it.
    # Either reading is safe as long as the DROP is rejected or not executed.
    try:
        checked = check("SELECT 1 /*/ ; DROP TABLE t; /*/", dialect)
    except StatementRejectedError:
        return
    assert "DROP" not in checked.sql.upper()


def test_what_runs_is_regenerated_without_comments() -> None:
    checked = check("SELECT a /* ; DROP TABLE t */ FROM t -- ; DELETE FROM t", "postgres")
    assert "DROP" not in checked.sql and "DELETE" not in checked.sql
    assert "--" not in checked.sql and "/*" not in checked.sql


def test_string_literals_survive_regeneration() -> None:
    checked = check("SELECT 'x; DELETE FROM t' AS note", "postgres")
    assert "'x; DELETE FROM t'" in checked.sql  # data, not a statement


def test_row_cap_wraps_every_query() -> None:
    assert check("SELECT * FROM t", "postgres", row_cap=50).sql.endswith("LIMIT 50")
    assert "TOP 50" in check("SELECT * FROM t", "tsql", row_cap=50).sql
    # a caller's own larger LIMIT cannot lift the cap
    capped = check("SELECT * FROM t LIMIT 1000000", "postgres", row_cap=50).sql
    assert capped.endswith("LIMIT 50")


def test_unknown_dialect_is_refused() -> None:
    from eda_mcp.db.guard import sqlglot_dialect

    with pytest.raises(StatementRejectedError):
        sqlglot_dialect("oracle")
