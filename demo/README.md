# GPU-OLTP Live Demo

This is the short live demo for a Postgres-looking presentation.

It shows:

- real `psql`
- the shared GPU service worker responding
- equality predicate pushdown
- batched multi-key lookup
- batched writes at commit
- read-your-writes and `ROLLBACK`
- primary-key validation with validate-before-mutate atomic commit

It intentionally avoids the known slow path:

- `k = ANY(ARRAY(SELECT ...))`, which falls back to a snapshot scan

## Run The Demo

From the repo root:

```bash
bash demo/run_demo.sh
```

By default this keeps the output clean and does not show timings.
The runner prints one command at a time with a short delay between statements.

To show timings after each query:

```bash
bash demo/run_demo.sh --time
```

To change the pacing:

```bash
DEMO_DELAY=1.5 bash demo/run_demo.sh
DEMO_DELAY=0.2 bash demo/run_demo.sh
```

Set your sudo password (required — the scripts no longer hardcode a default):

```bash
PGSUDO_PW='your_password' bash demo/run_demo.sh
```

The runner restarts PostgreSQL 14 before the demo. That is intentional: the GPU service worker owns an in-memory GPU index, so restarting gives the demo a clean index.

The runner prints one SQL statement at a time with a short delay, so the output is readable without requiring manual prompts.

## The Queries To Point Out

Point lookup:

```sql
SELECT v FROM kv_rgi WHERE k = 50000;
```

This is the cleanest predicate-pushdown example. The FDW extracts `k = 50000` and sends one key to the GPU index.

Multi-key lookup:

```sql
PREPARE demo_multiget(bigint[]) AS
  SELECT count(*) AS hits, sum(v) AS value_sum
  FROM kv_rgi
  WHERE k = ANY($1);

EXECUTE demo_multiget('{1,100,500,50000,99999}');
```

This is the application-shaped fast path: one SQL statement becomes one batched GPU find.

Bulk write:

```sql
UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;
```

Postgres identifies the rows. The FDW buffers the writes. Commit batch-applies them to the GPU index.

Rollback:

```sql
BEGIN;
UPDATE kv_rgi SET v = 424242 WHERE k = 42;
SELECT v FROM kv_rgi WHERE k = 42;
ROLLBACK;
SELECT v FROM kv_rgi WHERE k = 42;
```

This shows read-your-writes inside the transaction and then proves the GPU index was unchanged after rollback.

Atomic commit:

```sql
BEGIN;
DELETE FROM kv_rgi WHERE k = 2;
INSERT INTO kv_rgi VALUES (7, 70);
COMMIT;
```

The commit fails because key `7` already exists. Then the demo checks that key `2` still exists. This is the validate-before-mutate story.

## Speaker Line

Use this when the demo starts:

> This is unmodified Postgres from the user's point of view. The table is a foreign table, but the SQL is normal: inserts, selects, updates, deletes, begin, commit, rollback. The difference is that the storage path behind this table is a shared GPU service worker and an RGI index on the GPU.
