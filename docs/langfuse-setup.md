# Langfuse (self-hosted v4): setup walkthrough and reference

Phase E1 of the [roadmap](ROADMAP.MD). This is a record of how the local Langfuse stack was set up on 2026-10-13: every command, what it does, and what actually happened. Use it to redo the setup, to operate the stack day to day, or to debug it.

**What Langfuse is for in this project:**
- versioned prompts (E3–E4);
- one trace per chat request (Phase G);
- the golden eval set as a dataset, with eval runs linked to it (Phase H).

Our SQLite `traces` table stays the slim audit record. Langfuse is where you *look* at traces.

---

## 1. What runs, and how the pieces talk

```
                 your machine (127.0.0.1 only)
                  │                     │
            :3000 │               :6333 │ :6334
┌─────────────────┼─────────────────────┼──────────── compose network "dfsa-rag_default" ┐
│                 ▼                     ▼                                                │
│          langfuse-web             qdrant  (unchanged since Phase A)                    │
│        UI + public API                                                                 │
│         │   │    │   └──────────────► postgres:5432   users, orgs, projects, API keys, │
│         │   │    │                                    prompts, datasets                │
│         │   │    └──► minio:9000 ◄──┐ raw events (S3 bucket "langfuse")                │
│         │   └──► redis:6379 ◄───────┤ job queue (BullMQ)                               │
│         └──► clickhouse:8123 ◄──────┤ traces, observations, scores (column store)      │
│                                     │                                                  │
│                              langfuse-worker                                           │
│                    reads the queue, writes the events into ClickHouse                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**How a trace flows (from Phase G on):**
1. Our SDK sends the trace to `langfuse-web`.
2. web stores the raw event in **MinIO** and enqueues a job in **Redis**.
3. **langfuse-worker** picks up the job and writes it to **ClickHouse**.
4. The UI reads traces from ClickHouse, and everything else (users, projects, prompts, datasets) from **Postgres**.

Ingestion keeps working even if ClickHouse is briefly slow, because events wait in MinIO and Redis.

| Service | Image (pinned) | Holds | Host port |
|---|---|---|---|
| langfuse-web | `langfuse/langfuse:4.57.0` | UI and public API (Next.js) | `127.0.0.1:3000` |
| langfuse-worker | `langfuse/langfuse-worker:4.57.0` | Background jobs | none |
| postgres | `postgres:17.11` | Transactional data | none |
| clickhouse | `clickhouse/clickhouse-server:26.4.5.143` | Traces and scores | none |
| redis | `redis:7.4.11` | Queue | none |
| minio | `cgr.dev/chainguard/minio@sha256:dc455c…` | Raw event blobs | none |
| qdrant | `qdrant/qdrant:v1.19.0` | Our vector index (Phase A) | `127.0.0.1:6333-6334` |

**Why v4, not v3:**
- v4 has been GA since 2026-07-29, and v3 only gets security patches until 2027-01.
- It runs on the same components.
- A fresh install has nothing to migrate. The v4 data migrations in the worker logs all found 0 partitions.

---

## 2. Files involved

| File | What it does |
|---|---|
| `infra/docker-compose.yml` | The 6 Langfuse services next to `qdrant`. A YAML anchor (`x-langfuse-env`) holds the environment shared by web and worker. Every secret is `${VAR:?…}`, so compose refuses to start if one is missing |
| `scripts/init_langfuse_env.py` | Writes the missing `LANGFUSE_*` secrets into `.env`. Never prints values; never overwrites a set value |
| `tests/test_init_langfuse_env.py` | Tests that it fills every name, never overwrites, and fills blank placeholders in place |
| `.env.example` | The `LANGFUSE_*` names, with a comment on what each protects |
| `.env` | The real values (gitignored) |

**Decisions in the compose file that differ from Langfuse's upstream file:**
- **Only `langfuse-web` is published, and only on `127.0.0.1`.** Upstream binds web and MinIO on every network interface (`3000:3000`, `9090:9000`) and publishes the data stores on localhost. Ours reaches the data stores over the compose network only.
  - Side effect: media uploads (images, attachments) from the SDK won't work. We trace text only, so this is accepted.
- **`AUTH_DISABLE_SIGNUP=true`**: only the init user can log in.
- **`TELEMETRY_ENABLED=false`**, as for Qdrant.
- **The Redis healthcheck is a real check.** Upstream's `redis-cli ping` exits 0 even on an auth error. Ours authenticates (`REDISCLI_AUTH`) and greps for `PONG`.
- **Every image is pinned.** MinIO is pinned by digest, because Chainguard's free tier only publishes a moving `latest` tag.
- **`NEXTAUTH_URL=http://127.0.0.1:3000`.** Open the UI at `127.0.0.1`, not `localhost`: login cookies are tied to the hostname.

---

## 3. The setup, step by step

Every compose command is run from the repo root:

```
docker compose -f infra/docker-compose.yml --env-file .env <subcommand>
```

- `-f` names the compose file.
- `--env-file .env` points compose at the root `.env`. By default it would look next to the compose file, in `infra/`.

### Step 0: is Docker ready, and how much RAM does it have?
```
docker version
docker info --format "{{.MemTotal}}"
```
- **What it does:** Docker Desktop runs containers inside a WSL2 Linux VM, and they all share that VM's memory.
  - `docker version` shows that the CLI (Client) can reach the engine (Server).
  - `MemTotal` is the VM's memory ceiling, in bytes.
- **Result:**
  - Client and Server 29.8.2 (Docker Desktop 4.94.0).
  - `8171765760` bytes = **7.6 GiB** (WSL2 gives half the host's RAM by default). That's enough; the stack needs about 4–6 GB at peak.
- **The ceiling is not a reservation.** The VM only uses what its containers use.

### Step 1: generate the secrets
```
uv run python -m scripts.init_langfuse_env --email <your email>
```
- **What it does:** fills the missing or blank `LANGFUSE_*` names in `.env` with random values from Python's `secrets` module, and prints only the names.
- **Result:** 12 names added.

| Name | What it protects |
|---|---|
| `LANGFUSE_POSTGRES_PASSWORD` | Postgres login (web/worker ↔ Postgres) |
| `LANGFUSE_CLICKHOUSE_PASSWORD` | ClickHouse login |
| `LANGFUSE_REDIS_AUTH` | Redis password |
| `LANGFUSE_MINIO_PASSWORD` | MinIO (S3) secret |
| `LANGFUSE_NEXTAUTH_SECRET` | Signs the browser login session cookie |
| `LANGFUSE_SALT` | Used to hash API keys. **Never change it once data exists**, or every key stops working |
| `LANGFUSE_ENCRYPTION_KEY` | 64 hex characters; encrypts secrets Langfuse stores. **Never change it once data exists** |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | The project API key pair. The server creates the project with exactly these keys |
| `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD` | Your admin login for the UI |
| `LANGFUSE_HOST` | Where the SDK sends data: `http://127.0.0.1:3000` |

- **Why it never overwrites:**
  - rotating `SALT` or `ENCRYPTION_KEY` after data exists silently breaks Langfuse;
  - the database passwords are only applied when a volume is first created, so changing them later wouldn't change the stored password.

### Step 2: check the compose file
```
docker compose ... config --quiet
docker compose ... config --services
```
- **What it does:** parses the YAML, merges the anchor, substitutes every `${VAR}` and validates the schema, all without starting anything.
  - `--quiet` prints nothing when all is well.
  - A missing secret stops it with the `:?` message.
- **Result:** silent; 7 services listed.
- ⚠️ **Never run `config` without `--quiet` where others can see the output.** It prints the resolved file with every password.

### Step 3: pull the images
```
docker compose ... pull
```
- **What it does:** downloads the images (read-only filesystems) layer by layer. Layers already present are skipped.
  - A *tag* like `postgres:17.11` could in theory be moved by its publisher.
  - A *digest* like `@sha256:…` is the hash of the content, so it never changes.
- **Result:** all images pulled. `qdrant` was already present.

### Step 4: start the four data stores
```
docker compose ... up -d postgres clickhouse redis minio
docker compose ... ps
```
- **What it does:**
  - creates the 5 named volumes and the containers on the existing network, where containers find each other by service name;
  - runs each store's first-start setup (Postgres cluster and password, ClickHouse user, the MinIO bucket directory).
- **The healthchecks:**

| Service | Check |
|---|---|
| postgres | `pg_isready -U postgres` |
| clickhouse | HTTP `/ping` |
| redis | Authenticated `PING` → `PONG` |
| minio | `mc ready local` |

- **Result:** all four `(healthy)`, with no host ports.

### Step 5: start the worker and the web app
```
docker compose ... up -d langfuse-worker langfuse-web
docker compose ... logs --tail 80 langfuse-web
docker compose ... logs --tail 80 langfuse-worker
```
- **What it does:**
  - `depends_on: service_healthy` holds web and worker until the four stores are healthy.
  - web then applies the **Postgres** migrations (Prisma; same idea as our Alembic, recorded in `_prisma_migrations`) and the **ClickHouse** migrations.
  - web runs the **headless setup** (`LANGFUSE_INIT_*`), which creates the org and project `dfsa-rag` ("DFSA RAG"), the API key pair from `.env`, and the admin user.
  - Next.js starts.
- **Result (web):**
  - `All migrations have been successfully applied`;
  - ClickHouse migrations `1/u traces` … `51/u add_topics`, including the v4 `create_events_full` and `create_events_core`, with **no permission errors**;
  - `▲ Next.js 16.4.0 … ✓ Ready`.
  - Langfuse detected ClickHouse 26.4.5.143 and applied one compatibility setting. Its workarounds for two known ClickHouse bugs were not needed on this version.
- **Result (worker):**
  - queue executors started;
  - the v3 → v4 background migrations found 0 partitions and finished instantly (a fresh install).
- **One harmless line:** `events_full table does not exist. Retrying in 10s`. This is a start-up race: the worker looked before web's migration created the table, and 10 s later it reported `All prerequisites validated successfully`.

### Step 6: check it from the host
```
curl.exe -s http://127.0.0.1:3000/api/public/health
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:3000/api/public/projects
```
- Use `curl.exe`, not `curl`: in Windows PowerShell, `curl` can be an alias for `Invoke-WebRequest`.
- **Result:**
  - `{"status":"OK","version":"4.57.0"}`: the running version is exactly the pinned tag.
  - `401`: the API refuses calls without a key. It's the same test as `check_qdrant`'s "Without key".

### Step 7: log in to the UI
Open http://127.0.0.1:3000 and log in with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD` from `.env`.

**Result:**
- The org and project are **DFSA RAG**.
- Settings → API Keys shows **the same `pk-lf-…` as `.env`**. The secret key is masked, because it's stored only as a salted hash.
- Sign-up is **disabled**.

**Where things will show up later:**
- **Tracing**: Phase G.
- **Prompts**: E3.
- **Datasets**: H2.

### Step 8: what's exposed, and what it costs
```
docker compose ... ps
docker stats --no-stream
```
- **Reading the PORTS column:**
  - `127.0.0.1:3000->3000/tcp` means *host address:port → container port*, reachable from this laptop only.
  - A bare `5432/tcp` (no arrow) is reachable only on the compose network.
- **Result:** only `127.0.0.1:3000` (Langfuse) and `127.0.0.1:6333-6334` (Qdrant) are published, and nothing is on `0.0.0.0`.
- **Memory, idle, measured:**

| Container | Memory (idle) |
|---|---|
| langfuse-web | 972 MiB |
| langfuse-worker | 798 MiB |
| clickhouse | 604 MiB |
| qdrant | 120 MiB |
| postgres | 93 MiB |
| minio | 66 MiB |
| redis | 7 MiB |
| **Total** | **≈ 2.6 GiB** (Langfuse alone ≈ 2.5 GiB) |

- **Two things noticed:**
  - **ClickHouse showed 797 PIDS.** These are mostly its pre-started thread pools, which is normal.
  - **ClickHouse wrote 503 MB to disk in 30 idle minutes.** These are its own system log tables (`query_log`, `metric_log`, `trace_log`, …). It's harmless, but the volume grows over time (see Open items).
  - Redis received about 30 MB while idle: BullMQ polling its queues.

### Step 9: tear down and recreate, and prove the data survives
```
docker compose ... down
docker volume ls --filter name=dfsa-rag
docker compose ... up -d
docker compose ... logs --tail 25 langfuse-web
```
- **What it does:**
  - `down` removes **all 7 containers**, Qdrant included, plus the network. **Volumes are kept**: the containers are disposable, and the volumes hold the state.
  - `up -d` recreates everything, with `depends_on` ordering the start-up.
  - On the second start, migrations report nothing pending, and the headless setup creates nothing new.
- **Result:**
  - the 6 volumes were listed while no containers existed;
  - after `up -d`, there was still one project, one API key and the same `pk-lf-…`.

### Step 10: Qdrant untouched
```
uv run python -m scripts.check_qdrant
```
- **What it does:** the Phase C smoke test, run after Qdrant's container was recreated in Step 9.
- **Expected:** Qdrant 1.19.0; `HTTP 401` without the key; `dfsa_rules__text-embedding-3-small-1536` with **3,397 points**, status green.

---

## 4. Day-to-day operation

```
# start everything (Qdrant + Langfuse)
docker compose -f infra/docker-compose.yml --env-file .env up -d

# stop everything, keep containers and data (resume with `start`)
docker compose -f infra/docker-compose.yml --env-file .env stop
docker compose -f infra/docker-compose.yml --env-file .env start

# status, logs, resources
docker compose -f infra/docker-compose.yml --env-file .env ps
docker compose -f infra/docker-compose.yml --env-file .env logs --tail 50 langfuse-web
docker stats --no-stream

# health
curl.exe -s http://127.0.0.1:3000/api/public/health
```

**Memory:**
- Stopped containers use no memory.
- The WSL2 VM may hold freed memory as cache for a while. To return it to Windows, quit Docker Desktop, or let Resource Saver pause the VM after a few idle minutes.

**Restarts:** every service has `restart: unless-stopped`, so it comes back whenever Docker Desktop starts, unless you stopped it explicitly with `stop`.

**Commands that destroy data:**
- `docker compose ... down -v` deletes **all volumes**: Langfuse's data **and the Qdrant index** (about $0.012 and a minute to rebuild with `ingest`).
- `docker volume rm dfsa-rag_<name>` deletes one volume.

**Upgrading Langfuse:**
1. Bump **both** `langfuse/langfuse` and `langfuse/langfuse-worker` to the same new `4.x.y` in the compose file.
2. Run `pull`, then `up -d`.

Migrations run by themselves on start. Read the release notes before a jump of several minor versions.

---

## 5. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `required variable LANGFUSE_… is missing a value` | That name is missing or blank in `.env` | `uv run python -m scripts.init_langfuse_env --email …` |
| A data store `(unhealthy)` or `Restarting` | Bad config, or a password that differs from the one stored when the volume was created | `docker compose ... logs --tail 50 <service>` |
| `NOAUTH`, `WRONGPASS`, `password authentication failed` in web/worker logs | A secret in `.env` changed after the volume was created | Put the old value back. Only a reset (`down -v`) applies new passwords |
| Login loops, or the cookie isn't kept | You opened `localhost` instead of `127.0.0.1` | Use http://127.0.0.1:3000 |
| `curl: (7) Failed to connect` on port 3000 | web not started yet, or port 3000 already in use | Check `ps`; read web's logs for `Ready` or a bind error |
| `events_full table does not exist. Retrying` (worker, at start) | A start-up race with web's migrations | Harmless if `All prerequisites validated` follows |

---

## 6. Open items
- **ClickHouse system logs**: 503 MB was written in the first 30 minutes, which includes the migrations, so the idle rate is not measured yet. Check the volume size after a few days with `docker system df -v`. A config file mounted into `/etc/clickhouse-server/config.d/` can disable or shorten (TTL) `query_log`, `metric_log`, `trace_log`, etc. Not done yet.
- web and worker have no healthcheck in compose. `/api/public/health` (web) and the worker's internal port 3030 could be used.
- SDK media uploads don't work, because MinIO has no host port. Text-only tracing doesn't need them.
- Next, in **E2**: `LangfuseSettings`, `get_langfuse()` (does nothing when disabled, so tests stay offline) and `scripts/check_langfuse.py` (an authenticated SDK check with the keys in `.env`).
