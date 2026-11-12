# AI Expense Tracker (Nivunapro)

Upload a bank or UPI statement and the app cleans it up, categorizes every payment with an LLM,
shows where your money goes, writes a savings and investment plan from your own history, and answers
questions such as *"gold is falling, shall I invest now?"* using your plan plus a live web search.

It is a FastAPI backend with SQLite, one HTML dashboard, and free LLM models through OpenRouter.
There are no logins: each browser gets its own private database.

> **Not financial advice.** The reports and chat answers are for education only and are not from a
> SEBI-registered adviser. Past returns do not predict future returns.

---

## Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [How each feature works](#how-each-feature-works)
5. [Data and privacy model](#data-and-privacy-model)
6. [API reference](#api-reference)
7. [Setup](#setup)
8. [Configuration](#configuration)
9. [Testing and CI](#testing-and-ci)
10. [Scripts and evals](#scripts-and-evals)
11. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
12. [Known limitations](#known-limitations)

---

## What it does

```
AI Expense Tracker
├── Import statements            CSV, XLSX or PDF from any bank, with no per-bank parser
├── Categorize payments          15 categories, with a clean merchant name for each
├── Learn from corrections       change one category and the app remembers it for that merchant
├── Dashboard                    income, spending, left over, trends, top merchants
├── Budgets                      a monthly limit per category, with used / near / over status
├── AI insights report           where to cut, and how to split what you could invest
└── Ask about your money (chat)  uses your plan, live news and real fund prices
```

**Where AI is used, and where it is not**

```
Uses an LLM                                   Plain Python / pandas (no AI)
├── Reading a spreadsheet's layout            ├── All totals, trends and percentages
├── Extracting rows from a PDF                ├── Budgets and their status
├── Merchant name + category                  ├── Duplicate detection
├── Insights: summary, which cuts, split      ├── Recurring and fastest-growing spend
└── Chat: routing, fund names, the answer     ├── All rupee arithmetic and validation
                                              └── Fund returns (from live prices)
```

The rule that shapes the whole design: **the LLM never does arithmetic, and every claim it makes is
checked by Python before the user sees it.**

---

## Architecture

```
                       ┌──────────────────────────────┐
                       │  Browser                     │
                       │  app/ui/ai-expense-tracker.html
                       │  (one page, Chart.js)        │
                       └──────────────┬───────────────┘
                                      │ HTTP + cookie "browser_id"
                       ┌──────────────▼───────────────┐
                       │  FastAPI  (app/main.py)      │
                       │  middleware: browser_id      │
                       └──────────────┬───────────────┘
                                      │
          ┌───────────┬───────────────┼───────────────┬────────────┐
          ▼           ▼               ▼               ▼            ▼
    transactions   imports         budgets       analytics     insights / chat
          │           │               │               │            │
          └───────────┴───────┬───────┴───────────────┴────────────┘
                              │   routers/  : thin HTTP layer
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  services/  : the business logic                        │
        │                                                         │
        │  importer ─▶ categorizer ─▶ llm  ◀── insights ◀── chat  │
        │                              │          │          │    │
        │              analytics (pandas)         │          │    │
        │                                    market (mfapi)  web  │
        └──────┬──────────────────────────────┬───────────┬───────┘
               │                              │           │
               ▼                              ▼           ▼
        ┌─────────────┐              ┌──────────────┐  ┌──────────────┐
        │  SQLite     │              │  OpenRouter  │  │  mfapi.in    │
        │  per browser│              │  (free LLMs) │  │  DuckDuckGo  │
        │  + shared   │              └──────────────┘  └──────────────┘
        └─────────────┘
```

**Layers**

```
HTTP        routers/    validate input, call a service, turn errors into status codes
Logic       services/   everything that matters; no HTTP knowledge
Data        db.py       one SQLite file per browser, plus one shared file
            models.py   tables and request shapes (SQLModel)
```

### One request, from cookie to database

```
Request arrives
   │
   ▼
identify_browser middleware
   │   cookie missing or malformed?  → make a random 32-char hex id
   │   id becomes a FILE NAME, so it must match [0-9a-f]{32} exactly
   ▼
router  ──Depends(get_session)──▶  opens ~/.expense-tracker/browsers/<id>.db
   │                               (first visit: copies samples/seed.db into place)
   ▼
service does the work  ──▶  response  ──▶  cookie set if the browser was new
```

---

## Project structure

```
Nivunapro/
├── README.md
├── requirements.txt          pinned dependencies
├── pytest.ini
├── .env                      secrets (never committed)
├── .github/workflows/
│   └── tests.yml             CI: Python 3.14, install, pytest
│
├── app/                      THE APPLICATION
│   ├── main.py               app, middleware, router wiring, serves the dashboard
│   ├── config.py             reads .env: API key, model list, data folder
│   ├── db.py                 per-browser + shared SQLite, seed copy, reset
│   ├── models.py             tables, categories, request schemas
│   │
│   ├── routers/              HTTP ENDPOINTS
│   │   ├── transactions.py   CRUD, restore (undo), recategorize
│   │   ├── imports.py        POST /import, POST /reset
│   │   ├── budgets.py        monthly limit per category
│   │   ├── analytics.py      summary, categories, trend, merchants
│   │   ├── insights.py       /market, GET and POST /insights
│   │   └── chat.py           POST /chat
│   │
│   ├── services/             BUSINESS LOGIC
│   │   ├── importer.py       read a statement, dedupe, save
│   │   ├── categorizer.py    LLM merchant + category, learning from corrections
│   │   ├── analytics.py      pandas: every number on the dashboard
│   │   ├── insights.py       facts → LLM → validate → report
│   │   ├── chat.py           router → search → fund lookup → answer → validate
│   │   ├── llm.py            OpenRouter client: masking, cache, fallbacks, timeouts
│   │   ├── market.py         mfapi.in: fund prices, returns, name lookup
│   │   └── web.py            DuckDuckGo search: trusted sites only, cached
│   │
│   └── ui/
│       └── ai-expense-tracker.html    the whole frontend
│
├── samples/                  SYNTHETIC DEMO DATA ONLY
│   ├── seed.db               what every new browser starts with
│   ├── demo_statement.csv
│   └── demo_upi_statement.pdf
│
├── scripts/                  dev tools, not part of the server (see Scripts and evals)
└── tests/                    pytest suite; the LLM, search and fund API are faked
```

---

## How each feature works

### 1. Importing a statement

The LLM works out each file's layout, so no bank needs its own parser.

```
POST /import  (file, max 10 MB, .csv / .xlsx / .pdf)
   │
   ├─ CSV / XLSX
   │    ├─ read every cell as text (bank exports have uneven rows above the table)
   │    ├─ send ~12 sample rows to the LLM
   │    │     └─ it answers: header row, date / description / amount / debit / credit columns,
   │    │                    and whether dates are day-first
   │    └─ Python reads EVERY row using that map; non-transaction rows are counted as skipped
   │
   ├─ PDF
   │    ├─ extract the text (max 60 pages)
   │    ├─ split into chunks of 3 pages, 3 chunks sent in parallel
   │    ├─ LLM returns date, description, amount, income/expense for each chunk
   │    └─ GUARD: an amount that does not appear in the source text is dropped
   │              (models can invent rows)
   ▼
clean rows  (amount always positive; "type" gives the direction)
   │
   ▼
DEDUPE
   ├─ each row gets a hash of date + description + amount + type + its repeat number
   │    → re-importing the same file adds nothing
   │    → two genuine identical rows in one file (two teas in a day) are both kept
   └─ the same payment in a bank statement and a UPI statement, with different
      descriptions, matches on date + amount + type and counts as already imported
   ▼
categorize (next section)  →  save  →  { imported, duplicates, skipped, uncategorized, warning }
```

If the browser still holds the made-up demo rows, they are cleared first so demo and real rows never mix.

### 2. Categorization

```
new transactions
   │
   ▼
MASK     account / phone / reference numbers → [NUM], UPI ids and emails → [ID]
   │     (free models may log what they receive)
   ▼
GROUP    rows whose masked text and type match are sent once
   │     ("UPI/SWIGGY/[NUM]" appearing 40 times = 1 item)
   ▼
ANSWERED BEFORE?  look in the shared DescriptionAnswer table ──yes──▶ reuse, no LLM call
   │ no
   ▼
BATCH of 100 → LLM, with the merchant names already in use (so "Swiggy" stays "Swiggy")
   │           returns { id, merchant, category } for each item
   ▼
SAVE     the answer for everyone (shared DB), apply it to the rows (source = "llm")
```

If the LLM fails, rows stay `Uncategorized` and the user sees a warning. "Retry AI" tries again.

**Learning.** When a user changes a category:

```
user picks a new category for "Swiggy"
   ├─ that row                       → source = "user"
   ├─ every other Swiggy row         → updated too
   └─ MerchantCategory remembers it  → future imports use it
                                       a user's choice always beats the LLM
```

**Business rules** (in `models.py`)

```
Categories
├── Spending            everything except the three below
├── Not spending        Transfers, Investments   (moving your own money is not spending)
├── Income              Income
├── Refund              money coming in under a spending category reduces that category
└── Never advised cut   Rent & Housing, Bills & Utilities, Health, Other
```

### 3. Dashboard and budgets

Every figure is computed in `services/analytics.py` with pandas, never by the LLM.

```
GET /analytics/summary      income, spending, invested, left over, savings rate for a month
GET /analytics/categories   spending by category with share of total
GET /analytics/trend        one row per month, for the cash-flow chart
GET /analytics/merchants    top merchants
GET /budgets                limit, spent, % used and status (ok / near / over) per category
```

With no `?month=` the latest month that has data is used.

### 4. AI insights report

The LLM judges and writes. Python works out the facts before, and checks and calculates after.

```
POST /insights
   │
   ▼
STEP 1  PYTHON builds the facts (no AI)
   │      last 6 months of totals · top 25 merchants and their history
   │      fastest-growing merchants · same-charge-every-month payments
   │      largest payments · budget status
   │      averages over the last 3 WHOLE months (a month just started would drag them down)
   │      emergency fund target (6 months of spending) · term cover (10-15 years of income)
   │      live fund returns
   ▼
STEP 2  LLM gets the facts and strict instructions, returns
   │      summary · spending cuts (merchant + PERCENT to cut + action)
   │      investment split across 4 products, as PERCENTAGES
   ▼
STEP 3  PYTHON validates and calculates
   │      cut names a merchant that really can be cut?   no  → rejected
   │      rupee target = average × (100 − percent), rounded to Rs 100
   │      saving smaller than Rs 200 a month?            yes → rejected
   │      saving uses the 3-month average, so a one-off spike is not counted as saved
   │      available to invest = average left over + total savings   (never below zero)
   │      percentages over 100 are scaled down; each amount rounded down to Rs 100
   │      pasted copies of the answer and field names are stripped from the text
   ▼
STEP 4  saved as an InsightReport → the page shows it on reload with no LLM call
```

The LLM returns percentages and not rupees on purpose: in testing, a rupee target of "5,000"
sometimes came back as `5`. See the comments in `services/insights.py`.

The **Insights page** shows the result as a walk-through:

```
1  How much can you invest each month?
   [ Average left over ] + [ Saved by the cuts ] = [ Available to invest ]
2  Where does it go?          one bar per product, sized by its share
3  When is the emergency fund ready?    target ÷ monthly amount
```

### 5. Ask about your money (chat)

Questions like *"gold rate seems to be falling, shall I invest now?"* or *"which mutual fund has done
well recently, for my savings?"* are answered from your own plan plus the web.

```
POST /chat  { message, history }     the page keeps the conversation in memory and
   │                                 sends the last turns with each question
   │
   ├─ 1. CONTEXT      your averages, available to invest, emergency target, planned split
   │
   ├─ 2. ROUTER LLM   → intent: gold | mutual_fund | general
   │                    + a short generic search query
   │      PYTHON cleans the query: letters only, so no figure of yours can reach a search engine
   │
   ├─ 3. RETRIEVE
   │      gold         news search + the gold fund's 1-month / 3-month / 1-year moves (mfapi.in)
   │      mutual_fund  page search + news search
   │                   → LLM lists fund NAMES that the snippets mention
   │                   → PYTHON checks each name really appears in a snippet,
   │                     looks it up on mfapi.in, computes real returns,
   │                     drops names that match no fund or have no recent price,
   │                     ranks by 1-year return, keeps the best 4
   │      general      no search; answers from your own figures
   │
   ├─ 4. ANSWER LLM   personalised to your plan; cites snippet ids
   │
   └─ 5. PYTHON checks   cited ids must exist · sentences promising a result are removed
                         ("guaranteed", "risk free", "will rise") · disclaimer attached
```

```
Web search safety
├── only trusted domains (TRUSTED_DOMAINS in services/web.py), matched on the real hostname
├── titles and snippets only, never whole pages
├── snippets are marked as untrusted text in the prompt
├── results cached for a day; if search fails, yesterday's copy or nothing is used
└── if nothing was found the chat still answers, and the UI says live search was unavailable
```

A stalled free model is handled by a 45-second timeout and one automatic retry.

### 6. The LLM layer (`services/llm.py`)

Every LLM call goes through `ask_llm()`.

```
ask_llm(system, user, schema)
   │
   ├─ mask() the user text
   ├─ saved answer for this exact prompt?  ──yes──▶ return it (free and instant)
   ├─ OpenRouter, with a LIST of :free models (the first is main, the rest are fallbacks)
   │      refuses any model that is not ":free", because the key may belong to a paid account
   ├─ strict JSON-schema output, then parsed
   ├─ answer missing a required field?  → error, and nothing is saved
   └─ save the answer in the shared LLMCache
```

---

## Data and privacy model

```
~/.expense-tracker/                 (outside the project, so statements never reach git or OneDrive)
├── shared.db                       ONE file, used by everyone. Safe to share.
│     LLMCache            saved LLM answers
│     DescriptionAnswer   masked description → merchant + category
│     FundSnapshot        fund figures, reused for a day
│     SearchCache         web search results, reused for a day
│
└── browsers/
      ├── <32-hex-id>.db            ONE file PER BROWSER. Private.
      └── ...                         Transaction · Budget · MerchantCategory · InsightReport
```

```
Privacy measures
├── numbers and ids masked before anything is sent to an LLM
├── names of people (Transfers, Rent) are not sent as "known merchants"
├── the search query is letters only, and never built from user figures
├── the cookie is HttpOnly + SameSite=Lax (+ Secure on https), and its value is validated
│   because it becomes a file name
├── one database file per browser, not a user_id filter that could be forgotten
└── data lives outside the repo; .env and *.db are gitignored (seed.db is the exception)
```

### Tables

```
Transaction       id, date, description, merchant, amount, type, category,
                  category_source (llm | user | none), source (csv | xlsx | pdf | manual | demo),
                  dedupe_hash (unique)
MerchantCategory  key, name, category, source        a merchant's spelling and any user fix
Budget            category, monthly_limit
InsightReport     id, report (JSON)
─────────────────────────────────────────────────────  above: per browser
DescriptionAnswer text + type, merchant, category      below: shared
LLMCache          prompt_hash, response, model, created_at
FundSnapshot      code, fetched_on, snapshot (JSON)
SearchCache       query, fetched_on, results (JSON)
```

---

## API reference

Interactive docs are at `http://127.0.0.1:8000/docs` once the server runs.

```
GET    /                              the dashboard page
GET    /categories                    the 15 categories

POST   /import                        upload a .csv / .xlsx / .pdf statement
POST   /reset?demo=true|false         start again with demo data, or empty

GET    /transactions                  ?start &end &category &type &limit &offset (max 1000 per page)
POST   /transactions                  add one (category optional: the LLM picks)
GET    /transactions/{id}
PATCH  /transactions/{id}             edit; changing the category also teaches the app
DELETE /transactions/{id}
POST   /transactions/restore          undo a delete
POST   /transactions/recategorize     retry the LLM on rows left Uncategorized

GET    /budgets?month=YYYY-MM
PUT    /budgets/{category}            { monthly_limit }
DELETE /budgets/{category}

GET    /analytics/summary             ?month=
GET    /analytics/categories          ?month=
GET    /analytics/trend
GET    /analytics/merchants

GET    /market                        live figures for the three example funds
GET    /insights                      the last saved report (404 if none)
POST   /insights                      write a new report
POST   /chat                          { message (≤500 chars), history (≤20 turns) }
```

Errors: `400` for no data or a bad file, `404` for a missing row or report, `422` for invalid input,
`502` when the LLM cannot answer (the message is safe to show).

---

## Setup

You need Python and an [OpenRouter](https://openrouter.ai) API key. Free models are enough.
CI and the pinned versions target **Python 3.14**; the test suite also passes on Python 3.12.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
# source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1         # PowerShell

# 2. Install
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure (see below), then run
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. A new browser starts with made-up demo data from `samples/seed.db`.
Use **Upload statement**, or try `samples/demo_statement.csv` and `samples/demo_upi_statement.pdf`.

> `samples/` holds invented data only. Do not put real statements in the repository.

---

## Configuration

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
LLM_MODELS=model-a:free,model-b:free,model-c:free
# DATA_DIR=~/.expense-tracker          optional: where the databases are kept
```

```
OPENROUTER_API_KEY   required for any AI feature
LLM_MODELS           comma-separated, tried in order. Every name must end in ":free",
                     or the app refuses to send the request.
DATA_DIR             defaults to ~/.expense-tracker
```

Without a key the app still starts, and the demo data, dashboard, budgets and adding a transaction
with a category you choose all work. Anything that needs the LLM (importing a statement, insights,
chat, or letting the AI pick a category) shows a clear error.

---

## Testing and CI

```bash
pytest -q
```

```
tests/
├── conftest.py        temp database per test + a fake LLM, so no test sends a real request
├── test_importer.py   layouts, PDFs, dedupe, demo-data replacement
├── test_categorizer.py, test_transactions.py, test_budgets_analytics.py
├── test_insights.py   validation of cuts, savings, split, masking, caching
├── test_chat.py       search safety, fund lookup, routing, history, promises, timeouts
├── test_llm.py        masking, free-models-only guard, error bodies, saved answers
├── test_browser_data.py   per-browser isolation, cookie safety, demo seed and reset
├── test_ui.py         the page is served and has no leftover sample data
└── test_evals.py      the eval scripts' scoring, because it decides which LLM settings are used
```

GitHub Actions (`.github/workflows/tests.yml`) runs `pytest -q` on every push and pull request.
The LLM, DuckDuckGo and mfapi.in are all faked, so CI needs no key and no network access to them.

---

## Scripts and evals

Dev tools, run from the project root. The evals call the real OpenRouter API and count toward its
daily cap. All their data is invented and hand-labelled, so the scores are not measured rates on
real statements.

```
python scripts/make_demo_data.py            build the synthetic statements in samples/
python -m scripts.make_seed_db              rebuild samples/seed.db (uses the real LLM, ~3 requests)

python -m scripts.eval_categorizer          how accurate is categorization on unseen narrations?
python -m scripts.eval_known_merchants      does sending known names keep spellings consistent?
python -m scripts.eval_import               are statement layouts and PDFs read correctly?
python -m scripts.eval_insights             how good are the insights under each LLM setting?
python -m scripts.smoke_test_models         compare the free models in LLM_MODELS

Each eval accepts --settings, --runs and --out answers.json, and --rescore answers.json re-scores
saved answers with no new requests.
```

---

## Design decisions and trade-offs

```
Decision                                  Why                                    Cost
────────────────────────────────────────  ─────────────────────────────────────  ─────────────────────────
LLM reads layouts; Python reads rows      works for any bank, cheap (12 rows)    depends on the LLM
LLM gives percentages, Python does money  LLMs mangled rupee figures in testing   less natural wording
Validate every LLM claim in Python        free models hallucinate                more code than trusting it
One SQLite file per browser               no login, strong isolation             cookie is the only identity
Shared DB for cache and fund data         saves requests for everyone            must hold nothing private
Free models only, with fallbacks          no spend; resilient to one model down  slow, rate-limited, variable
Mask numbers before sending               free endpoints may log prompts         names of merchants still sent
Search: trusted domains, snippets only    avoids spam, injection, long prompts   narrower coverage
Funds ranked from real prices             not from how a site describes a fund   limited to funds mfapi.in has
Chat history held in the page             server stays stateless                 a reload starts a new chat
```

---

## Known limitations

```
├── Identity is a cookie. Clearing cookies loses access to that browser's data, and
│   old browser database files are never cleaned up.
├── SQLite suits one writer per file. A real multi-user deployment would move to
│   Postgres with accounts.
├── Free LLM models are slow and sometimes stall. Chat uses a 45-second timeout and one retry.
├── DuckDuckGo has no official free API. The ddgs library can be rate-limited or break;
│   the chat then answers from your own figures and says so.
├── Fund discovery depends on what the search returns that day, so a fund question
│   can occasionally find no funds. It says so and never invents one.
├── Scanned (image-only) PDFs cannot be read; there is no OCR.
├── Amounts are in rupees and the prompts assume Indian banks and UPI.
└── The chat gives general, educational guidance. Offering personalised investment advice to
    the public may need regulatory approval, so check this before deploying it widely.
```
