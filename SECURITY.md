# Security Policy — Credentials & Git (MANDATORY)

**Any process (Hermes, cron, scripts, or a human) that touches git commits MUST
follow this. Never commit secrets to any repository, public or private.**

## The one hard rule
> **Keys, passwords, tokens, API credentials and secrets NEVER go into git.**
> They live ONLY in a gitignored `.env` file or a non-git database.

## Where secrets go (allowed)
| Secret type | Storage | Example |
|---|---|---|
| API keys, tokens, passwords | gitignored `.env` / `.env.local` | `~/BacktestingMCP/.env` |
| altFINS login | gitignored `.env.altfins` | `~/BacktestingMCP/.env.altfins` |
| Exchange keys (Binance/Bybit/etc) | non-git DB | `trades.db` → `APIKeys` table (gitignored) |
| Runtime session tokens | gitignored JSON | `bitfunded_tokens_*.json` (gitignored) |
| Device/registration IDs | gitignored file | `data/bitfunded_device.txt` (gitignored) |

Committed template showing placeholders only: `.env.template` (safe).

## Never commit these files
`.env`*, `*.env`, any `*secret*`, `*password*`, `*credential*`, `*token*.json`,
`*api[_-]?key*.json`, `*.pem/.key/.p12/.jks`, `*.ovpn`, `id_rsa`,
`.git-credentials`, `*backup*cred*.json`.

## Before every git add / commit / push (guard checklist)
1. `git status --short` — review every file you're about to stage.
2. `git add <specific files>` — **never `git add .` / `-A`** blindly.
3. The **pre-commit hook** (below) auto-scans and BLOCKS secrets — but do not
   rely on it alone; review manually too.

## Enforcement (already installed here)
- **Skill `git-secret-guard`** — loaded by Hermes before any git work; contains
  full file/content patterns + incident response.
- **Pre-commit hook** at `.git/hooks/pre-commit` in both `BacktestingMCP`
  (home) and `/opt/Trading-WebHook-Bot` (VPS). Blocks commits staging secret
  files or content. Installed 2026-08-31.

## If a secret WAS committed (incident response)
1. Rotate the credential **immediately** — this is the true fix (untracking
   does not revoke already-pushed copies).
2. `git rm --cached <file>` + add to `.gitignore` + commit.
3. If it reached a **public** remote: purge history with
   `git filter-repo --force --path <file> --invert-paths`, re-add origin,
   force-push (with owner consent). Warn: already-cloned copies still contain it.

## Real incidents on this infra
- **Aug 2026:** `.env.altfins` (live altFINS password) was committed & pushed to a
  **public** BacktestingMCP repo. Resolved: password rotated; file untracked &
  gitignored; `git filter-repo` purged all 264 commits; history force-pushed.
- **Aug 2026:** `exchanges/bitfunded_tokens_1.json` + `data/bitfunded_device.txt`
  (live creds) were untracked-not-gitignored on the bot repo. Resolved: gitignored
  (were never in history, so no purge needed).