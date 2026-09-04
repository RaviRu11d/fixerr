# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-09-04

### Added
- **Deterministic Fingerprinting**: Added `compute_fingerprint()` function in `fixerr/fingerprint.py` that computes a compact, 16-character SHA-256 slice from normalized command and error text (collapsing line numbers, volatile paths, hex values, and secrets via `redaction.normalize`). Gracefully handles Tier 1 command-only captures using command and exit code.
- **Enhanced Store & SQLite Schema**: Added `fingerprint`, `occurrence_count`, `first_seen`, and `last_seen` columns to `errors` table, with an index `idx_errors_fingerprint`. Added automatic migration and backfilling for legacy databases.
- **Deduplication Logic**: When capturing an error with an existing fingerprint, increments `occurrence_count`, updates `last_seen`, and refreshes context rather than inserting duplicate records. Seamlessly upgrades previous Tier 1 placeholders with full stderr, normalized text, and embeddings.
- **Tier 1 Anti-Spam**: Honors immediate 5s anti-spam suppression while updating `occurrence_count` and `last_seen` on recurrence in `auto_capture_tier1()`.
- **Updated Match Dataclass**: Enhanced `Match` dataclass in `store.py` with new metadata, and updated `list_errors()` to order by `last_seen DESC`.
- **CLI Enhancements**: 
  - In `fixerr show <id>`: Displays `occurrences: <count> (first seen: <rel_time>, last seen: <rel_time>)` and `fingerprint: <hash>`.
  - In `fixerr search <query>`: Displays `[seen Nx]` badge for recurring errors.
- **Interactive Dashboard (TUI) Improvements**:
  - Added recurrence badges (`×<count>`) in error list items.
  - Added occurrence metrics and first/last seen relative timestamps in the detail view panel.
- **Comprehensive Test Suite**: Added `tests/test_dedup.py` with 14 unit tests covering deterministic fingerprint computation, deduplication logic, recurrence handling, Tier 1/Tier 2 upgrades, legacy schema migration, and CLI/dashboard formatting.