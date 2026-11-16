# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

Use GitHub's private reporting form: <https://github.com/Gitguy268/lvw-stt/security/advisories/new>

Say what you found, how to reproduce it, and your OS. You should get a reply within a week. Please give the fix time to ship before you publish details.

## Scope

- scripts that download or unpack files (archives, model snapshots) in an unsafe way;
- tokens leaking into logs or published model cards;
- the GitHub Actions workflows in this repository.

## Supported versions

Only the `main` branch is supported.
