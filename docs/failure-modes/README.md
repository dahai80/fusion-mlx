# Failure Modes Library

Institutional source: `insight/fusion-claude-insight-1102.md` §6.3 step 1
(eval#25 cross-repo process). Failures are product assets: every acceptance
FAIL / residual ticket that reveals a *structural* root cause must produce an
entry here at close time — the issue alone is not the record.

## Flow rule (mandatory, same enforcement style as the silent-close audit)

1. On closing any FAIL / residual issue, write `docs/failure-modes/<category>.md`
   (or append to an existing category file): symptom signature, detection
   method, structural fix, and links to the source issue(s).
2. The entry must end with an **eval linkage** line: the corresponding core
   task id in `dahai80/fusion-eval` (`tasks/<domain>/tN-*.yaml`) — update or
   add a core task so the failure is regression-covered. Failure → task is
   the reverse half of the E-1 "tasks sourced from verified failures" rule.
3. Closing an issue without either the entry or an explicit "no structural
   learning" note in the close comment counts as a silent-close defect.

## Index

| Category | Signature | Source issues | Eval linkage |
|----------|-----------|---------------|--------------|
| (empty — append rows as entries land) | | | |

## Entry template

```markdown
## <short name> (<category>) — YYYY-MM-DD

- Symptom: <observable signature, exact error strings>
- Root cause: <structural mechanism, not the one-off trigger>
- Detection: <how it was / should have been caught>
- Fix: <structural change, PR refs>
- Sources: <issue refs>
- Eval linkage: fusion-eval tasks/<domain>/<task>.yaml (added | updated | n/a+reason)
```
