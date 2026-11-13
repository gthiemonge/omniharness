---
name: changelog
description: Draft release notes from the git history of the current repository
---

# Changelog drafting

Draft user-facing release notes for the git repository in your working
directory. Range or hint from the user: `$ARGUMENTS` (if empty, use the commits
since the most recent tag, or the last 30 commits when there is no tag).

## Process

1. Run `bash $SKILL_DIR/collect.sh $ARGUMENTS` to get the commit range and the
   one-line log.
2. For commits whose subject is unclear, inspect them with `git show --stat <sha>`
   (never more than 10 commits).
3. Group changes under: **Features**, **Bug fixes**, **Deprecations / upgrade
   notes**, **Internal** (CI, refactors, docs). Drop pure merge commits.
4. Write one line per change, in the imperative, ending with the short sha.

## Output

Write `CHANGELOG_DRAFT.md` into the output directory you were given
(`$OUTPUT_DIR`), then reply with a 3-line summary.
