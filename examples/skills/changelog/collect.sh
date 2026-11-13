#!/bin/sh
# Print the commit range to summarize and its one-line log.
# Usage: collect.sh [<range> | <ref> | since <ref>]
set -e
arg="$*"
arg="${arg#since }"
if [ -n "$arg" ]; then
    case "$arg" in
        *..*) range="$arg" ;;
        *) range="$arg..HEAD" ;;
    esac
elif tag=$(git describe --tags --abbrev=0 2>/dev/null); then
    range="$tag..HEAD"
else
    range="HEAD"
    limit="-n 30"
fi
echo "Range: $range"
echo
# shellcheck disable=SC2086
git log --no-merges $limit --format='%h %s (%an)' "$range"
