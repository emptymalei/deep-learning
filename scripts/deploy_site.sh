#!/usr/bin/env bash
# Publish build/site/ to the gh-pages branch (the deploy from main).
#
#     scripts/deploy_site.sh [site-dir]   # default: build/site
#
# The branch is replaced by one fresh commit on every deploy, as before. What
# is new is that the previews of open pull requests, which live in pr-preview/
# on the same branch (scripts/pr_preview.sh), are carried over. Previews of
# pull requests that are no longer open are dropped here, in case their
# clean-up run did not happen; set OPEN_PRS to a space-separated list to
# override the lookup, which otherwise needs the gh CLI and GH_TOKEN.
#
# The push uses --force-with-lease and the whole step is retried, so a preview
# pushed while this runs is not lost.
set -euo pipefail

site=${1:-build/site}
: "${REMOTE:=origin}"
: "${PAGES_BRANCH:=gh-pages}"
: "${CNAME:=dl.leima.is}"
: "${GHP_IMPORT_VERSION:=2.1.0}"
sha=$(git rev-parse --short=7 HEAD)

[[ -f $site/index.html ]] || { echo "no built site in $site" >&2; exit 2; }
git config user.name >/dev/null || git config user.name "github-actions[bot]"
git config user.email >/dev/null || git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

open_prs() {
    if [[ -n ${OPEN_PRS+set} ]]; then
        echo "$OPEN_PRS"
    else
        gh pr list --state open --limit 500 --json number --jq '.[].number' | tr '\n' ' '
    fi
}

for attempt in 1 2 3; do
    rm -rf "$site/pr-preview"
    old=$(git ls-remote "$REMOTE" "refs/heads/$PAGES_BRANCH" | cut -f1)
    if [[ -n $old ]]; then
        git fetch --quiet --depth=1 "$REMOTE" "$PAGES_BRANCH"
        if git cat-file -e "FETCH_HEAD:pr-preview" 2>/dev/null; then
            git archive FETCH_HEAD pr-preview | tar -x -C "$site"
            # Keep only the previews of pull requests that are still open.
            if open=$(open_prs); then
                for dir in "$site"/pr-preview/pr-*; do
                    [[ -d $dir ]] || continue
                    number=${dir##*/pr-}
                    if [[ " $open " != *" $number "* ]]; then
                        echo "dropping the preview of closed pull request #$number"
                        rm -rf "$dir"
                    fi
                done
            else
                echo "could not list the open pull requests; keeping every preview" >&2
            fi
            rmdir "$site/pr-preview" 2>/dev/null || true
        fi
    fi

    # --no-jekyll adds .nojekyll; --cname keeps the custom domain;
    # --no-history makes the branch one fresh commit.
    uvx "ghp-import==$GHP_IMPORT_VERSION" --no-jekyll --cname "$CNAME" --no-history \
        --remote "$REMOTE" --branch "$PAGES_BRANCH" \
        --message "Deploy $sha (Zensical)" "$site"
    lease="--force"
    [[ -n $old ]] && lease="--force-with-lease=refs/heads/$PAGES_BRANCH:$old"
    if git push --quiet "$REMOTE" "$lease" "refs/heads/$PAGES_BRANCH:refs/heads/$PAGES_BRANCH"; then
        rm -rf "$site/pr-preview"
        echo "deployed $sha to $PAGES_BRANCH"
        exit 0
    fi
    echo "$PAGES_BRANCH changed during the deploy (attempt $attempt), trying again" >&2
done
echo "could not push to $PAGES_BRANCH" >&2
exit 1
