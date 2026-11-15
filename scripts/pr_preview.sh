#!/usr/bin/env bash
# Publish, or remove, the site preview of a pull request.
#
#     scripts/pr_preview.sh publish <pr-number> [site-dir]   # default: build/site
#     scripts/pr_preview.sh remove  <pr-number>
#     scripts/pr_preview.sh comment <pr-number> <commit-sha>
#
# Previews live next to the published site, in pr-preview/pr-<number>/ on the
# gh-pages branch, so they are served at
#
#     https://dl.leima.is/pr-preview/pr-<number>/
#
# The site uses relative links, so it works from a subfolder as built. The
# deploy from main (scripts/deploy_site.sh) carries pr-preview/ over.
#
# The push is retried: other pull requests and the deploy from main write to
# the same branch.
#
# `comment` writes the link into one comment on the pull request and updates
# that comment on later pushes. It needs the gh CLI, GH_TOKEN and
# GITHUB_REPOSITORY, which GitHub Actions provides.
set -euo pipefail

action=${1:?publish, remove or comment}
pr=${2:?pull request number}
site=${3:-build/site}
: "${PREVIEW_BASE_URL:=https://dl.leima.is}"
: "${REMOTE:=origin}"
: "${PAGES_BRANCH:=gh-pages}"

[[ $action == publish || $action == remove || $action == comment ]] || { echo "unknown action: $action" >&2; exit 2; }
[[ $pr =~ ^[0-9]+$ ]] || { echo "the pull request number must be digits: $pr" >&2; exit 2; }

if [[ $action == comment ]]; then
    sha=${3:?commit sha}
    url="$PREVIEW_BASE_URL/pr-preview/pr-$pr/"
    marker="<!-- pr-preview -->"
    body="$marker
**Site preview:** $url

Built from \`${sha:0:7}\`. After a push it can take a minute or two before the page updates; the badge in the bottom left corner of each page shows the commit you are looking at."
    comments="repos/$GITHUB_REPOSITORY/issues/$pr/comments"
    id=$(gh api "$comments" --paginate --jq ".[] | select(.body | startswith(\"$marker\")) | .id" | head -n 1)
    if [[ -n $id ]]; then
        gh api --silent --method PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$id" -f body="$body"
    else
        gh api --silent --method POST "$comments" -f body="$body"
    fi
    echo "commented on #$pr: $url"
    exit 0
fi
if [[ $action == publish && ! -f $site/index.html ]]; then
    echo "no built site in $site" >&2
    exit 2
fi

dest="pr-preview/pr-$pr"
work=$(mktemp -d)
cleanup() { git worktree remove --force "$work" 2>/dev/null || rm -rf "$work"; }
trap cleanup EXIT

for attempt in 1 2 3 4 5; do
    git fetch --quiet --depth=1 "$REMOTE" "$PAGES_BRANCH"
    git worktree remove --force "$work" 2>/dev/null || rm -rf "$work"
    git worktree add --quiet --detach "$work" FETCH_HEAD

    rm -rf "${work:?}/$dest"
    if [[ $action == publish ]]; then
        mkdir -p "$work/$dest"
        cp -R "$site/." "$work/$dest/"
        rm -rf "$work/$dest/pr-preview" "$work/$dest/CNAME"
    fi

    git -C "$work" add --all
    if git -C "$work" diff --cached --quiet; then
        echo "$dest: nothing to change"
        exit 0
    fi
    git -C "$work" \
        -c user.name="github-actions[bot]" \
        -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
        commit --quiet --message "Preview of pull request #$pr: $action"
    if git -C "$work" push --quiet "$REMOTE" "HEAD:refs/heads/$PAGES_BRANCH"; then
        [[ $action == publish ]] && echo "$dest: published" || echo "$dest: removed"
        exit 0
    fi
    echo "push rejected (attempt $attempt), trying again" >&2
    sleep $((attempt * 2))
done
echo "could not push $dest to $PAGES_BRANCH" >&2
exit 1
