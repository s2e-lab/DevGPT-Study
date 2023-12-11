#!/bin/bash

FILE="pr_unique_url_java.txt"

CLONE_DIR="cloned_repos"

mkdir -p $CLONE_DIR

extract_repo_and_pr_id() {
    local pr_url=$1
    local repo_url=$(echo $pr_url | sed -r 's|/pull/[0-9]+||')".git"
    local pr_id=$(basename $pr_url | sed -r 's|/pull/||')

    echo $repo_url $pr_id
}

while IFS= read -r pr_url
do
    echo "Processing PR: $pr_url"

    read repo_url pr_id <<< $(extract_repo_and_pr_id $pr_url)

    dir_name=$(echo $repo_url | sed -r 's|.*/(.*).git|\1|')

    if [ ! -d "$CLONE_DIR/$dir_name" ]; then
        git clone $repo_url "$CLONE_DIR/$dir_name"
    fi

    cd "$CLONE_DIR/$dir_name"

    git fetch origin pull/$pr_id/head:pr_$pr_id
    git checkout pr_$pr_id

    cd -

    echo "Repository for PR $pr_id cloned and checked out."

done < "$FILE"

echo "All PRs processed."
