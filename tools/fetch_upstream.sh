#!/bin/sh
# Downloads the pinned upstream code and checkpoint into <dir> (default: upstream/).
set -eu
dir=${1:-upstream}
commit=1e13597c42d4858b7cfd7cfdae01e297263364b2
hf_revision=b6559de2398d7140b985f28fd2c19fb5e47ddabe
mkdir -p "$dir"
if [ ! -d "$dir/maia3/.git" ]; then git clone --quiet https://github.com/CSSLab/maia3 "$dir/maia3"; fi
git -C "$dir/maia3" checkout --quiet "$commit"
if [ ! -f "$dir/maia3-5m.pt" ]; then
  curl -fsSL -o "$dir/maia3-5m.pt" "https://huggingface.co/UofTCSSLab/Maia3-5M/resolve/$hf_revision/maia3-5m.pt"
fi
echo "ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f  $dir/maia3-5m.pt" | shasum -a 256 -c -
