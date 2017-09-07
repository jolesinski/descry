#!/bin/bash

exec="$1"
dir="$2"

cd "$dir"

files=*.yaml

mkdir "$dir"/results

for f in $files
do
   echo "processing " "$f"
   "$exec" "$f" > "$dir"/results/"$f"-result
done
