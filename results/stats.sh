#!/bin/bash

file="$1"
if [[ -z $file ]]; then
    echo "No file given"
    exit
fi

function filter_steps {
    cat "$file" | grep took | sed 's/^.*info] \(.*\) took.*$/\1/p' | sort -u
}

step="$2"
if [[ -z $step ]]; then
    echo "No step given, select one of:"
    filter_steps
    exit
fi


function filter_took {
    cat "$file" | grep -i "$step" | sed 's/^.*took: \(.*\)ms.*$/\1/p'
 
}

echo "Average:"
filter_took | awk '{ total += $1 } END { print total/NR }'

echo "StdDev:"
filter_took | awk '{x+=$1;y+=$1^2} END {print sqrt(y/NR-(x/NR)^2)}'

echo "Max:"
filter_took | awk '{ print $1 }' | sort | tail -n 1

echo "Min:"
filter_took | awk '{ print $1 }' | sort | head -n 1
