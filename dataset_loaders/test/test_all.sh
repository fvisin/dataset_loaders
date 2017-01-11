#!/bin/bash

# Execute all mains
for f in ../images/*.py; do
    echo "\n\n********************"
    echo "Running $f"
    python "$f";
done
for f in ../videos/*.py; do
    echo "\n\n********************"
    echo "Running $f"
    python "$f";
done
