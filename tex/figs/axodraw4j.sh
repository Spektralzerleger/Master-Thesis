#! /bin/bash

read -p "Enter filename without .tex extension: " filename

latex $filename.tex
dvips $filename.dvi
ps2pdf $filename.ps

echo "Feynman diagram is produced."
