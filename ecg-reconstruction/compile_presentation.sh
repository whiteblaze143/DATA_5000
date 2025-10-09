#!/usr/bin/env bash
# Compile LaTeX presentation slides

echo "Compiling ECG Reconstruction Presentation..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install LaTeX (TeX Live or MiKTeX)"
    echo "On Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo "On macOS: brew install mactex"
    echo "On Windows: Install MiKTeX from https://miktex.org/"
    exit 1
fi

# Compile the presentation
pdflatex -interaction=nonstopmode presentation_slides.tex

# Run again for references (if any)
pdflatex -interaction=nonstopmode presentation_slides.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

echo "Presentation compiled successfully!"
echo "Output file: presentation_slides.pdf"