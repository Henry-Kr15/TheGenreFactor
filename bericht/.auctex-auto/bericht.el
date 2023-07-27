(TeX-add-style-hook
 "bericht"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrartcl" "12pt" "bibliography=totoc" "captions=tableheading" "titlepage=firstiscover" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("rerunfilecheck" "aux") ("geometry" "a4paper" "left=2.5cm" "right=2.5cm" "top=3.5cm" "bottom=3.5cm") ("babel" "english") ("unicode-math" "math-style=ISO" "bold-style=ISO" "sans-style=italic" "nabla=upright" "partial=upright" "warnings-off={           % ┐
    mathtools-colon,       % │ unnötige Warnungen ausschalten
    mathtools-overbracket, % │
  }" "") ("siunitx" "locale=US" "separate-uncertainty=true" "per-mode=symbol-or-fraction" "") ("mhchem" "version=4" "math-greek=default" "text-greek=default" "") ("csquotes" "autostyle") ("placeins" "section" "below" "") ("caption" "labelfont=bf" "font=small" "width=0.9\\textwidth" "") ("biblatex" "backend=biber" "sorting=none" "") ("hyperref" "german" "unicode" "pdfusetitle" "pdfcreator={}" "pdfproducer={}" "") ("extdash" "shortcuts")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "scrartcl"
    "scrartcl12"
    "scrhack"
    "rerunfilecheck"
    "amsmath"
    "amssymb"
    "mathtools"
    "geometry"
    "fontspec"
    "setspace"
    "babel"
    "unicode-math"
    "siunitx"
    "mhchem"
    "csquotes"
    "xfrac"
    "float"
    "placeins"
    "pdflscape"
    "caption"
    "subcaption"
    "graphicx"
    "grffile"
    "booktabs"
    "microtype"
    "tikz"
    "pgfplots"
    "feynmp-auto"
    "multicol"
    "biblatex"
    "hyperref"
    "bookmark"
    "extdash"
    "wrapfig")
   (LaTeX-add-labels
    "tab:attributes")
   (LaTeX-add-bibliographies
    "lit"))
 :latex)

