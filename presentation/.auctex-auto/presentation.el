(TeX-add-style-hook
 "presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "8pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "ngerman") ("unicode-math" "math-style=ISO" "bold-style=ISO" "sans-style=italic" "nabla=upright" "partial=upright" "warnings-off={           % ┐
    mathtools-colon,       % │ unnötige Warnungen ausschalten
    mathtools-overbracket, % │
  }" "") ("siunitx" "locale=DE" "separate-uncertainty=true" "per-mode=symbol-or-fraction" "") ("csquotes" "autostyle") ("biblatex" "backend=biber" "style=authoryear" "autocite=inline" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "amsmath"
    "amssymb"
    "mathtools"
    "fontspec"
    "babel"
    "unicode-math"
    "siunitx"
    "csquotes"
    "xfrac"
    "graphicx"
    "tikz"
    "pgfplots"
    "feynmp-auto"
    "caption"
    "biblatex")
   (LaTeX-add-bibliographies))
 :latex)

