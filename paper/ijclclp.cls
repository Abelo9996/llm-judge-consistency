% File: ijclclp.cls (modified - no CJK dependency)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijclclp}
\LoadClass[a4paper,10pt]{article}

% Packages
\RequirePackage{amsmath, amssymb, graphicx, fancyhdr, geometry, titlesec, natbib, booktabs}

\bibliographystyle{apalike}  

% Page Layout
\geometry{
    papersize={190mm,260mm}, top=3.7cm, bottom=2.3cm, left=2.7cm, right=2.6cm}

% Header and Footer
\fancypagestyle{firstpage}{
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{}
}
\renewcommand{\headrulewidth}{0pt}
\setlength{\headheight}{36pt}

% Define internal storage for affiliation and email
\newcommand{\AffiliationText}{}
\newcommand{\EmailText}{}

% Define \affiliation and \email commands
\newcommand{\affiliation}[1]{\renewcommand{\AffiliationText}{#1}}
\newcommand{\email}[1]{\renewcommand{\EmailText}{#1}}

% Title Formatting
\renewcommand{\maketitle}{
    \begin{center}
        \vspace*{1cm}
        {\LARGE \bfseries \@title \par}
        \vspace{1em}
        {\large \@author \par}
        \vspace{1em}
    \end{center}
    \ifx\AffiliationText\empty\else
        \footnotetext[1]{\AffiliationText \\  \ifx\EmailText\empty\else
            E-mail: \EmailText 
        \fi}
    \fi
}

% Abstract Formatting
\renewenvironment{abstract}{
    \begin{center}
        \textbf{Abstract}
    \end{center}
    \begin{quote}
}{
    \end{quote}
}

% Section Formatting
\titleformat{\section}
    {\normalfont\Large\bfseries}{\thesection.}{1em}{}

\titleformat{\subsection}
    {\normalfont\large\bfseries}{\thesubsection}{1em}{}

% Keywords Command
\newcommand{\keywords}[1]{%
    \vspace{1em}\noindent\textbf{Keywords:} #1
}

\endinput
