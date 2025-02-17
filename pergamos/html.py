# Typing
from .css import CSSStyle

def _HTML_HEADER(style: CSSStyle, title: str = None, tab = 0):
    indent = '\t' * (tab + 1)
    s = [f"<!DOCTYPE html>",
        f"<html>",
        _HTML_HEAD(style=style, tab=tab+1),
        indent + f"<title>{title}</title>" if title else "<!-- No title provided -->",
    ]
    return '\n'.join(s)

# HTML definitions
def _HTML_HEAD(style: CSSStyle, tab = 0):
    # Get indent for each line 
    indent = '\t' * tab
    s = [f"<head>",
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">']
    
    # Add style (if present)
    if len(style) > 0:
        s += ['<style>',
            f'{style.__html__(tab = tab+1)}',
            f'</style>']
    
    # Add closing head tag
    s += ['</head>']
    return '\n'.join([indent + line for line in s])

def _HTML_BODY(content, scripts = '', tab = 0):
    indent = '\t' * tab
    s = [indent + f"<body>"]
    s += [c.__html__(tab=tab+1) for c in content]

    # If scripts are provided, add them to the body
    if scripts and len(scripts) > 0:
        s += [indent + f"\t{script}" for script in scripts]
    s += [indent + f"</body>"]
    
    return '\n'.join(s)

def _HTML_FOOTER(tab = 0):
    indent = '\t' * tab
    return indent + "</html>"
