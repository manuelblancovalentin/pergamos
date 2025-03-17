import os

# GEt scripts path (__init__.py)
pergamos_path = os.path.dirname(__file__)

# Import css 
from . import css 

# Make sure ./css folder exists 
THEMES = {}
if os.path.exists(os.path.join(pergamos_path, 'css')):
    # Import each file in css folder for each theme
    for file in os.listdir(os.path.join(pergamos_path, 'css')):
        if file.endswith('.css'):
            # Remove .css extension to get the theme name
            theme_name = file[:-4]
            with open(os.path.join(pergamos_path, 'css', file), 'r') as f:
                css_string = f.read()
            
            # Create a CSSStyle object for each theme
            THEMES[theme_name] = css.CSSStyle(css_string)
            print(f'[INFO] - Loaded theme: {theme_name}')


# Define scripts 
from .js import SCRIPTS

# Import objects 
from .objects import Div, Text, Document, CollapsibleContainer, Table, Plot, InteractivePlot, Image, List
from .objects import Container, TabbedContainer, Markdown, Latex, Th, Tr, Td, Thead, Tbody, Tfoot, Terminal

#Tree, Video, Audio, File, Button, Link, Input, Select, TextArea, Form

# Define wrappers for decorators
from .wrappers import printable, html