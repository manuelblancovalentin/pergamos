""" This document defines the decorators that can be used in outside code to 
        wrap everything in a pergamos context.
 """

from .objects import CollapsibleContainer, Container, ListTable, NumpyArrayTable, DataFrameTable, \
    Text, HTMLBaseElement, Document

import pergamos as pg

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

def printable(func):
    """ Decorator that makes the output of a function printable. """
    def wrapper(*args, **kwargs):
        # Let's loop thru the args and parse them
        parents = func(*args, **kwargs)
        objs = parse_serial_obj(parents)
        return objs
    return wrapper

def parse_serial_obj(obj):
    # This is the mapping:
    # - obj=dict            -> CollapsibleContainer
    # - obj=[list,tuple]    -> ListContainer
    # - obj=np.ndarray      -> ArrayContainer
    # - obj=[str|int|float] -> Text
    # - obj=?               -> Unknown
    subs = []
    if isinstance(obj, dict):
        # Check if "__options" is a key and pop it, as this is for internal use
        __options = {}
        if "__options" in obj:
            __options = obj.pop("__options")

        # Now parse options 
        collapsible = __options.get("collapsible", True)
        orientation = __options.get("orientation", "vertical")

        # CollapsibleContainer with the keys as titles
        for key, value in obj.items():
            if collapsible:
                subobj = CollapsibleContainer(key, orientation)
            else:
                # Normal container 
                subobj = Container(key, orientation)
            # Call the function recursively
            sub = parse_serial_obj(value)
            # Append 
            for s in sub:
                if hasattr(s, '__iter__'):
                    subobj.extend(s)
                else:
                    subobj.append(s)
            subs.append(subobj)

    elif isinstance(obj, (list, tuple)):
        # Here we need to check what the content of the list is, 
        # if it's another iterable (dict, list, tuple, np.ndarray, pd.DataFrame, ...)
        # or basically anything that's not [str|int|float|bool], then we need to loop
        # and call this recursively
        if all(isinstance(x, (str, int, float, bool)) for x in obj):
            # ListContainer
            subs.append(ListTable(obj))
        else:
            # Loop and call recursively
            for x in obj:
                subs.extend(parse_serial_obj(x))
    
    elif isinstance(obj, np.ndarray):
        # ArrayContainer
        subs.append(NumpyArrayTable(obj))

    elif isinstance(obj, pd.DataFrame):
        # Pandas DataFrame
        subs.append(DataFrameTable(obj))
    
    elif isinstance(obj, (str, int, float, bool)):
        # Text
        subs.append(Text(obj))
    
    elif isinstance(obj, HTMLBaseElement):
        # Just append the object directly if this is a pergamos object
        subs.append(obj)
    
    else:
        # We don't know what this is, let's see if we can print it
        try:
            subs.append(Text(str(obj)))
        except Exception as e:
            subs.append(Text(f"Unknown object {type(obj)}: {str(e)}"))

    return subs



# IPython display inline
from IPython.display import display, HTML

#We want to create an "html" function that will display the HTML representation of the object
def html(obj, theme = 'default'):
    # if obj has attr html, then we can just display it
    if hasattr(obj, 'html'):
        if isinstance(obj.html, HTMLBaseElement):
            obj = obj.html
        #_cached_html = obj.html
        # check if callable or not 
        #if callable(_cached_html):
        #    _cached_html = _cached_html()
        
        if not isinstance(obj, Document):
            # We have to add styles 
            styles = pg.THEMES[theme]
            st = "".join([kw + ' { ' + ''.join(kv) + '} ' for kw,kv in styles.raw_strings.items()])
            # And the scripts
            scripts = [pg.SCRIPTS[script] for script in obj.required_scripts]
            sc = "".join(scripts)
            _cached_html = "<DOCTYPE html><html>" + f'<style>{st}</style>\n<body>\n{obj.html}\n{sc}</body></html>'
        return display(HTML(_cached_html))