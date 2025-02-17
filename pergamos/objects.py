# Basic 
import os 

# Typing
from typing import List as TypingList, Dict, Optional, Set, Union

# HTML Definitions
from . import html
# CSS defs 
from .css import CSSStyle
# scripts
from .js import SCRIPTS
# Import utils 
from . import utils

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" Matplotlib """
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

""" mpld3 for interactive plots """
import mpld3

""" Markdown """
import markdown as md 

# Import pergamos
import pergamos as pg 

# HTMLBaseElement
# │
# ├── SelfClosingElement (No children allowed)
# │      ├── <img>
# │      ├── <br>
# │      ├── <hr>
# │      ├── <meta>
# │      ├── <link>
# │
# ├── ContainerElement (Can have children)
# │      ├── <div> (Any children allowed)
# │      ├── <ul>  (Only <li> allowed)
# │      ├── <ol>  (Only <li> allowed)
# │      ├── <table> (Only <thead>, <tbody>, <tfoot> allowed)
# │      ├── <thead> (Only <tr> allowed)
# │      ├── <tr> (Only <td>, <th> allowed)
# │      ├── <form> (Only form elements allowed)


""" 
    We base our implementation of whether elements can be nested or not on the following table:
"""
# Tag	        Description	Can Contain Nested Elements?
# -----------|--------------------------------------------|---------------------------
# <div>	        Generic container for block-level content	✅ Yes (any element)
# <span>	    Inline container for styling text	        ✅ Yes (only inline elements)
# <p>	        Paragraph	                                ❌ No (only inline elements, no block-level)
# <h1>-<h6>	    Headings	                                ❌ No (only inline elements)
# <a>	        Hyperlink	                                ✅ Yes (only inline elements)
# <ul>	        Unordered list	                            ✅ Yes (<li> only)
# <ol>	        Ordered list	                            ✅ Yes (<li> only)
# <li>	        List item	                                ✅ Yes (only inline or other lists)
# <table>	    Table	                                    ✅ Yes (<tr>, <thead>, <tbody>, <tfoot> only)
# <tr>	        Table row	                                ✅ Yes (<td>, <th> only)
# <td>	        Table data cell	                            ✅ Yes (any content)
# <th>	        Table header cell	                        ✅ Yes (any content)
# <thead>	    Table head	                                ✅ Yes (<tr> only)
# <tbody>	    Table body	                                ✅ Yes (<tr> only)
# <tfoot>	    Table footer	                            ✅ Yes (<tr> only)
# <form>	    Form for user input	                        ✅ Yes (any form-related elements)
# <input>	    Input field	                                ❌ No (self-closing)
# <button>	    Clickable button	                        ✅ Yes (only inline elements)
# <label>	    Label for form elements	                    ✅ Yes (only inline elements)
# <select>	    Dropdown selection	                        ✅ Yes (<option> only)
# <option>	    Option inside <select>	                    ❌ No (only text)
# <textarea>	Multi-line text input	                    ❌ No (only text)
# <img>	        Image	                                    ❌ No (self-closing)
# <br>	        Line break	                                ❌ No (self-closing)
# <hr>	        Horizontal rule	                            ❌ No (self-closing)
# <meta>	    Metadata	                                ❌ No (self-closing)
# <link>	    Stylesheet or external resource	            ❌ No (self-closing)
# <script>	    JavaScript code	                            ✅ Yes (only script text)
# <style>	    Internal CSS	                            ✅ Yes (only CSS text)

""" 
    Base class for all HTML elements.
"""
class HTMLBaseElement:
    """Base class for all HTML elements.""" 
    def __init__(self, tag: str, 
                 attributes: Optional[Dict[str, str]] = None,
                 id: Optional[str] = None, 
                 class_name: Optional[str] = None,
                 style: Optional[CSSStyle] = None):
        self.tag = tag
        self.attributes = attributes or {}
        self.style = style  # Store CSS style specific to this element
        self.id = id
        self.class_name = class_name
        self.required_scripts: Set[str] = set()  # Scripts required by this element
    
    def _format_attributes(self) -> str:
        attr_str = ' '.join(f'{k.lstrip()}="{v.rstrip()}"' for k, v in self.attributes.items())
        if self.style and len(self.style) > 0:
            attr_str += f' style="{self.style.__html__()}"'
        return attr_str.strip()
    
    def __html__(self, tab: int = 0) -> str:
        raise NotImplementedError("Subclasses must implement __html__ method")

    @property
    def html(self) -> str:
        return self.__html__()

    def tree(self) -> str:
        return utils.generate_tree(self, repr_func=lambda e: e._repr_(add_attributes=False))

    def _repr_(self, add_attributes: bool = True):
        s = f'<{self.tag}'
        if self.id:
            s += f' id="{self.id}"'
        if self.class_name:
            s += f' class="{self.class_name}"'
        if add_attributes:
            ss = self._format_attributes().replace("\n", " ").replace("\r", "")
            s += f' {ss}'  # Add attributes if requested
        return s + '>'

    def __repr__(self):
        return self._repr_(add_attributes=True)
        

"""Represents self-closing elements (e.g., <img>, <br>)."""
class SelfClosingElement(HTMLBaseElement):
    def __init__(self, tag: str, **kwargs):
        super().__init__(tag, style=None, **kwargs)  # Self-closing elements do not accept CSS styles

    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        return f'{indent}<{self.tag} {self._format_attributes()} />'

"""Represents elements that can contain other elements (e.g., <div>, <ul>)."""
class ContainerElement(HTMLBaseElement):
    VALID_CHILDREN: Optional[TypingList[str]] = None  # Define valid child elements
    
    def __init__(self, tag: str, style: Optional[CSSStyle] = None, **kwargs):
        super().__init__(tag, style = style, **kwargs)
        self.children: TypingList[HTMLBaseElement] = []
    
    def append(self, child: HTMLBaseElement):
        if self.VALID_CHILDREN is not None and child.tag not in self.VALID_CHILDREN:
            raise ValueError(f"<{self.tag}> cannot contain <{child.tag}>")
        self.children.append(child)
        self.required_scripts.update(child.required_scripts)
    
    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        child_html = '\n'.join(child.__html__(tab + 1) for child in self.children)
        s = f'{indent}<{self.tag}'
        if self.id:
            s += f' id="{self.id}"'
        if self.class_name:
            s += f' class="{self.class_name}"'
        if len(self.attributes) > 0:
            s += f' {self._format_attributes()}'
        s += '>'
        if len(self.children) > 0:
            s += f'\n{child_html}\n{indent}'
        s += f'</{self.tag}>'
        return s
    

""" 
    HTML Elements that only have content (not children), like span, p, h1, ...
"""
class ContentElement(HTMLBaseElement):
    def __init__(self, tag: str, content: str = "", **kwargs):
        super().__init__(tag, **kwargs)
        self.content = content
    
    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        s = f'{indent}<{self.tag}'
        if len(self.attributes) > 0:
            s += f' {self._format_attributes()}'
        s += f'>{self.content}</{self.tag}>'
        return s


# Self-Closing Elements
class Img(SelfClosingElement):
    """Represents a standard HTML <img> element."""
    SUPPORTED_FORMATS = {'png', 'jpg', 'jpeg', 'svg', 'webp', 'gif'}
    
    def __init__(self, src: str, **kwargs):
        super().__init__('img', class_name='image', **kwargs)
        self.attributes['src'] = src
    
    @staticmethod
    def _encode_image(path: str) -> str:
        return utils.encode_image(path)
    
    @staticmethod
    def _encode_matplotlib(source: Union[plt.Figure, plt.Axes], fmt: str = 'png') -> str:
        """Encodes a matplotlib figure or axis as a base64 data URI in multiple formats."""
        if fmt not in Img.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Supported formats: {', '.join(Img.SUPPORTED_FORMATS)}")
        return utils.encode_matplotlib(source, fmt)

class Br(SelfClosingElement):
    def __init__(self):
        super().__init__('br')

class Hr(SelfClosingElement):
    def __init__(self):
        super().__init__('hr')

class Meta(SelfClosingElement):
    def __init__(self, attributes: Optional[Dict[str, str]] = None):
        super().__init__('meta', attributes = attributes)

class Link(SelfClosingElement):
    def __init__(self, href: str, rel: str = "stylesheet"):
        super().__init__('link', {'href': href, 'rel': rel})

# Container Elements
class Div(ContainerElement):
    def __init__(self, **kwargs):
        super().__init__('div', **kwargs)

""" Text Elements """
class Span(ContentElement):
    def __init__(self, content: str, **kwargs):
        super().__init__('span', content = content, **kwargs)

class P(ContentElement):
    def __init__(self, content:str, **kwargs):
        super().__init__('p', content = content, **kwargs)

# THIS IS A CUSTOM CLASS, NOT A STANDARD HTML ELEMENT, THAT WE WILL USE FOR TEXTS (e.g., <h1>, <h2>, <h3>)
class Text(ContentElement):
    """Represents heading and text elements like <h1> to <h6>, with optional inline modifiers."""
    VALID_MODIFIERS = {'strong', 'em', 'code', 'u', 'mark', 'small'}
    
    def __init__(self, content: str, tag: str = 'h1', 
                 modifiers: Optional[TypingList[str]] = None, **kwargs):
        assert tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span'], "Invalid heading or text tag"
        super().__init__(tag, content = content, **kwargs)
        self.modifiers = [m for m in (modifiers or []) if m in self.VALID_MODIFIERS]
    
    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        wrapped_content = self.content
        for modifier in self.modifiers:
            wrapped_content = f'<{modifier}>{wrapped_content}</{modifier}>'
        s = f'{indent}<{self.tag}'
        if len(self.attributes) > 0:
            s += f' {self._format_attributes()}'
        s += f'>{wrapped_content}</{self.tag}>'
        return s


""" List Elements """
class Ul(ContainerElement):
    VALID_CHILDREN = ['li']
    def __init__(self, **kwargs):
        super().__init__('ul', **kwargs)

class Ol(ContainerElement):
    VALID_CHILDREN = ['li']
    def __init__(self, **kwargs):
        super().__init__('ol', **kwargs)

class Li(ContainerElement):
    def __init__(self, **kwargs):
        super().__init__('li', **kwargs)
        self.content = None
    
    def __html__(self, tab: int = 0) -> str:
        # if we have content, we need to print it 
        if self.content:
            return f'<li>{self.content}</li>'
        else:
            return super().__html__(tab)


class List(ContainerElement):
    """Represents an HTML list, supporting nested lists and different data structures."""
    def __init__(self, data: Union[TypingList, np.ndarray, tuple, dict], ordered: bool = False, **kwargs):
        tag = "ol" if ordered else "ul"
        super().__init__(tag, **kwargs)  # Dynamically choose between 'ul' and 'ol'
        self.ordered = ordered
        self._process_data(data)

    def _process_data(self, data: Union[TypingList, np.ndarray, tuple, dict]):
        """Recursively processes the input data and constructs the list elements."""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif isinstance(data, tuple):
            data = list(data)
        
        if isinstance(data, dict):
            for key, value in data.items():
                li = Li()
                if isinstance(value, (list, np.ndarray, tuple, dict)):
                    li.append(Span(content=f"{key}: "))
                    li.append(List(value, ordered=self.ordered))
                else:
                    li.content = f"{key}: {value}"
                self.append(li)

        elif isinstance(data, list):
            for item in data:
                li = Li()
                if isinstance(item, (list, np.ndarray, tuple, dict)):
                    li.append(List(item, ordered=self.ordered))
                else:
                    li.content = str(item)
                self.append(li)


""" Table Elements """
class Table(ContainerElement):
    """Base Table class for rendering tables."""
    VALID_CHILDREN = ['thead', 'tbody', 'tfoot', 'tr']
    
    def __init__(self, **kwargs):
        super().__init__('table', class_name='table', **kwargs)

    @staticmethod
    def from_data(data: TypingList[TypingList[str]]):
        """Generates a table from a list of lists."""
        if isinstance(data, list) and all(isinstance(row, list) for row in data):
            if all(isinstance(cell, str) for row in data for cell in row):
                return ListTable(data)
            elif all(isinstance(cell, (int, float)) for row in data for cell in row):
                return ListTable(data)
        elif isinstance(data, np.ndarray):
            return NumpyArrayTable(data)
        elif isinstance(data, pd.DataFrame):
            return DataFrameTable(data)
        else:
            raise TypeError(f'Invalid data type for table: {type(data)}')

class Tr(ContainerElement):
    VALID_CHILDREN = ['td', 'th']
    def __init__(self):
        super().__init__('tr')

class Td(ContainerElement):
    def __init__(self, content: Optional[str] = ""):
        super().__init__('td')
        self.content = content
    
    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        return f'{indent}<{self.tag}>{self.content}</{self.tag}>'

class Th(ContainerElement):
    def __init__(self, content: Optional[str] = ""):
        super().__init__('th')
        self.content = content
    
    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        return f'{indent}<{self.tag}>{self.content}</{self.tag}>'

class Thead(ContainerElement):
    VALID_CHILDREN = ['tr']
    def __init__(self):
        super().__init__('thead')

class Tbody(ContainerElement):
    VALID_CHILDREN = ['tr']
    def __init__(self):
        super().__init__('tbody')

class Tfoot(ContainerElement):
    VALID_CHILDREN = ['tr']
    def __init__(self):
        super().__init__('tfoot')


""" List of lists table """
class ListTable(Table):
    """Handles rendering lists of lists into an HTML table."""
    def __init__(self, data: TypingList[TypingList[str]]):
        super().__init__()
        tbody = Tbody()
        for row_data in data:
            row = Tr()
            for cell in row_data:
                row.append(Td(content=str(cell)))
            tbody.append(row)
        self.append(tbody)

""" Numpy Array table is actually a div """
class NumpyArrayTable(Div):
    """Handles rendering numpy arrays into an HTML table."""
    def __init__(self, data: np.ndarray, subshape=None, cumshape=(), id : Optional[str] = None, **kwargs):
        super().__init__(id=id or "custom-container", **kwargs)
        self._generate_numpy_table(data, subshape = subshape, cumshape = cumshape)
    
    def _generate_numpy_table(self, array: np.ndarray, subshape=None, cumshape=(), **kwargs):
        if subshape is None:
            subshape = array.shape
        
        if len(subshape) > 2:
            for i in range(subshape[-1]):
                self._generate_numpy_table(array[..., i], subshape[:-1], cumshape=(i,) + cumshape)
                #self.append(NumpyArrayTable(array[..., i], subshape[:-1], cumshape=(i,) + cumshape))
        
        elif len(subshape) == 2:
            container = Div(class_name='container horizontal-layout', attributes={'style': "margin-bottom: 10px;"})
            if cumshape:
                # content will be a span
                index_header = Div(class_name='header')
                index_header.append(Span(content=f"Index: (...,{','.join(map(str, cumshape))})"))
                container.append(index_header)
            table = Table()
            tbody = Tbody()
            for i in range(subshape[0]):
                row = Tr()
                for j in range(subshape[1]):
                    row.append(Td(content=str(array[i, j])))
                tbody.append(row)
            table.append(tbody)
            container.append(table)
            self.append(container)

        elif len(subshape) == 1:
            container = Div(class_name='container horizontal-layout', attributes={'style': "margin-bottom: 10px;"})
            if cumshape:
                index_header = Div(class_name='header')
                index_header.append(Span(content=f"Index: (...,{','.join(map(str, cumshape))})"))
                container.append(index_header)
            table = Table()
            tbody = Tbody()
            row = Tr()
            for i in range(subshape[0]):
                row.append(Td(content=str(array[i])))
            tbody.append(row)
            table.append(tbody)
            container.append(table)
            self.append(container)

""" Table with dataframe input """
class DataFrameTable(Table):
    """Handles rendering pandas DataFrames into an HTML table with headers."""
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        thead = Thead()
        header_row = Tr()
        for col in df.columns:
            header_row.append(Th(content=str(col)))
        thead.append(header_row)
        self.append(thead)
        tbody = Tbody()
        for _, row_data in df.iterrows():
            row = Tr()
            for cell in row_data:
                row.append(Td(content=str(cell)))
            tbody.append(row)
        self.append(tbody)


""" Container (Custom) just to plot some elements gathered in groups """
class Container(Div):
    """Represents a collapsible section that can contain other HTML elements."""
    def __init__(self, title: str = None, layout: str = "vertical", id: Optional[str] = None, **kwargs):
        super().__init__(id=id or "custom-container", class_name="collapsible-container", **kwargs)
        self.layout = layout
        self.title = title

        # If title, add header 
        if title:
            # Header (clickable, contains title and triangle)
            self.header = Div(class_name="header")
            self.header.append(Span(title))  # Title text
        
        # Content (container for user elements)
        self.wrapper = Div(class_name=f"{layout}-wrapper")
        self.content_div = Div(class_name="content")
        self.content_div.append(self.wrapper)

        # Build structure (using super, cause we modified the append method locally)
        if title:
            super().append(self.header)
        super().append(self.content_div)
        

""" Collapsible container """
class CollapsibleContainer(Container):
    """Represents a collapsible section that can contain other HTML elements."""
    def __init__(self, title: str, layout: str = "vertical", id: Optional[str] = None, **kwargs):
        
        # Use title = None, because we will create the header here
        super().__init__(title = None, id = id, layout = layout, **kwargs)
        
        # Modify the header (clickable, contains title and triangle)
        self.header = Div(class_name="header", attributes={"onclick": "toggleContent(this)"})
        self.header.append(Div(class_name="triangle"))  # Triangle for toggle icon
        self.header.append(Span(title))  # Title text

        # Content (container for user elements)
        self.wrapper = Div(class_name=f"{layout}-wrapper")
        self.content_div = Div(class_name="content")
        self.content_div.append(self.wrapper)

        # Add required function for toggling
        self.required_scripts.add("toggleContent")  # Add script requirement

        # Restart the children (otherwise the list will be out of order because of the super call)
        self.children = []

        # Build structure (using super, cause we modified the append method locally)
        super().append(self.header)
        super().append(self.content_div)

    def append(self, element: HTMLBaseElement):
        """Appends content to the wrapper div."""
        self.wrapper.append(element)
        self.required_scripts.update(element.required_scripts)  # Merge scripts
    


""" Image """
class Image(ContainerElement):
    """A wrapper around Img to support additional functionality like embedding and displaying headers."""
    def __init__(self, source: Union[str, plt.Figure, plt.Axes], embed: bool = False, **kwargs):
        super().__init__('div', id='image-container', **kwargs)
        
        # Parse if this is a string (path) or a matplotlib object
        attributes = {}
        if isinstance(source, str):
            if embed:
                img_element = Img(self._encode_image(source))
            else:
                img_element = Img(source)
            # The header must be the same width as the image
            # from source 
            
            # Load the image
            if source.startswith("http"):
                # Load from URL
                raise NotImplementedError("Loading images from URLs is not yet supported.")
            
            elif os.path.isfile(source):
                image = mpimg.imread(source)
                # Get the dimensions of the image
                height, width, color_channels = image.shape
                attributes['style'] = f"width: {width}px;"


            header = Div(class_name="header")
            header.append(Span(f"Image: {source}"))
            self.append(header)
        
        elif isinstance(source, (plt.Figure, plt.Axes)):
            img_element = Img(self._encode_matplotlib(source))
            # Get the width 
            width = source.get_figwidth()
            attributes['style'] = f"width: {width}px;"
            # Close the figure if it's a temporary one
            if isinstance(source, plt.Figure):
                plt.close(source)
        else:
            raise TypeError("Unsupported image source type. Must be a file path, URL, or matplotlib figure/axis.")
        self.append(img_element)
        self.attributes.update(attributes)
    
    def _encode_image(self, path: str) -> str:
        return Img._encode_image(path)

    def _encode_matplotlib(self, source: Union[plt.Figure, plt.Axes]) -> str:
        return Img._encode_matplotlib(source)


class Plot(Img):
    """Handles rendering static Matplotlib plots as images."""
    def __init__(self, source: Union[plt.Figure, plt.Axes], fmt: str = 'png', **kwargs):
        if fmt not in Img.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Supported formats: {', '.join(Img.SUPPORTED_FORMATS)}")
        img_src = Img._encode_matplotlib(source, fmt)
        # Close the figure if it's a temporary one
        if isinstance(source, plt.Figure):
            plt.close(source)
        super().__init__(img_src, **kwargs)

class InteractivePlot(ContainerElement):
    """Embeds an interactive Matplotlib plot using mpld3."""
    def __init__(self, figure: plt.Figure, **kwargs):
        super().__init__('div', class_name='interactive-plot', **kwargs)
        self.figure = figure
        self.script = self._generate_mpld3_script()

    def _generate_mpld3_script(self) -> str:
        """Generates an interactive mpld3 script from a Matplotlib figure."""
        return mpld3.fig_to_html(self.figure)

    def __html__(self, tab: int = 0) -> str:
        indent = '\t' * tab
        return f'{indent}<div class="interactive-plot">{self.script}</div>'


class TabbedContainer(Div):
    """A container with multiple tabs, allowing tabbed navigation."""
    
    def __init__(self, tabs: Dict[str, HTMLBaseElement], **kwargs):
        """
        :param tabs: A dictionary where keys are tab names and values are content elements.
        """
        super().__init__(class_name="tabbed-container", **kwargs)
        
        self.tabs = tabs
        self.tab_headers = Div(class_name="tab-headers")
        self.tab_contents = Div(class_name="tab-contents")

        for index, (tab_name, content) in enumerate(tabs.items()):
            tab_id = f"tab-{index}"
            button = Div(class_name="tab-button", attributes={"onclick": f"switchTab('{tab_id}')"})
            button.append(Span(content=tab_name))

            content_div = Div(class_name="tab-content", attributes={"id": tab_id})
            content_div.append(content)

            # Set first tab as active
            if index == 0:
                button.attributes["class"] = button.attributes.get("class", "") + " active"
                content_div.attributes["style"] = "display: block;"
            else:
                content_div.attributes["style"] = "display: none;"

            self.tab_headers.append(button)
            self.tab_contents.append(content_div)

        self.append(self.tab_headers)
        self.append(self.tab_contents)

        # Ensure JavaScript is included
        self.required_scripts.add("switchTab")


class Markdown(Div):
    """A div that renders Markdown content as HTML with syntax highlighting."""
    def __init__(self, text: str, **kwargs):
        super().__init__(class_name="markdown-body", **kwargs)
        self.text = text
        self.rendered_html = self._convert_markdown(text)

        content_wrapper = Div(class_name="md-content")
        content_wrapper.append(RawHTML(self.rendered_html))

        self.append(content_wrapper)

    @staticmethod
    def _convert_markdown(text: str) -> str:
        """Converts markdown text to HTML with syntax highlighting."""
        return md.markdown(text, extensions=["extra", "codehilite", "fenced_code"])


class RawHTML(Div):
    """A wrapper to inject raw HTML into the document."""
    
    def __init__(self, html: str, **kwargs):
        super().__init__(**kwargs)
        self.html_content = html

    def __html__(self, tab: int = 0) -> str:
        indent = "\t" * tab
        return f"{indent}{self.html_content}"


class Latex(Div):
    """A div that renders LaTeX equations using MathJax."""
    
    def __init__(self, latex_text: str, inline: bool = False, **kwargs):
        """
        :param latex_text: The LaTeX string to be rendered.
        :param inline: If True, render as an inline equation; otherwise, block.
        """
        super().__init__(class_name="latex-equation", **kwargs)
        
        if inline:
            wrapped_text = f"${latex_text}$" # Use $...$ for inline math
        else:
            wrapped_text = f"\\[ {latex_text} \\]" # Use \[...\] for block math
        
        self.append(RawHTML(wrapped_text))

        # Ensure MathJax is included
        self.required_scripts.add("mathjax")

"""
    HTML Document custom class
"""
class Document:
    """Represents an entire HTML document, using predefined HTML rendering functions."""
    def __init__(self, title: str = "Document", theme: str = "default"):
        self.title = title
        self.children: TypingList[HTMLBaseElement] = []
        self.styles = pg.THEMES[theme] if theme in pg.THEMES else CSSStyle()  # Load theme-based styles
        self.required_scripts: Set[str] = set()
    
    def append(self, element: HTMLBaseElement):
        self.children.append(element)
        if element.style:
            self.styles += element.style  # Merge styles dynamically
        self.required_scripts.update(element.required_scripts)  # Merge scripts
    
    def __html__(self, tab: int = 0) -> str:
        # get scripts 
        scripts = [SCRIPTS[script] for script in self.required_scripts]
        return '\n'.join([
            html._HTML_HEADER(style=self.styles, title=self.title, tab=tab),
            html._HTML_BODY(content=self.children, tab=tab, scripts = scripts),
            html._HTML_FOOTER(tab=tab)
        ])

    def tree(self) -> str:
        return utils.generate_tree(self, repr_func=lambda e: e._repr_(add_attributes=False))
    
    @property
    def html(self) -> str:
        return self.__html__()

    def _repr_(self, add_attributes: bool = True):
        return f"<Document title='{self.title}'>"

    def __repr__(self):
        return self._repr_(add_attributes=True)

    # Save to document 
    def save(self, filename: str):
        with open(filename, 'w') as f:
            f.write(self.__html__())