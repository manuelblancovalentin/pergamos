# Typing
from typing import Optional, Callable, Union

# Encoding
import base64
import io

# Matplotlib
import matplotlib.pyplot as plt

"""Generates a tree representation of an element and its children using a custom representation function."""
def generate_tree(element, level: int = 0, prefix: str = "", is_last: bool = True, repr_func: Optional[Callable] = None) -> str:
    """Generates a tree representation of an element and its children using a custom representation function."""
    connector = "└── " if is_last else "├── "
    # if level is 0, no need for a connector
    connector = "" if level == 0 else connector
    node_repr = repr_func(element) if repr_func else str(element)
    tree_str = f"{prefix}{connector}{node_repr}"
    
    if hasattr(element, 'children') and isinstance(element.children, list):
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(element.children):
            tree_str += f"\n{generate_tree(child, level + 1, new_prefix, i == len(element.children) - 1, repr_func)}"
    
    if hasattr(element, 'content') and isinstance(element.content, str) and element.content.strip():
        content_prefix = prefix + ("    " if is_last else "│   ")
        tree_str += f"\n{content_prefix}└── {element.content}"
    
    return tree_str

""" Encoding of images """
def encode_matplotlib(source: Union[plt.Figure, plt.Axes], fmt: str = 'png') -> str:
    """Encodes a matplotlib figure or axis as a base64 data URI in multiple formats."""
    buffer = io.BytesIO()
    if isinstance(source, plt.Axes):
        source = source.get_figure()
    source.savefig(buffer, format=fmt, bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


def encode_image(path: str) -> str:
    """Encodes an image file as a base64 data URI."""
    try:
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        raise ValueError(f"Image file not found: {path}")