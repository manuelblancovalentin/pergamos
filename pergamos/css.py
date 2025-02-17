# Import regex
import re

""" Basic CSS Style class """
class CSSStyle:
    def __init__(self, css_string=""):
        self.styles = {}  # Stores parsed rules: { selector: { property: value } }
        self.raw_strings = {}  # Stores full formatted lines (including comments)
        self.parse_css(css_string)

    def parse_css(self, css_string):
        """Parses a given CSS string to extract selectors, rules, and comments."""
        matches = re.findall(r'([^{}]+)\{([^{}]+)\}', css_string, re.DOTALL)
        for selector_string, rules_string in matches:
            selectors = [s.strip() for s in selector_string.split(',')]
            rules = {}
            raw_lines = []  # Store formatted rules with comments
            
            for line in rules_string.strip().split('\n'):
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # Skip empty lines

                # Extract key-value pairs while preserving comments
                rule_match = re.match(r'([^:]+):\s*([^;]+);', stripped_line)
                if rule_match:
                    key, value = rule_match.groups()
                    rules[key.strip()] = value.strip()
                
                # Store the original rule line (to preserve comments)
                raw_lines.append(stripped_line)

            for selector in selectors:
                if selector in self.styles:
                    self.styles[selector].update(rules)  # Merge styles if selector already exists
                    self.raw_strings[selector].extend(raw_lines)
                else:
                    self.styles[selector] = rules
                    self.raw_strings[selector] = raw_lines

    def __html__(self, tab=0):
        """Generates a formatted CSS string with optional tabulation, preserving comments."""
        indent = '\t' * tab
        css_text = ""
        
        for selector, rules in self.styles.items():
            lines = [l.strip().replace('\t',' ') for l in self.raw_strings[selector]]
            rules_str = '\n'.join(f"{indent} {line}" for line in lines)
            css_text += f"{indent}{selector} {{\n{rules_str}\n{indent}}}\n\n"
        
        return css_text.strip()

    def __repr__(self):
        return self.__html__(tab=0)

    def __add__(self, other):
        """Combines two CSSStyle objects while keeping separate selector groups and preserving comments."""
        combined = CSSStyle()
        combined.styles = {**self.styles}  # Copy current styles
        combined.raw_strings = {**self.raw_strings}  # Copy formatted rules with comments

        for selector, rules in other.styles.items():
            if selector in combined.styles:
                combined.styles[selector].update(rules)  # Merge styles for shared selectors
                combined.raw_strings[selector].extend(other.raw_strings[selector])  # Merge comments
            else:
                combined.styles[selector] = rules
                combined.raw_strings[selector] = other.raw_strings[selector]

        return combined
    
    def __len__(self):
        return len(self.styles)





