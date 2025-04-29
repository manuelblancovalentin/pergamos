toggleContent = f"""<script>
        // Toggle the visibility of the collapsible content
        function toggleContent(header) {{
            const content = header.nextElementSibling;
            const triangle = header.querySelector(".triangle");
            content.style.display = content.style.display === "none" || content.style.display === ""
                                    ? "block"
                                    : "none";
            // Toggle triangle direction
            if (content.style.display === "block") {{
                triangle.classList.add('expanded');
            }} else {{
                triangle.classList.remove('expanded');
            }}
        }}
        </script>
    """


switchTab = """<script>
    // Switch between tabs
    function switchTab(tabId) {
        let contents = document.getElementsByClassName('tab-content');
        for (let content of contents) {
            content.style.display = 'none';
        }
        document.getElementById(tabId).style.display = 'block';

        let buttons = document.getElementsByClassName('tab-button');
        for (let button of buttons) {
            button.classList.remove('active');
        }
        document.querySelector(`[onclick="switchTab('${tabId}')"]`).classList.add('active');
    }
    </script>
    """

# MathJax for latex rendering
mathjax = """
    <script>
        MathJax = {
            tex: { inlineMath: [['$', '$']] },
            svg: { fontCache: 'global' },
            options: { processHtmlClass: 'math-content' },
            macros: {
                beginenumerate: "\\begin{array}{l}",
                endenumerate: "\\end{array}",
                item: "\\quad \\bullet \\quad"
            }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """


SCRIPTS = {'toggleContent': toggleContent, 'switchTab': switchTab, 'mathjax': mathjax}

