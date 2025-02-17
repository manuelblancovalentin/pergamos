# Pergamos: Dynamic HTML Reporting for Python

**Pergamos** is a lightweight Python module for **automatically generating HTML reports** with support for:
- Markdown rendering 🗙️
- LaTeX equations using MathJax 💢
- Syntax-highlighted code blocks 🎨
- Tables from `numpy` arrays and `pandas` DataFrames 📊
- Static and interactive Matplotlib plots 📈

---

## 🚀 **Installation**
Install via `pip`:
```sh
pip install pergamos
```
For development:
```sh
pip install -e .[dev]
```

---

## 📌 **Features**
- 🗙️ **Markdown** rendering with `markdown` and `pygments`
- 🧬 **LaTeX support** via MathJax for equations
- 🎨 **Syntax-highlighted code blocks** (Python, JS, C++)
- 📊 **Tables** from lists, NumPy arrays, and Pandas DataFrames
- 📈 **Plots** using Matplotlib (both static & interactive)
- 📁 **Collapsible & Tabbed Containers** for better layout

---

## 🛠 **Usage Examples**

### **1️⃣ Creating an HTML Document**
```python
import pergamos as pg

doc = pg.Document("My Report")
doc.append(pg.Text("🚀 My Dynamic Report", tag='h1'))
doc.append(pg.Text("This is a dynamically generated report using Pergamos."))

doc.save("report.html")
```
🔹 Generates a simple **HTML report** with a title and text.

---

### **2️⃣ Adding Markdown Content**
```python
md_text = """
# Markdown Example
This is **bold**, *italic*, and `inline code`.
"""

doc.append(pg.Markdown(md_text))
```
🔹 Supports **headings, bold, italics, and inline code**.

---

### **3️⃣ Adding a Code Block with Syntax Highlighting**
```python
code = """
```python
def hello():
    print("Hello, World!")
\```
"""
doc.append(pg.Markdown(code))
```

🔹 Renders Python **syntax-highlighted** inside a styled `<pre><code>` block.

---

### **4️⃣ Rendering LaTeX Equations**
```python
doc.append(pg.Latex(r"E = mc^2", inline=True))
doc.append(pg.Latex(r"\int_a^b x^2 \,dx", inline=False))
```
🔹 Supports **inline and block LaTeX equations**.

---

### **5️⃣ Displaying Tables**
```python
import numpy as np
import pandas as pd

array_data = np.random.randint(1, 100, (5, 5))
df = pd.DataFrame(array_data, columns=["A", "B", "C", "D", "E"])

doc.append(pg.Table(array_data))  # Numpy array
doc.append(pg.Table(df))  # Pandas DataFrame
```
🔹 Supports **tables from lists, NumPy, and Pandas**.

---

### **6️⃣ Adding a Static Plot**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])

doc.append(pg.Plot(fig))
```
🔹 Renders a **static Matplotlib plot** as an image.

---

### **7️⃣ Adding an Interactive Plot**
```python
doc.append(pg.InteractivePlot(fig))
```
🔹 Uses **Mpld3** to create **interactive zoomable plots**.

---

## 🏠 **Development & Contribution**
To contribute:
1. Clone the repo:
   ```sh
   git clone https://github.com/manuelblancovalentin/pergamos.git
   cd pergamos
   ```
2. Install dependencies:
   ```sh
   pip install -e .[dev]
   ```
3. Run tests:
   ```sh
   pytest
   ```
4. Submit a pull request 🚀

---

## 📍 **License**
This project is licensed under the **MIT License**.

📌 **GitHub Repository**: [Pergamos](https://github.com/manuelblancovalentin/pergamos)

