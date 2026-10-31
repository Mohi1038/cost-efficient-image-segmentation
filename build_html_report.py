import markdown
import os

def build_html_report():
    md_path = "report/REPORT.md"
    html_path = "report/REPORT.html"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Convert markdown to html with tables, fenced code, and attributes
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite', 'def_list', 'attr_list'])
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSL7360 Assignment 2 Report - Cost-Efficient Saliency Segmentation</title>
    
    <!-- Modern Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    
    <!-- MathJax for rendering publication-grade LaTeX equations -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>

    <style>
        :root {{
            --primary: #1e40af;
            --primary-light: #eff6ff;
            --primary-dark: #1e3a8a;
            --secondary: #059669;
            --text-main: #1e293b;
            --text-muted: #475569;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --border-color: #e2e8f0;
            --code-bg: #0f172a;
        }}

        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.7;
            color: var(--text-main);
            background-color: var(--bg-page);
            margin: 0;
            padding: 40px 20px;
            font-size: 15px;
            -webkit-font-smoothing: antialiased;
        }}
        
        .container {{
            max-width: 1040px;
            margin: 0 auto;
            background: var(--bg-card);
            padding: 55px 70px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
            border: 1px solid var(--border-color);
        }}
        
        h1, h2, h3, h4 {{
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-top: 2.2em;
            margin-bottom: 0.8em;
            page-break-after: avoid;
        }}
        
        h1 {{
            font-size: 2.4rem;
            font-weight: 800;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 20px;
            margin-top: 0;
            color: #0f172a;
            line-height: 1.25;
        }}
        
        h2 {{
            font-size: 1.6rem;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 12px;
            color: #1e3a8a;
            display: flex;
            align-items: center;
        }}
        
        h3 {{
            font-size: 1.25rem;
            color: #334155;
            margin-top: 1.6em;
        }}
        
        p, li {{
            font-size: 1.02rem;
            color: var(--text-muted);
            margin-bottom: 1.1em;
        }}

        strong {{
            color: #0f172a;
            font-weight: 600;
        }}

        /* Table Styling */
        .table-wrapper {{
            overflow-x: auto;
            margin: 28px 0;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            text-align: left;
            margin: 0;
            background: #ffffff;
            page-break-inside: avoid;
        }}
        
        th, td {{
            padding: 13px 16px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: middle;
        }}
        
        th {{
            background: #f8fafc;
            color: #0f172a;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            border-top: none;
        }}
        
        tr:nth-child(even) td {{
            background-color: #fbfcfe;
        }}
        
        tr:hover td {{
            background-color: #f1f5f9;
        }}
        
        /* Code blocks */
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: #f1f5f9;
            color: #0f172a;
            padding: 3px 6px;
            border-radius: 5px;
            font-size: 0.88em;
            border: 1px solid #e2e8f0;
        }}
        
        pre {{
            background: var(--code-bg);
            color: #f8fafc;
            padding: 20px 24px;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 24px 0;
            border: 1px solid #1e293b;
        }}
        
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
            border: none;
            font-size: 0.9em;
        }}
        
        /* Figures & Images */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin: 24px 0 10px 0;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            display: block;
            page-break-inside: avoid;
        }}

        blockquote {{
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 24px 0;
            color: #1e40af;
        }}

        blockquote p {{
            margin: 0;
            color: #1e3a8a;
            font-weight: 500;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 45px 0;
        }}
        
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
                font-size: 13px;
            }}
            .container {{
                box-shadow: none;
                border: none;
                padding: 15px;
                max-width: 100%;
            }}
            h1 {{
                font-size: 2rem;
            }}
            h2 {{
                font-size: 1.35rem;
                page-break-before: auto;
            }}
            pre, table, img, blockquote {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
    
    <script>
    // Wrap tables in table-wrapper for smooth scrolling
    document.querySelectorAll('table').forEach(tbl => {{
        const wrap = document.createElement('div');
        wrap.className = 'table-wrapper';
        tbl.parentNode.insertBefore(wrap, tbl);
        wrap.appendChild(tbl);
    }});
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated clean HTML report at {html_path}")

if __name__ == '__main__':
    build_html_report()
