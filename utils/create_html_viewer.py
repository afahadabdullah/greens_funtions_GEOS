#!/usr/bin/env python3
"""
Generate an interactive, standalone HTML gallery to browse all plots
produced by the Green's Function parameter tuning and post-processing.
Uses relative paths so it is fully portable.
"""

import os
import glob

def create_viewer(repo_dir=None):
    if repo_dir is None:
        # Default to repo root (parent of utils/)
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
    plots_dir = os.path.join(repo_dir, "plots")
    output_html = os.path.join(repo_dir, "plots_viewer.html")
    
    if not os.path.exists(plots_dir):
        print(f"Error: {plots_dir} does not exist.")
        return

    png_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")))
    if not png_files:
        print("No PNG files found in plots/!")
        return

    # Categorize images for easier navigation
    categories = {
        "Global Summary & Cost Breakdowns": [],
        "Validation & Verification (GF_OPT vs Baseline)": [],
        "Spatial Difference & Comparison Maps": [],
        "Atmospheric Diagnostics (DJF & JJA)": [],
        "Ocean Diagnostics (DJF & JJA)": []
    }

    for f in png_files:
        basename = os.path.basename(f)
        rel_path = f"plots/{basename}"
        
        if "Cost_Breakdown" in basename or "Warming" in basename:
            categories["Global Summary & Cost Breakdowns"].append((basename, rel_path))
        elif any(k in basename for k in ["Global_TS", "Global_SST", "Nino34", "SST_OceanOnly", "TS_Compare", "GEOS_TS"]):
            categories["Validation & Verification (GF_OPT vs Baseline)"].append((basename, rel_path))
        elif "Map_" in basename:
            categories["Spatial Difference & Comparison Maps"].append((basename, rel_path))
        elif any(k in basename for k in ["THETA", "SALT"]):
            categories["Ocean Diagnostics (DJF & JJA)"].append((basename, rel_path))
        else:
            categories["Atmospheric Diagnostics (DJF & JJA)"].append((basename, rel_path))

    # Build options grouped by optgroup
    options_html = []
    first_image = f"plots/{os.path.basename(png_files[0])}"
    
    for cat_name, items in categories.items():
        if items:
            options_html.append(f'  <optgroup label="{cat_name}">')
            for name, path in items:
                options_html.append(f'    <option value="{path}">{name}</option>')
            options_html.append('  </optgroup>')

    options_str = "\n".join(options_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GEOS-MIT Green's Function Tuning - Plot Gallery</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #0d1117;
            color: #c9d1d9;
            text-align: center;
        }}
        h1 {{
            color: #58a6ff;
            margin-bottom: 8px;
        }}
        p.subtitle {{
            color: #8b949e;
            margin-top: 0;
            margin-bottom: 24px;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        select {{
            padding: 10px 14px;
            font-size: 15px;
            width: 480px;
            max-width: 90vw;
            border-radius: 6px;
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            outline: none;
            cursor: pointer;
        }}
        select:focus {{
            border-color: #58a6ff;
        }}
        optgroup {{
            background-color: #21262d;
            color: #58a6ff;
            font-weight: bold;
        }}
        option {{
            background-color: #161b22;
            color: #c9d1d9;
            font-weight: normal;
        }}
        #plot-wrapper {{
            display: inline-block;
            background: #161b22;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            max-width: 95vw;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
        }}
        .footer {{
            margin-top: 40px;
            color: #8b949e;
            font-size: 13px;
        }}
    </style>
</head>
<body>

    <h1>GEOS-MIT Green's Function Optimization</h1>
    <p class="subtitle">Diagnostic and Verification Plot Gallery</p>
    
    <div class="controls">
        <label for="plot-select" style="margin-right: 10px; font-weight: 500;">Select Figure:</label>
        <select id="plot-select" onchange="updatePlot()">
{options_str}
        </select>
    </div>

    <div id="plot-wrapper">
        <img id="current-plot" src="{first_image}" alt="Selected Plot">
    </div>

    <div class="footer">
        NASA GEOS-MIT Coupled Climate Model &bull; Green's Function Optimization
    </div>

    <script>
        function updatePlot() {{
            var selector = document.getElementById('plot-select');
            var image = document.getElementById('current-plot');
            image.src = selector.value;
        }}
        // Keyboard navigation with arrow keys
        document.addEventListener('keydown', function(e) {{
            var select = document.getElementById('plot-select');
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                if (select.selectedIndex < select.options.length - 1) {{
                    select.selectedIndex++;
                    updatePlot();
                    e.preventDefault();
                }}
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                if (select.selectedIndex > 0) {{
                    select.selectedIndex--;
                    updatePlot();
                    e.preventDefault();
                }}
            }}
        }});
    </script>

</body>
</html>
"""
    with open(output_html, "w") as f:
        f.write(html_content)
        
    print(f"Generated portable HTML viewer: {output_html}")
    print(f"Total plots indexed: {len(png_files)}")

if __name__ == "__main__":
    create_viewer()
