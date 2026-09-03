project = "trx-rs"
author = "TRX developers"
copyright = "2026, TRX developers"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_design"
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "tasklist",
    ]

exclude_patterns = ["_build", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tee-ar-ex/trx-rs",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/trx_logo.png",
        "image_dark": "_static/trx_logo.png",
        "alt_text": "TRX",
        "link": "https://tee-ar-ex.github.io",
    },}

html_title = "trx-rs"

templates_path = ["_templates"]

html_sidebars = {
    "**": ["sidebar-nav-bs.html", "implementation-links.html"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
