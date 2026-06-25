project = "trx-rs"
author = "TRX developers"
copyright = "2026, TRX developers"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

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
        "text": "trx-rs",
    },
}
html_title = "trx-rs"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
