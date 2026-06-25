project = "trxrs"
author = "TRX developers"
copyright = "2026, TRX developers"

extensions = ["myst_parser"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
}
html_title = "trxrs"

