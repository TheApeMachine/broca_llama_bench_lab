from core.infra.constants import BRAND, BRAND_BG, BRAND_DEEP, BRAND_SOFT

# The following fragments are defined here and imported by sibling modules
# ``core.tui.state`` (StatePanel), ``core.tui.systems`` (SystemsMatrix), and
# ``core.tui.components`` (placeholder lines and activity-log coloring).

# Shared CSS fragment for bordered side panels (Textual widget body, indented).
_CSS_BRAND_PANEL_BODY = f"""        
    border: round {BRAND} 70%;
    padding: 0 1;
    height: auto;
    margin-bottom: 1;
"""

# Placeholder line for empty rich panels.
_DIM_EM_DASH = "[dim]—[/dim]"

_ACTIVITY_LOG_LEVEL_MARKUP = {
    "DEBUG": "dim",
    "INFO": "white",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}

CSS = f"""
Screen {{
    layout: vertical;
    background: {BRAND_BG};
}}
Header {{
    background: {BRAND_DEEP};
    color: $text;
}}
Footer {{
    background: {BRAND_DEEP};
}}
#main {{
    height: 1fr;
}}
#left, #right {{
    width: 34;
    min-width: 30;
    padding: 1;
    border-right: solid {BRAND} 40%;
}}
#right {{
    border-right: none;
    border-left: solid {BRAND} 40%;
}}
#center {{
    width: 1fr;
    padding: 0 1;
}}
#chatlog {{
    height: 1fr;
    border: round {BRAND} 70%;
}}
#input {{
    height: 3;
    border: round {BRAND_SOFT} 80%;
}}
#streaming {{
    height: auto;
    max-height: 8;
    padding: 0 1;
    color: $text-muted;
}}
#status {{
    height: 1;
    background: {BRAND} 25%;
    color: $text;
    padding: 0 1;
}}
#activity {{
    height: 8;
    border: round {BRAND} 40%;
}}
Sparkline {{
    height: 3;
    margin-bottom: 1;
}}
Sparkline > .sparkline--max-color {{
    color: {BRAND};
}}
Sparkline > .sparkline--min-color {{
    color: {BRAND_SOFT};
}}
"""
