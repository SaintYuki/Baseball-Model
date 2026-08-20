"""
generate_report_html.py — renders the day's ranked slate, parlays, and Value
Board into a single self-contained, mobile-friendly HTML file. No external
API calls at view-time — everything's baked in at generation time, so it
works as a static page (GitHub Pages, or just opening the file directly).

DESIGN: a ballpark scoreboard, not a generic dashboard. Dark scoreboard-
housing background, LED-amber for the main ranked numbers, a diamond-green
accent for the Value Board section so it reads as visually distinct from
"just more of the same list." Monospace for anything score-shaped so the
digits actually line up like they would on a real board.

USAGE (called from hr_scanner_auto.py automatically, or standalone):
    from generate_report_html import render_html
    html = render_html(ranked, parlays, value_plays, value_parlays, target_date)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
"""

import html as html_lib
from datetime import datetime, timezone


def _esc(s) -> str:
    return html_lib.escape(str(s))


def _pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x*100:.0f}%"


def _wx_label(x: float) -> str:
    if not x:
        return "neutral"
    sign = "+" if x > 0 else ""
    return f"{sign}{x*100:.1f}%"


def _row_html(r: dict, rank_override: int | None = None) -> str:
    rank = rank_override if rank_override is not None else r["rank"]
    matchup = r.get("matchup_adjustment", 0.0)
    matchup_html = f'<span class="mu {"pos" if matchup>0 else "neg" if matchup<0 else ""}">{matchup:+.3f}</span>' if matchup else '<span class="mu dim">n/a</span>'
    return f"""
      <div class="row">
        <div class="rank">{rank}</div>
        <div class="who">
          <div class="name">{_esc(r['player'])}</div>
          <div class="sub">{_esc(r['team'])} &middot; {_esc(r['game'])} &middot; vs {_esc(r.get('opp_pitcher') or '?')}</div>
          <div class="tags">
            <span class="tag">park {_pct(r['park_factor'])}</span>
            <span class="tag">wx {_wx_label(r['weather_adj'])}</span>
            {matchup_html}
          </div>
        </div>
        <div class="score">{r['score']:.3f}</div>
      </div>"""


def _value_row_html(r: dict) -> str:
    return f"""
      <div class="row value">
        <div class="rank">#{r['rank']}</div>
        <div class="who">
          <div class="name">{_esc(r['player'])}</div>
          <div class="sub">{_esc(r['team'])} &middot; {_esc(r['game'])}</div>
          <div class="tags">
            <span class="tag boost">boost {_pct(r['situational_boost'])}</span>
            <span class="tag">power {r['power_score']:.2f}</span>
          </div>
        </div>
        <div class="score value-score">{r['score']:.3f}</div>
      </div>"""


def _parlay_html(parlays: list, css_class: str = "") -> str:
    if not parlays:
        return '<p class="empty">No parlays available for this slate.</p>'
    blocks = []
    for i, p in enumerate(parlays, 1):
        legs = "".join(f"""
          <div class="leg">
            <span class="leg-name">{_esc(leg['player'])}</span>
            <span class="leg-detail">{_esc(leg['team'])} &middot; {_esc(leg['game'])}</span>
            <span class="leg-score">{leg['score']:.3f}</span>
          </div>""" for leg in p)
        blocks.append(f'<div class="parlay {css_class}"><div class="parlay-title">Lay {i}</div>{legs}</div>')
    return "".join(blocks)


def render_html(ranked: list[dict], parlays: list[list[dict]],
                 value_plays: list[dict], value_parlays: list[list[dict]],
                 target_date: str, top_n_shown: int = 30) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_rows = "".join(_row_html(r) for r in ranked[:top_n_shown])
    rest_rows = "".join(_row_html(r) for r in ranked[top_n_shown:])
    value_rows = "".join(_value_row_html(r) for r in value_plays) if value_plays else '<p class="empty">No value plays surfaced today.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Board — {_esc(target_date)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #14181A;
    --panel: #1C2224;
    --panel-2: #20272A;
    --border: #2A3133;
    --text: #EDEFE9;
    --text-dim: #8B9490;
    --amber: #FFA23C;
    --amber-glow: rgba(255, 162, 60, 0.35);
    --green: #6FCF97;
    --green-glow: rgba(111, 207, 151, 0.30);
    --red: #E8737A;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    background-image: radial-gradient(ellipse at top, #1A2022 0%, #14181A 60%);
    color: var(--text);
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 14px;
    line-height: 1.5;
    padding-bottom: 48px;
  }}
  h1, h2, .parlay-title {{
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  header {{
    padding: 22px 16px 18px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
    background: rgba(20, 24, 26, 0.92);
    backdrop-filter: blur(6px);
  }}
  header h1 {{
    margin: 0 0 4px;
    font-size: 22px;
    font-weight: 700;
    color: var(--amber);
    text-shadow: 0 0 18px var(--amber-glow);
  }}
  header .meta {{
    color: var(--text-dim);
    font-size: 12px;
  }}
  main {{ max-width: 640px; margin: 0 auto; padding: 0 12px; }}
  section {{ margin-top: 28px; }}
  h2 {{
    font-size: 15px;
    color: var(--text-dim);
    margin: 0 0 10px;
    padding-left: 2px;
  }}
  h2 .accent {{ color: var(--amber); }}
  h2 .accent-green {{ color: var(--green); }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .row {{
    display: grid;
    grid-template-columns: 34px 1fr auto;
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .row:last-child {{ border-bottom: none; }}
  .row .rank {{
    color: var(--text-dim);
    font-weight: 600;
    text-align: center;
  }}
  .row.value .rank {{ color: var(--green); }}
  .name {{
    font-family: 'Oswald', sans-serif;
    font-size: 14.5px;
    letter-spacing: 0.01em;
  }}
  .sub {{ color: var(--text-dim); font-size: 11px; margin-top: 1px; }}
  .tags {{ margin-top: 5px; display: flex; gap: 6px; flex-wrap: wrap; }}
  .tag {{
    font-size: 10px;
    color: var(--text-dim);
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
  }}
  .tag.boost {{ color: var(--green); border-color: rgba(111,207,151,0.35); }}
  .mu {{ font-size: 10px; padding: 1px 6px; border-radius: 4px; background: var(--panel-2); border: 1px solid var(--border); }}
  .mu.pos {{ color: var(--green); }}
  .mu.neg {{ color: var(--red); }}
  .mu.dim {{ color: var(--text-dim); }}
  .score {{
    font-size: 17px;
    font-weight: 600;
    color: var(--amber);
    text-shadow: 0 0 10px var(--amber-glow);
    min-width: 52px;
    text-align: right;
  }}
  .value-score {{ color: var(--green); text-shadow: 0 0 10px var(--green-glow); }}
  details {{ margin-top: 10px; }}
  summary {{
    cursor: pointer;
    color: var(--text-dim);
    font-size: 12px;
    padding: 8px 2px;
  }}
  summary:hover {{ color: var(--text); }}
  .parlay {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
  }}
  .parlay.value-parlay {{ border-color: rgba(111,207,151,0.35); }}
  .parlay-title {{ color: var(--amber); font-size: 12px; margin-bottom: 8px; }}
  .parlay.value-parlay .parlay-title {{ color: var(--green); }}
  .leg {{ display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; padding: 5px 0; border-bottom: 1px dashed var(--border); font-size: 12.5px; }}
  .leg:last-child {{ border-bottom: none; }}
  .leg-name {{ font-family: 'Oswald', sans-serif; }}
  .leg-detail {{ grid-column: 1; color: var(--text-dim); font-size: 10.5px; }}
  .leg-score {{ grid-row: 1 / 3; align-self: center; color: var(--amber); font-weight: 600; }}
  .empty {{ color: var(--text-dim); padding: 14px; text-align: center; font-size: 12px; }}
  footer {{ max-width: 640px; margin: 28px auto 0; padding: 0 16px; color: var(--text-dim); font-size: 10.5px; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>&#9917; HR Board</h1>
  <div class="meta">{_esc(target_date)} &middot; generated {_esc(generated_at)} &middot; {len(ranked)} batters</div>
</header>
<main>
  <section>
    <h2><span class="accent">&#9650;</span> Top Picks</h2>
    <div class="panel">{top_rows if top_rows else '<p class="empty">No slate loaded.</p>'}</div>
    {f'<details><summary>Show remaining {len(ranked)-top_n_shown} batters</summary><div class="panel">{rest_rows}</div></details>' if rest_rows else ''}
  </section>

  <section>
    <h2><span class="accent">&#9650;</span> Parlay Lays</h2>
    {_parlay_html(parlays)}
  </section>

  <section>
    <h2><span class="accent-green">&#9670;</span> Value Board — under the radar</h2>
    <div class="panel">{value_rows}</div>
  </section>

  <section>
    <h2><span class="accent-green">&#9670;</span> Value Parlays</h2>
    {_parlay_html(value_parlays, css_class="value-parlay")}
  </section>
</main>
<footer>hr-model &middot; not financial advice &middot; scores are a relative index, not a literal probability</footer>
</body>
</html>"""


if __name__ == "__main__":
    # Standalone smoke test with synthetic data
    ranked = [{
        "player": f"Test Player {i}", "team": "TST", "game": "AAA@BBB",
        "opp_pitcher": "Some Pitcher", "park_factor": 0.1 if i % 5 == 0 else 0.0,
        "weather_adj": 0.05 if i % 7 == 0 else 0.0, "matchup_adjustment": 0.02 if i < 40 else 0.0,
        "power_score": max(0.1, 1.0 - i * 0.01), "situational_boost": 0.1 if i % 3 == 0 else 0.0,
        "score": round(max(0.05, 1.1 - i * 0.01), 3), "rank": i + 1, "mlbam_id": 1000 + i,
    } for i in range(60)]
    value_plays = ranked[20:25]
    parlays = [ranked[0:3], ranked[3:6]]
    value_parlays = [value_plays[0:3]]
    out = render_html(ranked, parlays, value_plays, value_parlays, "2026-08-12")
    with open("/tmp/test_report.html", "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {len(out)} chars to /tmp/test_report.html")
