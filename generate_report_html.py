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


def _value_row_html_hits(r: dict) -> str:
    """Same layout as _value_row_html, but reads hit_rate (not power_score)
    and labels it 'contact' — hits_scanner.py's rank_slate() output uses a
    different field name deliberately (see that file's Batter class
    docstring), so this can't just reuse the HR version with a relabel."""
    return f"""
      <div class="row value">
        <div class="rank">#{r['rank']}</div>
        <div class="who">
          <div class="name">{_esc(r['player'])}</div>
          <div class="sub">{_esc(r['team'])} &middot; {_esc(r['game'])}</div>
          <div class="tags">
            <span class="tag boost">boost {_pct(r['situational_boost'])}</span>
            <span class="tag">contact {r['hit_rate']:.2f}</span>
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
                 target_date: str, top_n_shown: int = 30,
                 hits_ranked: list[dict] | None = None,
                 hits_parlays: list[list[dict]] | None = None,
                 hits_value_plays: list[dict] | None = None,
                 hits_value_parlays: list[list[dict]] | None = None) -> str:
    """hits_* params are all optional and default to None — pass nothing and
    this renders exactly the HR-only page it always has. Pass hits data too
    (as hr_scanner_auto.py now does) and a second, separately-anchored board
    renders below the HR one, reusing _row_html/_parlay_html (already
    prop-agnostic) and the new _value_row_html_hits for correct labeling."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_rows = "".join(_row_html(r) for r in ranked[:top_n_shown])
    rest_rows = "".join(_row_html(r) for r in ranked[top_n_shown:])
    value_rows = "".join(_value_row_html(r) for r in value_plays) if value_plays else '<p class="empty">No value plays surfaced today.</p>'

    has_hits = hits_ranked is not None
    if has_hits:
        hits_top_rows = "".join(_row_html(r) for r in hits_ranked[:top_n_shown])
        hits_rest_rows = "".join(_row_html(r) for r in hits_ranked[top_n_shown:])
        hits_value_rows = "".join(_value_row_html_hits(r) for r in hits_value_plays) if hits_value_plays else '<p class="empty">No value plays surfaced today.</p>'

    nav_html = f"""
  <nav class="board-nav">
    <a href="#hr-board" class="nav-pill nav-hr">&#9917; HR</a>
    {f'<a href="#hits-board" class="nav-pill nav-hits">&#127959; Hits</a>' if has_hits else ''}
  </nav>""" if has_hits else ""

    hits_section_html = f"""
  <section id="hits-board" class="board-section">
    <div class="board-divider"><span class="accent-blue">&#127959; HITS BOARD</span></div>

    <section>
      <h2><span class="accent-blue">&#9650;</span> Top Picks</h2>
      <div class="panel">{hits_top_rows if hits_top_rows else '<p class="empty">No slate loaded.</p>'}</div>
      {f'<details><summary>Show remaining {len(hits_ranked)-top_n_shown} batters</summary><div class="panel">{hits_rest_rows}</div></details>' if hits_rest_rows else ''}
    </section>

    <section>
      <h2><span class="accent-blue">&#9650;</span> Parlay Lays</h2>
      {_parlay_html(hits_parlays, css_class="hits-parlay")}
    </section>

    <section>
      <h2><span class="accent-green">&#9670;</span> Value Board — under the radar</h2>
      <div class="panel">{hits_value_rows}</div>
    </section>

    <section>
      <h2><span class="accent-green">&#9670;</span> Value Parlays</h2>
      {_parlay_html(hits_value_parlays, css_class="value-parlay")}
    </section>
  </section>""" if has_hits else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{"HR + Hits Board" if has_hits else "HR Board"} — {_esc(target_date)}</title>
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
    --blue: #5DA9E9;
    --blue-glow: rgba(93, 169, 233, 0.30);
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
  .board-nav {{ display: flex; gap: 8px; margin-top: 10px; }}
  .nav-pill {{
    display: inline-block;
    font-family: 'Oswald', sans-serif;
    font-size: 12px;
    letter-spacing: 0.03em;
    text-decoration: none;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--panel-2);
  }}
  .nav-pill.nav-hr {{ color: var(--amber); border-color: rgba(255,162,60,0.35); }}
  .nav-pill.nav-hits {{ color: var(--blue); border-color: rgba(93,169,233,0.35); }}
  main {{ max-width: 640px; margin: 0 auto; padding: 0 12px; }}
  section {{ margin-top: 28px; }}
  .board-section {{ margin-top: 40px; }}
  .board-divider {{
    text-align: center;
    font-family: 'Oswald', sans-serif;
    font-size: 13px;
    letter-spacing: 0.08em;
    padding: 10px 0;
    margin-bottom: 8px;
    border-top: 1px dashed var(--border);
    border-bottom: 1px dashed var(--border);
  }}
  h2 {{
    font-size: 15px;
    color: var(--text-dim);
    margin: 0 0 10px;
    padding-left: 2px;
  }}
  h2 .accent {{ color: var(--amber); }}
  h2 .accent-green {{ color: var(--green); }}
  h2 .accent-blue {{ color: var(--blue); }}
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
  .parlay.hits-parlay {{ border-color: rgba(93,169,233,0.35); }}
  .parlay-title {{ color: var(--amber); font-size: 12px; margin-bottom: 8px; }}
  .parlay.value-parlay .parlay-title {{ color: var(--green); }}
  .parlay.hits-parlay .parlay-title {{ color: var(--blue); }}
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
  <h1>&#9917; {"HR + Hits Board" if has_hits else "HR Board"}</h1>
  <div class="meta">{_esc(target_date)} &middot; generated {_esc(generated_at)} &middot; {len(ranked)} batters{f' &middot; {len(hits_ranked)} batters (hits)' if has_hits else ''}</div>
  {nav_html}
</header>
<main>
  <section id="hr-board">
  <div class="board-divider"><span class="accent">&#9917; HR BOARD</span></div>
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
  </section>
  {hits_section_html}
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

    # Backward-compat check: HR-only call, no hits args at all
    out_hr_only = render_html(ranked, parlays, value_plays, value_parlays, "2026-08-12")
    assert "hits-board" not in out_hr_only, "hits section should not appear when no hits data is passed"
    assert '<nav class="board-nav">' not in out_hr_only, "nav element should not render with no hits data"
    with open("/tmp/test_report_hr_only.html", "w", encoding="utf-8") as f:
        f.write(out_hr_only)
    print(f"HR-only: wrote {len(out_hr_only)} chars, correctly has no hits section")

    # HR + hits call
    hits_ranked = [{
        "player": f"Hits Player {i}", "team": "TST", "game": "AAA@BBB",
        "opp_pitcher": "Some Pitcher", "park_factor": 0.05 if i % 6 == 0 else 0.0,
        "weather_adj": 0.0, "matchup_adjustment": 0.01 if i < 30 else 0.0,
        "hit_rate": max(0.15, 0.45 - i * 0.005), "situational_boost": 0.08 if i % 4 == 0 else 0.0,
        "score": round(max(0.1, 0.5 - i * 0.005), 3), "rank": i + 1, "mlbam_id": 2000 + i,
    } for i in range(50)]
    hits_value_plays = hits_ranked[20:25]
    hits_parlays = [hits_ranked[0:3], hits_ranked[3:6]]
    hits_value_parlays = [hits_value_plays[0:3]]

    out_both = render_html(ranked, parlays, value_plays, value_parlays, "2026-08-12",
                            hits_ranked=hits_ranked, hits_parlays=hits_parlays,
                            hits_value_plays=hits_value_plays, hits_value_parlays=hits_value_parlays)
    assert "hits-board" in out_both, "hits section should appear when hits data is passed"
    assert '<nav class="board-nav">' in out_both, "nav element should render with hits data"
    assert "power 0." not in out_both.split('id="hits-board"')[1].split("Value Parlays")[0], \
        "hits Value Board should never show a 'power' label"
    assert "contact 0." in out_both, "hits Value Board should show 'contact', not 'power'"
    with open("/tmp/test_report_hr_and_hits.html", "w", encoding="utf-8") as f:
        f.write(out_both)
    print(f"HR+hits: wrote {len(out_both)} chars, hits section present with correct 'contact' labeling")
