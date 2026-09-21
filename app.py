import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re
import os
import json
from datetime import datetime, timedelta
import time
import pytz
import math
import google.generativeai as genai
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import textwrap
# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════
# REALTIME PLOTLY HELPER
# ═══════════════════════════════════════════════════════════════
def render_plotly_realtime(fig, height=420, haptic=True):
    """
    Embed Plotly.js ala Stockbit (v3):
    - Tap + geser = vline + tooltip follow jari realtime
    - Haptic vibration tiap index berubah (Android)
    - Legend HTML manual di bawah chart (anti-crop di semua browser)
    - Auto-hide setelah 2.5 detik idle
    """
    if fig is None:
        return

    fig_json = fig.to_json()
    haptic_js = "true" if haptic else "false"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            html, body {{
                margin: 0; padding: 0;
                background: #0f1116;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                overflow: hidden;
            }}
            #wrapper {{ position: relative; width: 100%; height: {height}px; }}
            #chart {{ width: 100%; height: {height}px; touch-action: pan-y; }}
            #tooltip {{
                position: absolute;
                top: 8px; left: 8px;
                background: rgba(30, 41, 59, 0.96);
                color: #e2e8f0;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 11px;
                line-height: 1.5;
                pointer-events: none;
                display: none;
                z-index: 999;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                border-left: 3px solid #a855f7;
                max-width: 68%;
            }}
            #tooltip b {{ color: #a855f7; font-size: 12px; }}
            #tooltip .row {{ margin-top: 3px; display: flex; align-items: center; }}
            #tooltip .dot {{
                display: inline-block;
                width: 7px; height: 7px;
                border-radius: 50%;
                margin-right: 6px;
                flex-shrink: 0;
            }}
            #legend {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 6px 14px;
                padding: 8px 10px 10px 10px;
                margin: 0;
                background: #0f1116;
                font-size: 11px;
                color: #94a3b8;
            }}
            #legend .item {{
                display: inline-flex;
                align-items: center;
                white-space: nowrap;
                cursor: pointer;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
                padding: 2px 6px;
                border-radius: 4px;
                transition: opacity 0.15s, background 0.15s;
            }}
            #legend .item.hidden {{
                opacity: 0.30;
                text-decoration: line-through;
            }}
            #legend .item:active {{
                background: rgba(168, 85, 247, 0.20);
            }}
            #legend .swatch {{
                display: inline-block;
                width: 10px; height: 10px;
                border-radius: 2px;
                margin-right: 5px;
                flex-shrink: 0;
            }}
        </style>
    </head>
    <body>
        <div id="wrapper">
            <div id="chart"></div>
            <div id="tooltip"></div>
        </div>
        <div id="legend"></div>
        <script>
            (function() {{
                var figData = {fig_json};
                var HAPTIC = {haptic_js};

                if (figData.layout) {{
                    figData.layout.hovermode = false;
                    figData.layout.showlegend = false;
                }}

                var config = {{
                    displayModeBar: false,
                    responsive: true,
                    scrollZoom: false,
                    displaylogo: false
                }};

                var gd = document.getElementById('chart');
                var tooltip = document.getElementById('tooltip');
                var legendDiv = document.getElementById('legend');

                Plotly.newPlot(gd, figData.data, figData.layout, config).then(function() {{
                    // ── Bangun legend HTML dari trace (dengan interaksi) ──
                    var legendHtml = '';
                    for (var i = 0; i < gd._fullData.length; i++) {{
                        var t = gd._fullData[i];
                        var name = t.name || ('Series ' + i);
                        var color = (t.line && t.line.color) ||
                                    (t.marker && t.marker.color) ||
                                    '#a855f7';
                        legendHtml += '<span class="item" data-idx="' + i + '">' +
                                      '<span class="swatch" style="background:' + color + '"></span>' +
                                      name + '</span>';
                    }}
                    legendDiv.innerHTML = legendHtml;

                    // ── Click handler: toggle trace visibility ──
                    var hiddenTraces = {{}};
                    legendDiv.querySelectorAll('.item').forEach(function(el) {{
                        el.addEventListener('click', function() {{
                            var idx = parseInt(el.getAttribute('data-idx'));
                            var nowHidden = !hiddenTraces[idx];

                            // Plotly: 'legendonly' = hidden, true = visible
                            var vis = nowHidden ? 'legendonly' : true;
                            Plotly.restyle(gd, {{ visible: vis }}, [idx]);

                            hiddenTraces[idx] = nowHidden;
                            if (nowHidden) {{
                                el.classList.add('hidden');
                            }} else {{
                                el.classList.remove('hidden');
                            }}

                            // Haptic feedback (Android only)
                            if (HAPTIC && navigator.vibrate) {{
                                try {{ navigator.vibrate(10); }} catch(e) {{}}
                            }}
                        }});
                    }});

                    // ── Ambil xValues ──
                    var xValues = null;
                    for (var i = 0; i < gd._fullData.length; i++) {{
                        var xd = gd._fullData[i].x;
                        if (xd && xd.length > 0) {{
                            xValues = xd;
                            break;
                        }}
                    }}
                    if (!xValues || xValues.length === 0) {{
                        console.error('[Bandarmology] No x values');
                        return;
                    }}

                    // ── y per trace ──
                    var yPerTrace = [];
                    var namePerTrace = [];
                    var colorPerTrace = [];
                    for (var i = 0; i < gd._fullData.length; i++) {{
                        var t = gd._fullData[i];
                        yPerTrace.push(t.y || []);
                        namePerTrace.push(t.name || ('Series ' + i));
                        var c = (t.line && t.line.color) ||
                                (t.marker && t.marker.color) || '#a855f7';
                        colorPerTrace.push(c);
                    }}

                    // ── Shape vline ──
                    var initX = xValues[0];
                    var baseShapes = (gd.layout.shapes || []).slice();
                    baseShapes.push({{
                        type: 'line',
                        xref: 'x', x0: initX, x1: initX,
                        yref: 'paper', y0: 0, y1: 1,
                        line: {{ color: '#a855f7', width: 2, dash: 'dash' }},
                        opacity: 0
                    }});
                    Plotly.relayout(gd, {{ shapes: baseShapes }});
                    var V_IDX = baseShapes.length - 1;

                    function pixelToIndex(clientX) {{
                        var rect = gd.getBoundingClientRect();
                        var fl = gd._fullLayout;
                        var plotLeft = fl.margin.l;
                        var plotRight = rect.width - fl.margin.r;
                        var plotWidth = plotRight - plotLeft;
                        if (plotWidth <= 0) return 0;
                        var relX = clientX - rect.left - plotLeft;
                        var ratio = relX / plotWidth;
                        ratio = Math.max(0, Math.min(1, ratio));
                        return Math.round(ratio * (xValues.length - 1));
                    }}

                    function vibrate() {{
                        if (!HAPTIC || !navigator.vibrate) return;
                        try {{ navigator.vibrate(6); }} catch(e) {{}}
                    }}

                    var lastIdx = -1;
                    var rafPending = false;
                    var pendingX = null;
                    var active = false;
                    var hideTimer = null;

                    function draw(clientX) {{
                        var idx = pixelToIndex(clientX);
                        if (idx < 0) idx = 0;
                        if (idx >= xValues.length) idx = xValues.length - 1;
                        var xVal = xValues[idx];

                        var upd = {{}};
                        upd['shapes[' + V_IDX + '].x0'] = xVal;
                        upd['shapes[' + V_IDX + '].x1'] = xVal;
                        upd['shapes[' + V_IDX + '].opacity'] = 1;
                        Plotly.relayout(gd, upd);

                        function formatKMB(val, name) {{
                            if (val === null || val === undefined || isNaN(val)) return 'N/A';
                            var n = name ? name.toLowerCase() : '';
                            var isPrice = n.indexOf('price') !== -1 || n.indexOf('harga') !== -1;
                            var absVal = Math.abs(val);
                            if (isPrice && absVal < 1000000) {{
                                return val.toLocaleString('id-ID', {{ maximumFractionDigits: 2 }});
                            }}
                            var sign = val < 0 ? '-' : '';
                            var res = '';
                            if (absVal >= 1e9) {{
                                res = sign + (absVal / 1e9).toFixed(2).replace(/\.00$/, '') + 'B';
                            }} else if (absVal >= 1e6) {{
                                res = sign + (absVal / 1e6).toFixed(2).replace(/\.00$/, '') + 'M';
                            }} else if (absVal >= 1e3) {{
                                res = sign + (absVal / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
                            }} else {{
                                res = sign + absVal.toLocaleString('id-ID', {{ maximumFractionDigits: 2 }});
                            }}
                            if (n.indexOf('accum') !== -1 || n.indexOf('dist') !== -1 || n.indexOf('lot') !== -1) {{
                                res += ' Lot';
                            }}
                            return res;
                        }}

                        var lines = ['<b>' + xVal + '</b>'];
                        for (var i = 0; i < yPerTrace.length; i++) {{
                            // Skip trace yang di-hide via legend
                            if (hiddenTraces[i]) continue;

                            var yv = yPerTrace[i][idx];
                            if (typeof yv === 'number' && !isNaN(yv)) {{
                                var fv = formatKMB(yv, namePerTrace[i]);
                                lines.push(
                                    '<div class="row"><span class="dot" style="background:' +
                                    colorPerTrace[i] + '"></span>' +
                                    namePerTrace[i] + ': ' + fv + '</div>'
                                );
                            }}
                        }}
                        tooltip.innerHTML = lines.join('');
                        tooltip.style.display = 'block';

                        var rect = gd.getBoundingClientRect();
                        var tw = tooltip.offsetWidth || 140;
                        var left = clientX - rect.left + 14;
                        if (left + tw > rect.width - 4) {{
                            left = clientX - rect.left - tw - 14;
                        }}
                        if (left < 4) left = 4;
                        tooltip.style.left = left + 'px';

                        if (idx !== lastIdx) {{
                            vibrate();
                            lastIdx = idx;
                        }}

                        if (hideTimer) clearTimeout(hideTimer);
                        hideTimer = setTimeout(function() {{
                            tooltip.style.display = 'none';
                            var u = {{}};
                            u['shapes[' + V_IDX + '].opacity'] = 0;
                            Plotly.relayout(gd, u);
                        }}, 2500);
                    }}

                    function throttled(clientX) {{
                        pendingX = clientX;
                        if (!rafPending) {{
                            rafPending = true;
                            requestAnimationFrame(function() {{
                                rafPending = false;
                                if (pendingX !== null) {{
                                    draw(pendingX);
                                    pendingX = null;
                                }}
                            }});
                        }}
                    }}

                    gd.addEventListener('touchstart', function(e) {{
                        active = true;
                        lastIdx = -1;
                        if (hideTimer) clearTimeout(hideTimer);
                        draw(e.touches[0].clientX);
                    }}, {{ passive: true }});

                    gd.addEventListener('touchmove', function(e) {{
                        if (!active) return;
                        throttled(e.touches[0].clientX);
                    }}, {{ passive: true }});

                    gd.addEventListener('touchend', function() {{ active = false; }}, {{ passive: true }});
                    gd.addEventListener('touchcancel', function() {{ active = false; }}, {{ passive: true }});

                    gd.addEventListener('mousemove', function(e) {{
                        if (e.buttons === 0) {{
                            if (hideTimer) clearTimeout(hideTimer);
                            throttled(e.clientX);
                        }}
                    }});

                    window.addEventListener('resize', function() {{
                        Plotly.Plots.resize(gd);
                    }});
                }});
            }})();
        </script>
    </body>
    </html>
    """
        # ── Hitung tinggi iframe dinamis ──
    # Legend HTML bisa wrap ke beberapa baris tergantung jumlah trace & lebar layar
    # Di iOS (layar sempit) worst case ~2-3 item per baris
    n_traces = len(fig.data)
    legend_rows = max(2, (n_traces + 2) // 3)   # ceil(n/3), min 2 baris
    legend_h = legend_rows * 28 + 20            # ~28px per baris + padding
    iframe_h = height + legend_h + 10           # +10 buffer iOS safe area

    components.html(html, height=iframe_h, scrolling=False)
def render_sankey_interactive(fig, height=520):
    """
    Sankey interaktif lengkap (defensive):
    - Chip per-node (YP-B / YP-S)
    - Event delegation — chip selalu clickable
    - Highlight arah flow
    - Gradient link (best-effort) + fallback solid
    - Toggle Volume / Value
    """
    if fig is None:
        return

    try:
        labels = list(fig.data[0].node.label or [])
    except Exception:
        labels = []

    try:
        n_buyers = int(fig.layout.meta.get('n_buyers', 0)) if fig.layout.meta else 0
    except Exception:
        n_buyers = 0

    import re as _re

    def _clean_label(s):
        s = _re.sub(r'\s*\([^)]*\)', '', str(s)).strip()
        s = _re.sub(r'^[\d\.,]+\s*[MBK]?\s*', '', s).strip()
        s = _re.sub(r'\s*[\d\.,]+\s*[MBK]?$', '', s).strip()
        return s

    broker_counts = {}
    for lb in labels:
        c = _clean_label(lb)
        if c:
            broker_counts[c] = broker_counts.get(c, 0) + 1

    chips_html = ""
    for i, lb in enumerate(labels):
        c = _clean_label(lb)
        if not c:
            continue
        side = "B" if i < n_buyers else "S"
        suffix = f"-{side}" if broker_counts.get(c, 0) > 1 else ""
        chips_html += (
            f'<button class="chip" data-node-idx="{i}" data-broker="{c}" '
            f'data-side="{side}">{c}{suffix}</button>'
        )

    fig_json = fig.to_json()
    n_buyers_js = n_buyers

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            html, body {{ margin:0; padding:0; background:#0f1116;
                font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                overflow:hidden; }}
            #chart {{ width:100%; height:{height}px; }}
            #mode-toggle {{ display:flex; gap:6px; justify-content:center;
                padding:6px 0 2px 0; background:#0f1116; }}
            .mode-btn {{ background:#1e293b; color:#94a3b8;
                border:1px solid #334155; border-radius:16px;
                padding:6px 16px; font-size:12px; font-weight:600;
                cursor:pointer; user-select:none;
                -webkit-tap-highlight-color:transparent; transition:all 0.15s; }}
            .mode-btn:active {{ background:#334155; }}
            .mode-btn.active {{ background:rgba(168,85,247,0.20) !important;
                color:#a855f7 !important; border-color:#a855f7 !important; }}
            #hint {{ text-align:center; color:#64748b; font-size:10px;
                padding:2px 0 4px 0; background:#0f1116; }}
            #panel {{ padding:4px 8px 10px 8px; background:#0f1116;
                display:flex; flex-wrap:wrap; justify-content:center; gap:4px; }}
            .chip {{ background:#1e293b; color:#cbd5e1;
                border:1px solid #334155; border-radius:14px;
                padding:5px 12px; font-size:11px; cursor:pointer;
                margin:2px; user-select:none;
                -webkit-tap-highlight-color:transparent; transition:all 0.15s; }}
            .chip:active {{ background:#334155; }}
            .chip.active {{ background:#a855f7 !important; color:#fff !important;
                border-color:#a855f7 !important; }}
            #reset {{ background:rgba(168,85,247,0.15); color:#a855f7;
                border:1px solid #a855f7; border-radius:14px;
                padding:5px 14px; font-size:11px; cursor:pointer;
                font-weight:bold; margin:2px 2px 2px 6px;
                -webkit-tap-highlight-color:transparent; }}
            #reset:active {{ background:rgba(168,85,247,0.30); }}
        </style>
    </head>
    <body>
        <div id="mode-toggle">
            <button class="mode-btn active" data-mode="volume">📊 Volume (Lot)</button>
            <button class="mode-btn" data-mode="value">💰 Value (Rp)</button>
        </div>
        <div id="hint">👆 Tap broker di bawah untuk highlight</div>
        <div id="panel">
            {chips_html}
            <button id="reset">↻ Reset</button>
        </div>
        <div id="chart"></div>
        <script>
            (function() {{
                var figData = {fig_json};
                var config = {{ displayModeBar:false, responsive:true,
                    scrollZoom:false, displaylogo:false }};
                var gd = document.getElementById('chart');
                var SVG_NS = 'http://www.w3.org/2000/svg';

                // ▼ Guard semua akses meta
                var META = (figData.layout && figData.layout.meta) || {{}};
                var MODE_PAYLOAD = META.mode_toggle || null;
                var N_BUYERS = META.n_buyers || {n_buyers_js};

                var ORIG = JSON.parse(JSON.stringify(figData.data[0]));
                var LAYOUT = JSON.parse(JSON.stringify(figData.layout));
                // Hapus meta dari layout biar gak ganggu render
                delete LAYOUT.meta;

                var labels = [].concat(ORIG.node.label || []);
                var sources = [].concat(ORIG.link.source || []);
                var targets = [].concat(ORIG.link.target || []);
                var origNodeColors = [].concat(ORIG.node.color || []);
                var origNodeLabels = [].concat(ORIG.node.label || []);
                var origLinkValues = [].concat(ORIG.link.value || []);
                var origLinkColors = [].concat(ORIG.link.color || []);

                if (origNodeColors.length < labels.length) {{
                    var base = origNodeColors[0] || '#a855f7';
                    origNodeColors = labels.map(function(){{ return base; }});
                }}
                if (origLinkColors.length < sources.length) {{
                    var baseL = 'rgba(148,163,184,0.55)';
                    var tmp = [];
                    for (var q = 0; q < sources.length; q++) tmp.push(baseL);
                    origLinkColors = tmp;
                }}

                var GRAY_N = 'rgba(100, 116, 139, 0.15)';
                var GRAY_L = 'rgba(100, 116, 139, 0.05)';
                var currentHighlightIdx = null;
                var CURRENT_MODE = 'volume';

                console.log('[Sankey] init N_BUYERS:', N_BUYERS,
                    'nodes:', labels.length, 'links:', sources.length);

                function fmtFlow(v) {{
                    if (v >= 1e12) return (v/1e12).toFixed(2) + 'T';
                    if (v >= 1e9) return (v/1e9).toFixed(2) + 'B';
                    if (v >= 1e6) return (v/1e6).toFixed(2) + 'M';
                    if (v >= 1e3) return Math.round(v).toLocaleString('id-ID');
                    return Math.round(v).toString();
                }}

                function buildDynamicLabels(activeNodes, activeLinksMap) {{
                    if (!activeLinksMap) return origNodeLabels.slice();
                    var flowPerNode = {{}};
                    for (var i = 0; i < labels.length; i++) flowPerNode[i] = 0;
                    for (var j = 0; j < sources.length; j++) {{
                        if (!activeLinksMap[j]) continue;
                        var v = origLinkValues[j] || 0;
                        flowPerNode[sources[j]] += v;
                        flowPerNode[targets[j]] += v;
                    }}
                    var out = [];
                    for (var k = 0; k < labels.length; k++) {{
                        var codeMatch = String(origNodeLabels[k]).match(/^([A-Z]{{2,4}})/);
                        var code = codeMatch ? codeMatch[1] : String(origNodeLabels[k]).split(' ')[0];
                        if (activeNodes[k]) {{
                            out.push(code + ' (' + fmtFlow(flowPerNode[k]) + ')');
                        }} else {{
                            out.push(code);
                        }}
                    }}
                    return out;
                }}

                function buildTrace(activeNodes, activeLinksMap) {{
                    var t = JSON.parse(JSON.stringify(ORIG));
                    // Node colors
                    var newN = [];
                    for (var i = 0; i < labels.length; i++) {{
                        newN.push(activeNodes[i] ? origNodeColors[i] : GRAY_N);
                    }}
                    t.node.color = newN;
                    // Link colors — fallback solid kalau gradient gagal
                    var newL = [];
                    for (var j = 0; j < sources.length; j++) {{
                        if (!activeLinksMap) {{
                            newL.push(origLinkColors[j] || 'rgba(148,163,184,0.55)');
                        }} else if (activeLinksMap[j]) {{
                            newL.push(origLinkColors[j] || 'rgba(148,163,184,0.55)');
                        }} else {{
                            newL.push(GRAY_L);
                        }}
                    }}
                    t.link.color = newL;
                    t.node.label = buildDynamicLabels(activeNodes, activeLinksMap);
                    return t;
                }}

                function findLinkElements() {{
                    var sels = [
                        '.sankey-link',
                        'path.sankey-link',
                        'g.sankey-links path',
                        'g.sankey-links > path',
                        'g.link path',
                        'g.sankey path',
                        'path[class*="sankey"]',
                        'path[class*="link"]'
                    ];
                    for (var i = 0; i < sels.length; i++) {{
                        try {{
                            var els = gd.querySelectorAll(sels[i]);
                            console.log('[Sankey] try selector:', sels[i],
                                        '→', els.length, 'elements');
                            if (els && els.length > 0) return els;
                        }} catch (e) {{}}
                    }}
                    var svg = gd.querySelector('svg');
                    if (!svg) {{
                        console.warn('[Sankey] ❌ no svg element');
                        return null;
                    }}
                    var allPaths = svg.querySelectorAll('path');
                    console.log('[Sankey] total paths in svg:', allPaths.length);
                    var withCurve = [];
                    allPaths.forEach(function(p) {{
                        var d = p.getAttribute('d') || '';
                        if (d.indexOf('C') !== -1 || d.indexOf('c') !== -1) {{
                            withCurve.push(p);
                        }}
                    }});
                    console.log('[Sankey] paths with curve:', withCurve.length);
                    if (withCurve.length > 0) return withCurve;
                    return allPaths.length > 0 ? allPaths : null;
                }}

                function applyGradients(activeLinksMap) {{
                    var svg = gd.querySelector('svg');
                    if (!svg) return false;

                    var linkEls = findLinkElements();
                    if (!linkEls || linkEls.length === 0) return false;

                    // Buang defs lama
                    var oldDefs = svg.querySelector('#sg-defs');
                    if (oldDefs) oldDefs.remove();
                    var defs = document.createElementNS(SVG_NS, 'defs');
                    defs.setAttribute('id', 'sg-defs');
                    svg.insertBefore(defs, svg.firstChild);

                    // ▼ Hitung lebar SVG dalam user units (viewBox)
                    var svgW = 1000;
                    try {{
                        var vb = svg.viewBox && svg.viewBox.baseVal;
                        if (vb && vb.width > 0) {{
                            svgW = vb.width;
                        }} else {{
                            svgW = svg.clientWidth ||
                                   svg.getBoundingClientRect().width || 1000;
                        }}
                    }} catch(e) {{}}
                    console.log('[Sankey] svgW:', svgW);

                    var nLinks = sources.length;
                    var applied = 0;

                    // ▼ Loop SEMUA element (128), map index → link asli dengan modulo
                    for (var i = 0; i < linkEls.length; i++) {{
                        var linkEl = linkEls[i];
                        var linkIdx = i % nLinks;

                        var isActive = !activeLinksMap || activeLinksMap[linkIdx];
                        if (!isActive) continue;

                        var srcColor = origNodeColors[sources[linkIdx]] || '#64748b';
                        var tgtColor = origNodeColors[targets[linkIdx]] || '#64748b';

                        var gid = 'sg-grad-' + i;
                        var gr = document.createElementNS(SVG_NS, 'linearGradient');
                        gr.setAttribute('id', gid);

                        // ▼ KUNCI FIX: userSpaceOnUse + koordinat absolute SVG
                        gr.setAttribute('gradientUnits', 'userSpaceOnUse');
                        gr.setAttribute('x1', 0);
                        gr.setAttribute('y1', 0);
                        gr.setAttribute('x2', svgW);
                        gr.setAttribute('y2', 0);

                        var s1 = document.createElementNS(SVG_NS, 'stop');
                        s1.setAttribute('offset', '0%');
                        s1.setAttribute('stop-color', srcColor);
                        s1.setAttribute('stop-opacity', '0.85');

                        var s2 = document.createElementNS(SVG_NS, 'stop');
                        s2.setAttribute('offset', '100%');
                        s2.setAttribute('stop-color', tgtColor);
                        s2.setAttribute('stop-opacity', '0.85');

                        gr.appendChild(s1);
                        gr.appendChild(s2);
                        defs.appendChild(gr);

                        linkEl.setAttribute('fill', 'url(#' + gid + ')');
                        linkEl.style.setProperty('fill', 'url(#' + gid + ')', 'important');
                        applied++;
                    }}

                    console.log('[Sankey] ✅ gradient applied:', applied,
                                'of', linkEls.length, 'elements (svgW:' + svgW + ')');
                    return applied > 0;
                }}

                function renderWithGradients(activeNodes, activeLinksMap) {{
                    var t = buildTrace(activeNodes, activeLinksMap);
                    return Plotly.react(gd, [t], LAYOUT, config).then(function() {{
                        function tryApply(attempt) {{
                            if (attempt > 6) {{
                                console.warn('[Sankey] gradient retry exhausted');
                                return;
                            }}
                            var ok = applyGradients(activeLinksMap);
                            if (!ok) {{
                                setTimeout(function() {{
                                    tryApply(attempt + 1);
                                }}, 120 + attempt * 80);
                            }}
                        }}
                        requestAnimationFrame(function() {{
                            requestAnimationFrame(function() {{
                                tryApply(0);
                            }});
                        }});
                    }});
                }}

                function highlightNode(nodeIdx) {{
                    var isBuyer = nodeIdx < N_BUYERS;
                    var activeNodes = {{}};
                    var activeLinks = {{}};
                    activeNodes[nodeIdx] = true;

                    for (var j = 0; j < sources.length; j++) {{
                        if (isBuyer) {{
                            if (sources[j] === nodeIdx) {{
                                activeLinks[j] = true;
                                activeNodes[targets[j]] = true;
                            }}
                        }} else {{
                            if (targets[j] === nodeIdx) {{
                                activeLinks[j] = true;
                                activeNodes[sources[j]] = true;
                            }}
                        }}
                    }}

                    currentHighlightIdx = nodeIdx;
                    renderWithGradients(activeNodes, activeLinks);

                    document.querySelectorAll('.chip').forEach(function(c) {{
                        var idx = parseInt(c.getAttribute('data-node-idx'));
                        if (idx === nodeIdx) c.classList.add('active');
                        else c.classList.remove('active');
                    }});

                    if (navigator.vibrate) {{ try {{ navigator.vibrate(8); }} catch(e) {{}} }}
                }}

                function resetAll() {{
                    currentHighlightIdx = null;
                    var all = {{}};
                    for (var i = 0; i < labels.length; i++) all[i] = true;
                    renderWithGradients(all, null);
                    document.querySelectorAll('.chip').forEach(function(c) {{
                        c.classList.remove('active');
                    }});
                    if (navigator.vibrate) {{ try {{ navigator.vibrate(8); }} catch(e) {{}} }}
                }}

                function applyMode(mode) {{
                    if (!MODE_PAYLOAD || !MODE_PAYLOAD[mode]) return;
                    var md = MODE_PAYLOAD[mode];
                    ORIG.link.value = md.link_values.slice();
                    ORIG.node.label = md.node_labels.slice();
                    origNodeLabels = md.node_labels.slice();
                    origLinkValues = md.link_values.slice();
                    labels = ORIG.node.label.slice();
                    CURRENT_MODE = mode;

                    document.querySelectorAll('.mode-btn').forEach(function(b) {{
                        if (b.getAttribute('data-mode') === mode) b.classList.add('active');
                        else b.classList.remove('active');
                    }});

                    if (currentHighlightIdx !== null) highlightNode(currentHighlightIdx);
                    else resetAll();
                }}

                // ═══════════════════════════════════════════════════
                // INITIAL RENDER 
                // ═══════════════════════════════════════════════════
                var initNodes = {{}};
                for (var ii = 0; ii < labels.length; ii++) initNodes[ii] = true;
                var initTrace = buildTrace(initNodes, null);

                Plotly.newPlot(gd, [initTrace], LAYOUT, config).then(function() {{
                    function tryApplyInit(attempt) {{
                        if (attempt > 6) {{
                            console.warn('[Sankey] init gradient retry exhausted');
                            return;
                        }}
                        var ok = applyGradients(null);
                        if (!ok) {{
                            setTimeout(function() {{
                                tryApplyInit(attempt + 1);
                            }}, 120 + attempt * 80);
                        }}
                    }}
                    requestAnimationFrame(function() {{
                        requestAnimationFrame(function() {{
                            tryApplyInit(0);
                        }});
                    }});
                }}).catch(function(err) {{
                    console.error('[Sankey] newPlot error:', err);
                }});

                // ═══════════════════════════════════════════════════
                // EVENT DELEGATION — attach ke document, bukan chip
                // ═══════════════════════════════════════════════════
                document.addEventListener('click', function(ev) {{
                    var target = ev.target;
                    if (!target) return;

                    // Chip
                    var chip = target.closest ? target.closest('.chip') : null;
                    if (chip) {{
                        var idx = parseInt(chip.getAttribute('data-node-idx'));
                        if (currentHighlightIdx === idx) resetAll();
                        else highlightNode(idx);
                        return;
                    }}

                    // Mode button
                    var mbtn = target.closest ? target.closest('.mode-btn') : null;
                    if (mbtn) {{
                        var m = mbtn.getAttribute('data-mode');
                        if (m !== CURRENT_MODE) applyMode(m);
                        return;
                    }}

                    // Reset button
                    var rbtn = target.closest ? target.closest('#reset') : null;
                    if (rbtn) resetAll();
                }});

                // Native Sankey click (best-effort)
                gd.on('plotly_click', function(data) {{
                    if (!data || !data.points || !data.points.length) return;
                    var pt = data.points[0];
                    if (pt.pointType === 'node' && typeof pt.pointNumber === 'number') {{
                        highlightNode(pt.pointNumber);
                    }}
                }});
            }})();
        </script>
    </body>
    </html>
    """
    components.html(html, height=height + 100, scrolling=False)
# ====================== FALLBACK HANDLERS ======================
PIL_AVAILABLE = True
try:
    from PIL import Image
    import io
except ImportError:
    PIL_AVAILABLE = False

def compress_image_for_gemini(image, max_width=1280, max_height=960, quality=85):
    """
    Kompresi gambar untuk hemat token Gemini Vision.
    Resize & reduce quality sambil maintain readable content.
    
    Support input:
    - PIL.Image.Image
    - str (file path)
    - Streamlit UploadedFile (auto-convert)
    """
    if not PIL_AVAILABLE:
        return image

    img = None  # ← inisialisasi awal

    try:
        # ── Step 1: Convert input ke PIL Image ──
        if isinstance(image, str):
            img = Image.open(image)
        elif hasattr(image, 'read'):
            # Streamlit UploadedFile / BytesIO
            img = Image.open(image)
        else:
            # Asumsi sudah PIL Image
            img = image

        if img is None:
            return image

        # ── Step 2: Resize kalau terlalu besar ──
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # ── Step 3: Convert RGBA/P → RGB (untuk JPEG) ──
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            try:
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
            except Exception:
                rgb_img.paste(img.convert('RGB'))
            img = rgb_img

        # ── Step 4: Save ke bytes dengan kompresi ──
        compressed_io = io.BytesIO()
        img.save(compressed_io, format='JPEG', quality=quality, optimize=True)
        compressed_io.seek(0)

        # ── Step 5: Return PIL Image ──
        return Image.open(compressed_io)

    except Exception as e:
        st.warning(f"⚠️ Kompresi gambar gagal: {e}. Menggunakan gambar original.")
        return image

PLOTLY_AVAILABLE = True
try: import plotly.graph_objects as go
except ImportError: PLOTLY_AVAILABLE = False

SENTIMENT_AVAILABLE = True
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: nltk.download('vader_lexicon', quiet=True)
except ImportError: SENTIMENT_AVAILABLE = False

RSS_AVAILABLE = True
try: import feedparser
except ImportError: RSS_AVAILABLE = False

TRANSLATOR_AVAILABLE = True
try: from deep_translator import GoogleTranslator
except ImportError: TRANSLATOR_AVAILABLE = False

warnings.filterwarnings("ignore")
def safe_float(value, default=0.0):
    """Konversi aman ke float, kembalikan default jika gagal."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_gemini_api_key():
    """Helper function to get Gemini API Key dari session state atau secrets."""
    if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
        return st.session_state.gemini_api_key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except:
        return os.getenv("GEMINI_API_KEY", "")
BROKER_TYPES = {
    "AK": "Foreign", "BK": "Foreign", "ZP": "Foreign", "RX": "Foreign",
    "YU": "Foreign", "KZ": "Foreign", "KK": "Foreign", "YU": "Foreign",
    "YP": "Foreign", "CP": "Foreign", "DU": "Foreign", "BQ": "Foreign",
    "HD": "Foreign", "DR": "Foreign", "TP": "Foreign", "AG": "Foreign",
    "XA": "Foreign", "AI": "Foreign", "FS": "Foreign", "LS": "Foreign",
    "DP": "Foreign", "RB": "Foreign", "AH": "Foreign", "GI": "Foreign",
    "CC": "BUMN", "NI": "BUMN", "OD": "BUMN","DX": "BUMN",
    "XL": "Domestic", "LG": "Domestic", "PD": "Domestic", "SQ": "Domestic",
    "XC": "Domestic", "MG": "Domestic", "AZ": "Domestic", "GR": "Domestic",
    "DH": "Domestic", "YB": "Domestic", "EP": "Domestic", "FZ": "Domestic",
    "KI": "Domestic", "IF": "Domestic", "PP": "Domestic", "HP": "Domestic",
    "BB": "Domestic", "YJ": "Domestic", "AP": "Domestic", "CD": "Domestic",
    "PO": "Domestic", "AO": "Domestic", "BR": "Domestic", "SS": "Domestic",
    "AT": "Domestic", "IN": "Domestic", "RF": "Domestic", "EL": "Domestic",
    "RO": "Domestic", "SH": "Domestic", "PC": "Domestic", "PG": "Domestic",
    "II": "Domestic", "IH": "Domestic", "IU": "Domestic", "AR": "Domestic",
    "ES": "Domestic", "ZR": "Domestic", "MI": "Domestic", "PI": "Domestic",
    "SA": "Domestic", "MU": "Domestic", "SF": "Domestic", "QA": "Domestic",
    "ID": "Domestic", "RS": "Domestic", "RG": "Domestic", "GA": "Domestic",
    "AF": "Domestic", "TS": "Domestic", "PF": "Domestic", "BS": "Domestic",
    "AD": "Domestic", "TF": "Domestic", "OK": "Domestic", "JB": "Domestic",
    "IC": "Domestic", "BF": "Domestic", "IT": "Domestic", "DD": "Domestic",
    "YO": "Domestic", "FO": "Domestic"
}

def get_broker_label(code):
    """Format: 'XL (L)', 'CC (G)', 'AK (F)'"""
    cat = BROKER_TYPES.get(code, "Domestic")
    badge = "F" if cat == "Foreign" else ("G" if cat == "BUMN" else "L")
    return f"{code} ({badge})"

def _parse_broker_list(raw):
    """Parse JSON string/list dari Google Sheets → list of dict."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []
# ═══════════════════════════════════════════════════════════════
# KLASIFIKASI BROKER — RETAIL vs BANDAR
# ═══════════════════════════════════════════════════════════════
RETAIL_BROKERS = {
    "YP",  # Mirae Asset (banyak retail)
    "PD",  # Indo Premier (Stockbit-heavy retail)
    "XL",  # Stockbit Sekuritas
    "XC",  # Bahana (retail)
    "CC",  # Mandiri (mixed, banyak retail)
    "NI",  # BNI Sekuritas (mixed)
    "OD",  # BRI Danareksa (mixed)
    "ID",  # Inti Fikasa (retail)
    "SQ",  # BCA Sekuritas (retail)
    "KK",  # Phillip (retail-heavy)
    "DX",  # Bahana (retail)
    "AZ",  # Sucor (retail)
    "BK",  # Binaartha
    "HP",  # Henan Putihrai
    "KS",  # Kresna
    "TF",  # Trimegah
}

BANDAR_BROKERS = {
    "AK",  # UBS (institusi asing)
    "KZ",  # Credit Suisse
    "CS",  # Credit Suisse
    "RX",  # Macquarie
    "ZP",  # Maybank Kim Eng
    "YU",  # CIMB
    "DR",  # DBS Vickers
    "CP",  # Valbury (institusi)
    "HD",  # KGI
    "AI",  # UOB Kay Hian
    "LS",  # Reliance
    "BB",  # Vickers (institusi)
    "TP",  # OCBC
    "DU",  # Danatama
    "FS",  # Fajar Surya
    "DP",  # Dipo Stars
}

def klasifikasi_broker(broker_code, volume_lot, freq=None):
    """
    Klasifikasi broker: Bandar / Retail / Mixed.

    Rule:
    1. Kalau code ada di BANDAR_BROKERS → Bandar 🐋
    2. Kalau code ada di RETAIL_BROKERS → Retail 🧑
    3. Kalau ada freq → pakai avg_lot_per_freq:
       - avg > 200 lot/transaksi → Bandar
       - avg < 50 lot/transaksi → Retail
       - 50–200 → Mixed
    4. Default → Mixed ⚖️

    Return: (kategori_str, icon_str)
    """
    code = str(broker_code).upper().strip()
    if code in BANDAR_BROKERS:
        return "Bandar", "🐋"
    if code in RETAIL_BROKERS:
        return "Retail", "🧑"

    try:
        vol = float(volume_lot or 0)
        frq = float(freq) if freq not in (None, "", "null") else None
    except Exception:
        vol, frq = 0, None

    if frq and frq > 0 and vol > 0:
        avg_per_freq = vol / frq
        if avg_per_freq > 200:
            return "Bandar", "🐋"
        elif avg_per_freq < 50:
            return "Retail", "🧑"
        
            return "Mixed", "⚖️"

    return "Mixed", "⚖️"

def enrich_broker_kategori(res_json):
    """
    Tambahkan field 'kategori' + 'kategori_icon' ke setiap item
    di top_buyers & top_sellers. Mengembalikan dict res_json yang sama
    (in-place) untuk chaining.

    Input: res_json hasil analisis_broksum_gemini_vision (atau OCR)
    Output: res_json dengan tambahan field kategori
    """
    if not isinstance(res_json, dict):
        return res_json

    for side_key in ("top_buyers", "top_sellers"):
        for item in res_json.get(side_key, []) or []:
            if not isinstance(item, dict):
                continue
            kode = item.get("broker", "")
            vol = item.get("volume_lot", 0) or 0
            frq = item.get("freq")
            kat, icon = klasifikasi_broker(kode, vol, frq)
            item["kategori"] = kat
            item["kategori_icon"] = icon

    # Hitung ringkasan statistik Bandar vs Retail
    tot_buyer_vol = sum(float(b.get("volume_lot", 0) or 0) for b in res_json.get("top_buyers", []) if isinstance(b, dict))
    tot_seller_vol = sum(float(s.get("volume_lot", 0) or 0) for s in res_json.get("top_sellers", []) if isinstance(s, dict))
    
    b_buy_vol = sum(float(b.get("volume_lot", 0) or 0) for b in res_json.get("top_buyers", []) if isinstance(b, dict) and b.get("kategori") == "Bandar")
    r_buy_vol = sum(float(b.get("volume_lot", 0) or 0) for b in res_json.get("top_buyers", []) if isinstance(b, dict) and b.get("kategori") == "Retail")
    
    b_sell_vol = sum(float(s.get("volume_lot", 0) or 0) for s in res_json.get("top_sellers", []) if isinstance(s, dict) and s.get("kategori") == "Bandar")
    r_sell_vol = sum(float(s.get("volume_lot", 0) or 0) for s in res_json.get("top_sellers", []) if isinstance(s, dict) and s.get("kategori") == "Retail")

    b_buy_pct = (b_buy_vol / tot_buyer_vol * 100) if tot_buyer_vol > 0 else 0
    r_buy_pct = (r_buy_vol / tot_buyer_vol * 100) if tot_buyer_vol > 0 else 0
    b_sell_pct = (b_sell_vol / tot_seller_vol * 100) if tot_seller_vol > 0 else 0
    r_sell_pct = (r_sell_vol / tot_seller_vol * 100) if tot_seller_vol > 0 else 0

    res_json["broker_summary_stats"] = {
        "buyer_bandar_pct": round(b_buy_pct, 1),
        "buyer_retail_pct": round(r_buy_pct, 1),
        "seller_bandar_pct": round(b_sell_pct, 1),
        "seller_retail_pct": round(r_sell_pct, 1),
    }

    # Jika summary_narrative belum ada info klasifikasi, tambahkan note ringkas di akhir
    curr_narrative = res_json.get("summary_narrative", "")
    if curr_narrative and "Bandar" not in curr_narrative and "Retail" not in curr_narrative:
        kat_note = f" (Komposisi Pembeli: {b_buy_pct:.0f}% Bandar 🐋 / {r_buy_pct:.0f}% Retail 🧑 | Penjual: {b_sell_pct:.0f}% Bandar 🐋 / {r_sell_pct:.0f}% Retail 🧑)"
        res_json["summary_narrative"] = curr_narrative.rstrip(".") + kat_note + "."

    return res_json

def format_broker_list_for_ai(broker_list):
    """
    Mengubah list broker (top_buyers / top_sellers) menjadi formatted text
    dengan klasifikasi kategori (Bandar 🐋, Retail 🧑, Mixed ⚖️) dan statistik persentase.
    """
    if not broker_list:
        return "- (tidak ada data)", "Tidak ada data"

    lines = []
    tot_vol = 0
    vol_by_cat = {"Bandar": 0, "Retail": 0, "Mixed": 0}

    for item in broker_list:
        if not isinstance(item, dict):
            continue
        code = str(item.get("broker", "N/A")).upper().strip()
        vol = float(item.get("volume_lot", 0) or 0)
        frq = item.get("freq")
        kat = item.get("kategori")
        icon = item.get("kategori_icon")
        if not kat or not icon:
            kat, icon = klasifikasi_broker(code, vol, frq)
        
        tot_vol += vol
        vol_by_cat[kat] = vol_by_cat.get(kat, 0) + vol
        lines.append(f"- {code} ({kat} {icon}): {vol:,.0f} lot")

    cat_summary = []
    if tot_vol > 0:
        for cat_name, cat_icon in [("Bandar", "🐋"), ("Retail", "🧑"), ("Mixed", "⚖️")]:
            v = vol_by_cat.get(cat_name, 0)
            if v > 0:
                pct = (v / tot_vol) * 100
                cat_summary.append(f"{cat_name} {cat_icon}: {pct:.1f}% ({v:,.0f} lot)")
    
    summary_str = " | ".join(cat_summary) if cat_summary else "N/A"
    list_str = "\n".join(lines) if lines else "- (tidak ada data)"
    return list_str, summary_str
# ═══════════════════════════════════════════════════════════════
# V12 ADAPTIVE ENGINE – KONSTANTA & STATE
# ═══════════════════════════════════════════════════════════════
FACTOR_KEYS   = ["Momentum","AI_Senti","MeanRev","Beta_IHSG","Coppock","OFI", "Bandar_Flow", "Foreign_ZScore"]
WEIGHT_MIN    = 0.08
WEIGHT_MAX    = 0.40
SOFTMAX_TEMP  = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM  = 0.82

# ====================== FRAKSI HARGA BEI ======================
def fraksi_bei(harga):
    """Membulatkan harga ke kelipatan fraksi sesuai aturan BEI.
    
    Defensive: kalau harga NaN / None / invalid, return 0.
    """
    try:
        # ── NaN / None guard ──
        if harga is None:
            return 0
        h = float(harga)
        if math.isnan(h) or math.isinf(h) or h <= 0:
            return 0

        if h < 200:
            fraksi = 1
        elif h < 500:
            fraksi = 2
        elif h < 2000:
            fraksi = 5
        elif h < 5000:
            fraksi = 10
        
            fraksi = 25
        return round(h / fraksi) * fraksi
    except (ValueError, TypeError, OverflowError):
        return 0

def fraksi_step(harga):
    """Mengembalikan nilai kelipatan 1 fraksi BEI.

    Defensive: kalau harga NaN / None / invalid, return 1 (fraksi minimum).
    """
    try:
        if harga is None:
            return 1
        h = float(harga)
        if math.isnan(h) or math.isinf(h) or h <= 0:
            return 1

        if h < 200:
            return 1
        elif h < 500:
            return 2
        elif h < 2000:
            return 5
        elif h < 5000:
            return 10
        else:
            return 25
    except (ValueError, TypeError, OverflowError):
        return 1

# ====================== GOOGLE SHEETS FUNCTIONS ======================
def get_gsheet():
    """Mengembalikan objek spreadsheet berdasarkan secrets."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["google_sheets"]["sheet_id"])

def init_sheets():
    """Membuat sheet 'riwayat', 'v12_memory', 'v12_predictions', 'broksum_history', dan 'foreign_flow_history' jika belum ada."""
    try:
        sheet = get_gsheet()
        existing = {ws.title: ws for ws in sheet.worksheets()}

        if "riwayat" not in existing:
            sheet.add_worksheet("riwayat", rows=3000, cols=35)

        if "v12_memory" not in existing:
            sheet.add_worksheet("v12_memory", rows=100, cols=3)

        if "v12_predictions" not in existing:
            sheet.add_worksheet("v12_predictions", rows=500, cols=9)
        else:
            ws = existing["v12_predictions"]
            if ws.col_count < 9:
                ws.add_cols(9 - ws.col_count)

        if "riwayat_actual" not in existing:
            ws = sheet.add_worksheet("riwayat_actual", rows=100, cols=8)
            ws.update("A1:H1",
                [["Waktu", "Saham", "Mode", "Actual_High", "Actual_Low",
                  "Actual_Close", "Outcome", "Entry_Miss"]],
                value_input_option='RAW')

        if "broksum_history" not in existing:
            ws = sheet.add_worksheet("broksum_history", rows=2000, cols=8)
            ws.update(
                "A1:H1",
                [["ticker", "upload_date", "bandarmology_status", "top_buyers",
                  "top_sellers", "summary_narrative", "full_data", "source"]],
                value_input_option='RAW'
            )

        # ▼ BARU: Sheet foreign_flow_history
        if "foreign_flow_history" not in existing:
            ws = sheet.add_worksheet("foreign_flow_history", rows=5000, cols=7)
            ws.update(
                "A1:G1",
                [["ticker", "date", "close",
                  "foreign_buy", "foreign_sell", "net_foreign", "source"]],
                value_input_option='RAW'
            )

    except Exception as e:
        st.error(f"❌ Gagal inisialisasi Google Sheets: {e}")

#V12 Memory (Google Sheets)
def load_v12_memory():
    mem = {}
    try:
        sheet = get_gsheet().worksheet("v12_memory")
        records = sheet.get_all_records()
        for row in records:
            t = row.get('ticker')
            if t and 'data' in row and row['data']:
                try:
                    mem[t] = json.loads(row['data'])
                except:
                    pass
    except Exception as e:
        st.error(f"Gagal memuat V12 memory: {e}")
    return mem

def save_v12_memory(mem):
    try:
        sheet = get_gsheet().worksheet("v12_memory")
        rows = [{'ticker': t, 'data': json.dumps(d)} for t, d in mem.items()]
        sheet.clear()
        if rows:
            all_values = [['ticker', 'data']] + [[r['ticker'], r['data']] for r in rows]
            sheet.update(all_values, value_input_option='RAW')
    except Exception as e:
        st.error(f"Gagal menyimpan V12 memory: {e}")

def load_v12_predictions(ticker, mode="swing"):
    try:
        sheet = get_gsheet().worksheet("v12_predictions")
        records = sheet.get_all_records()
        for row in records:
            if row.get('ticker') == ticker and row.get('mode') == mode:
                return row
        
    except Exception as e:
        st.error(f"Gagal memuat prediksi: {e}")
        return None

def save_v12_prediction(ticker, close_price, factor_signals, entry_low=None, entry_high=None, mode="swing"):
    try:
        sheet = get_gsheet()
        ws = sheet.worksheet("v12_predictions")
        new_row = {
            'ticker': ticker,
            'mode': mode,               # <-- tambah ini
            'close_price': close_price,
            'timestamp': datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S"),
            'entry_low': entry_low,
            'entry_high': entry_high
        }
        for k in FACTOR_KEYS:
            new_row[f'sig_{k}'] = factor_signals.get(k, 0.0)

        headers = list(new_row.keys())   # sekarang 10 kolom (termasuk mode)
        # ---------- Pastikan jumlah kolom cukup ----------
        if ws.col_count < len(headers):
            ws.add_cols(len(headers) - ws.col_count)

        # Tulis ulang header
        last_col = chr(64 + len(headers))
        ws.update(f'A1:{last_col}1', [headers], value_input_option='RAW')

        records = ws.get_all_records()
        row_index = None
        for i, row in enumerate(records):
            if row.get('ticker') == ticker and row.get('mode') == mode:
                row_index = i + 2
                break

        if row_index:
            values = [new_row[h] for h in headers]
            last_col = chr(64 + len(headers))
            ws.update(f'A{row_index}:{last_col}{row_index}', [values], value_input_option='RAW')
        else:
            values = [new_row[h] for h in headers]
            ws.append_row(values, value_input_option='RAW')
    except Exception as e:
        st.error(f"Gagal menyimpan prediksi: {e}")

# ====================== BROKSUM HISTORY (GOOGLE SHEETS) ======================
def save_broksum_data(ticker, res_json, source="gemini"):
    """Simpan hasil parsing broker flow ke Google Sheets broksum_history."""
    try:
        sheet = get_gsheet()
        ws = sheet.worksheet("broksum_history")
        
        upload_date = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            'ticker': ticker.upper(),
            'upload_date': upload_date,
            'bandarmology_status': res_json.get('bandarmology_status', 'N/A'),
            'top_buyers': json.dumps(res_json.get('top_buyers', [])),
            'top_sellers': json.dumps(res_json.get('top_sellers', [])),
            'summary_narrative': res_json.get('summary_narrative', ''),
            'full_data': json.dumps(res_json),
            'source': source
        }
        
        # Pastikan header ada
        try:
            headers = ws.row_values(1)
            if not headers or headers[0] != 'ticker':
                ws.update("A1:H1", [["ticker", "upload_date", "bandarmology_status", "top_buyers", "top_sellers", "summary_narrative", "full_data", "source"]], value_input_option='RAW')
        except:
            ws.update("A1:H1", [["ticker", "upload_date", "bandarmology_status", "top_buyers", "top_sellers", "summary_narrative", "full_data", "source"]], value_input_option='RAW')
        
        # Append row baru (keep history)
        values = [
            new_row['ticker'],
            new_row['upload_date'],
            new_row['bandarmology_status'],
            new_row['top_buyers'],
            new_row['top_sellers'],
            new_row['summary_narrative'],
            new_row['full_data'],
            new_row['source']
        ]
        ws.append_row(values, value_input_option='RAW')
        return True
    except Exception as e:
        st.error(f"❌ Gagal menyimpan broker flow ke Sheets: {e}")
        return False

def load_broksum_history(ticker):
    """Load semua history broker flow untuk ticker tertentu dari Sheets."""
    try:
        sheet = get_gsheet()
        ws = sheet.worksheet("broksum_history")
        records = ws.get_all_records()
        
        ticker_upper = ticker.upper()
        history = [row for row in records if row.get('ticker', '').upper() == ticker_upper]
        
        return history  # Return sorted by date (newest first bisa di handle di UI)
    except Exception as e:
        st.error(f"❌ Gagal memuat broker flow history: {e}")
        return []

def get_latest_broksum_for_ticker(ticker):
    """Ambil data broker flow TERBARU untuk ticker tertentu."""
    try:
        history = load_broksum_history(ticker)
        if history:
            # Asumsi records sudah sorted by upload_date DESC
            latest = history[0]
            # Parse JSON fields
            return {
                'ticker': latest.get('ticker'),
                'upload_date': latest.get('upload_date'),
                'bandarmology_status': latest.get('bandarmology_status'),
                'top_buyers': json.loads(latest.get('top_buyers', '[]')),
                'top_sellers': json.loads(latest.get('top_sellers', '[]')),
                'summary_narrative': latest.get('summary_narrative'),
                'full_data': json.loads(latest.get('full_data', '{}'))
            }
        return None
    except Exception as e:
        st.error(f"❌ Error get latest broksum: {e}")
        return None
    
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_idx_all_stock_summary():
    """Ambil semua data saham dari IDX sekali request (cache 30 menit).
    
    PENTING: Kalau gagal, raise exception supaya Streamlit TIDAK cache None.
    """
    url = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9,id;q=0.8",
        "egrum": "isAjax:true",
        "referer": "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-requested-with": "XMLHttpRequest",
    }
    
    try:
        try:
            from curl_cffi import requests as curl_requests
            r = curl_requests.get(url, headers=headers, timeout=25,
                                   impersonate="chrome120")
        except ImportError:
            r = requests.get(url, headers=headers, timeout=20)

        if r.status_code != 200:
            # ▼ Raise, BUKAN return None — biar gak di-cache
            raise RuntimeError(f"IDX HTTP {r.status_code}")
        
        payload = r.json()
        if isinstance(payload, dict):
            data = payload.get("data") or payload.get("Data") or []
        elif isinstance(payload, list):
            data = payload
        else:
            data = []
        
        if not data:
            raise RuntimeError("IDX response kosong")
        
        return data
    except RuntimeError:
        raise  # ← propagate ke caller, JANGAN di-cache
    except Exception as e:
        # Wrap exception lain, juga jangan di-cache
        raise RuntimeError(f"IDX fetch error: {e}")
    
def save_foreign_flow_snapshot(ticker):
    """
    Ambil data foreign flow IDX hari ini, simpan ke sheet.
    Skip kalau ticker + tanggal sudah ada di sheet.
    Return: True kalau berhasil/skip, False kalau gagal.
    """
    try:
        ticker_clean = str(ticker).upper().replace(".JK", "").strip()
        if not ticker_clean:
            return False
        
        today_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d")
        try:
            _sheet = get_gsheet().worksheet("foreign_flow_history")
            _records = _sheet.get_all_records()
            for r in _records:
                if (str(r.get("ticker", "")).upper() == ticker_clean
                        and str(r.get("date", "")) == today_wib):
                    return True  # sudah ada, skip tanpa fetch IDX
        except Exception:
            pass
        items = _fetch_idx_all_stock_summary()
        if not items:
            return False

        row_data = None
        for it in items:
            if not isinstance(it, dict):
                continue
            code = str(it.get("StockCode", "")).upper().strip()
            if code != ticker_clean:
                continue

            def _f(k):
                v = it.get(k)
                if v in (None, "", "N/A", "-"):
                    return 0.0
                try:
                    return float(str(v).replace(",", ""))
                except Exception:
                    return 0.0

            fb = _f("ForeignBuy")
            fs = _f("ForeignSell")
            close = _f("Close")

            fb_rp = fb * close if close > 0 else 0.0
            fs_rp = fs * close if close > 0 else 0.0

            date_str = str(it.get("Date") or "")[:10]
            if not date_str:
                date_str = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d")

            row_data = {
                "ticker": ticker_clean,
                "date": date_str,
                "close": close,
                "foreign_buy": fb_rp,
                "foreign_sell": fs_rp,
                "net_foreign": fb_rp - fs_rp,
                "source": "idx",
            }
            break

        if not row_data:
            return False

        sheet = get_gsheet().worksheet("foreign_flow_history")
        records = sheet.get_all_records()

        # Cek duplikat: ticker + date
        for r in records:
            if (str(r.get("ticker", "")).upper() == row_data["ticker"]
                    and str(r.get("date", "")) == row_data["date"]):
                return True   # sudah ada, skip

        sheet.append_row([
            row_data["ticker"],
            row_data["date"],
            row_data["close"],
            row_data["foreign_buy"],
            row_data["foreign_sell"],
            row_data["net_foreign"],
            row_data["source"],
        ], value_input_option='RAW')
        return True

    except Exception as e:
        st.error(f"❌ Gagal simpan foreign flow snapshot: {e}")
        return False
def load_foreign_flow_history(ticker, days=30):
    """
    Ambil history foreign flow dari sheet.
    Return: DataFrame [date, close, foreign_buy, foreign_sell, net_foreign]
            atau None kalau kosong.
    """
    try:
        ticker_clean = str(ticker).upper().replace(".JK", "").strip()
        sheet = get_gsheet().worksheet("foreign_flow_history")
        records = sheet.get_all_records()

        rows = []
        for r in records:
            if str(r.get("ticker", "")).upper() != ticker_clean:
                continue
            try:
                rows.append({
                    "date": str(r.get("date", "")),
                    "close": float(r.get("close", 0) or 0),
                    "foreign_buy": float(r.get("foreign_buy", 0) or 0),
                    "foreign_sell": float(r.get("foreign_sell", 0) or 0),
                    "net_foreign": float(r.get("net_foreign", 0) or 0),
                })
            except Exception:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values("date").tail(days).reset_index(drop=True)
        return df if not df.empty else None

    except Exception:
        return None
def fmt_money(v):
    """Format Rupiah dengan satuan T/B/M/K."""
    try:
        v = float(v)
    except Exception:
        return "0"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e12:
        return f"{sign}{av/1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}{av/1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}{av/1e6:.2f}M"
    if av >= 1e3:
        return f"{sign}{av/1e3:.1f}K"
    return f"{sign}{av:,.0f}"

def analyze_broksum_insight_with_gemini(ticker, broksum_data, price_data, api_key):
    """
    Analisis broker flow + harga menggunakan Gemini.
    Input:
    - ticker: Kode saham
    - broksum_data: Hasil parse broker flow (dari database)
    - price_data: Data harga & teknikal (opsional)
    - api_key: Gemini API Key
    
    Output: (insight_text, error)
    """
    if not api_key:
        return None, "API Key Gemini belum diisi."
    
    try:
        genai.configure(api_key=api_key)
        model = None
        available = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if available:
            for model_id in available:
                try:
                    model = genai.GenerativeModel(model_id)
                    model.generate_content("test", generation_config={"max_output_tokens": 1})
                    break
                except:
                    continue
        
        if not model:
            return None, "Model Gemini tidak tersedia."
        
        # Format broker flow data dengan klasifikasi Bandar vs Retail
        top_buyers = broksum_data.get('top_buyers', [])
        top_sellers = broksum_data.get('top_sellers', [])
        if isinstance(top_buyers, str):
            try: top_buyers = json.loads(top_buyers)
            except: top_buyers = []
        if isinstance(top_sellers, str):
            try: top_sellers = json.loads(top_sellers)
            except: top_sellers = []

        buyers_text, buyers_cat_sum = format_broker_list_for_ai(top_buyers)
        sellers_text, sellers_cat_sum = format_broker_list_for_ai(top_sellers)
        
        price_context = ""
        if price_data:
            price_context = f"""
Konteks Harga & Teknikal:
- Current Price: {price_data.get('current_price', 'N/A')}
- Support: {price_data.get('support', 'N/A')}
- Resistance: {price_data.get('resistance', 'N/A')}
- Trend: {price_data.get('trend', 'N/A')}
- Volume: {price_data.get('volume', 'N/A')}
"""
        
        prompt = f"""Anda adalah analis pasar saham & Bandarmology profesional di BEI/IDX. Analisis data broker flow untuk saham {ticker}:

**Status Bandarmologi:** {broksum_data.get('bandarmology_status', 'N/A')}

**Top Buyers (Pembeli Utama & Klasifikasi Broker):**
{buyers_text}
📊 Komposisi Pembeli: {buyers_cat_sum}

**Top Sellers (Penjual Utama & Klasifikasi Broker):**
{sellers_text}
📊 Komposisi Penjual: {sellers_cat_sum}

**Summary Narrative Saat Ini:**
{broksum_data.get('summary_narrative', 'N/A')}
{price_context}

Berikan analisis mendalam yang mencakup:
1. **Analisis Klasifikasi Broker (Bandar 🐋 vs Retail 🧑):** Siapa yang mendominasi pembeli (akumulasi) dan penjual (distribusi)? Apakah Bandar sedang menampung dari Retail atau sebaliknya?
2. **Implikasi Pergerakan Harga:** Pengaruh konfirmasi broker flow ini terhadap harga ke depan (bullish/bearish/neutral).
3. **Rekomendasi Aksi Investor:** Aksi praktis yang disarankan (Buy/Hold/Wait/Sell).
4. **Risk & Opportunity:** Tingkat risiko dan peluang berdasarkan peta kekuatan Bandar vs Retail.

Jadilah singkat, profesional, dan actionable (max 300 kata)."""

        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 500, "temperature": 0.7}
        )
        
        insight = response.text.strip() if response else ""
        return insight, None
        
    except Exception as e:
        return None, f"Error Gemini Analysis: {str(e)}"

def default_weight(factor, regime):
    defaults = {
        "STABLE BULLISH": {"Momentum":0.25,"AI_Senti":0.18,"MeanRev":0.12,"Beta_IHSG":0.15,"Coppock":0.30,"OFI": 0.12, "Bandar_Flow":0.20, "Foreign_ZScore":0.15},
        "VOLATILE UPTREND": {"Momentum":0.28,"AI_Senti":0.14,"MeanRev":0.12,"Beta_IHSG":0.16,"Coppock":0.30,"OFI": 0.10, "Bandar_Flow":0.18, "Foreign_ZScore":0.10},
        "HIGH-STRESS PANIC": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.22,"Beta_IHSG":0.15,"Coppock":0.30,"OFI": 0.18, "Bandar_Flow":0.22, "Foreign_ZScore":0.15},
        "SIDEWAYS / CONSOLIDATION": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.27,"Beta_IHSG":0.12,"Coppock":0.28,"OFI": 0.14, "Bandar_Flow":0.20, "Foreign_ZScore":0.12},
        "BEARISH ACCUMULATION": {"Momentum":0.20,"AI_Senti":0.18,"MeanRev":0.20,"Beta_IHSG":0.15,"Coppock":0.27,"OFI": 0.13, "Bandar_Flow":0.25, "Foreign_ZScore":0.15}
    }
    return defaults.get(regime, {"Momentum":0.23,"AI_Senti":0.17,"MeanRev":0.15,"Beta_IHSG":0.15,"Coppock":0.30,"OFI":0.10,"Bandar_Flow":0.15,"Foreign_ZScore":0.10}).get(factor,0.15)

# ---------- Coppock Curve ----------
def coppock_curve(prices, rP1=14, rP2=11, wP=10):
    if len(prices) < max(rP1,rP2)+wP+2: return 0.0,0.0
    roc1 = [(prices[i]-prices[i-rP1])/prices[i-rP1]*100 for i in range(rP1,len(prices))]
    roc2 = [(prices[i]-prices[i-rP2])/prices[i-rP2]*100 for i in range(rP2,len(prices))]
    mn = min(len(roc1),len(roc2))
    combined = [roc1[i]+roc2[i] for i in range(-mn,0)]
    def wma(data,per):
        if len(data)<per: return 0.0
        w = np.arange(1,per+1)
        vals = [np.dot(data[i:i+per],w)/w.sum() for i in range(len(data)-per+1)]
        return vals[-1]
    curr = wma(combined,wP)
    prev = wma(combined[:-1],wP) if len(combined)>wP else 0.0
    return curr,prev    
def hitung_bars_remaining(now_jkt, actual_interval, bars_per_day_map):
    h, m = now_jkt.hour, now_jkt.minute
    interval_minutes_map = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
    interval_menit = interval_minutes_map.get(actual_interval, 5)

    if h < 12 or (h == 12 and m == 0):
        menit_sesi1_tersisa = 12*60 - (h*60 + m)
        sisa_menit = menit_sesi1_tersisa + 90
        bars = math.ceil(sisa_menit / interval_menit)
    elif h == 12 or (h == 13 and m < 30):
        bars = math.ceil(90 / interval_menit)
    elif (h == 13 and m >= 30) or h == 14 or (h == 15 and m == 0):
        sisa_menit = 15*60 - (h*60 + m)
        bars = math.ceil(sisa_menit / interval_menit)
    else:
        bars = bars_per_day_map.get(actual_interval, 54)
    return max(1, bars)

# ---------- Adaptive Weights ----------
def get_adaptive_weights(ticker, regime, v12_mem=None):
    if v12_mem is not None:
        mem = v12_mem.get(ticker, {})
    else:
        try:
            mem = st.session_state.v12_memory.get(ticker, {})
        except (AttributeError, Exception):
            mem = {}
    defs = {k: default_weight(k, regime) for k in FACTOR_KEYS}
    w_pri = {}
    for k in FACTOR_KEYS:
        w = mem.get('weights',{}).get(k, defs[k])
        acc = mem.get('accuracy',{}).get(k,0.5)
        if acc>=0.65: w = min(w*1.15, WEIGHT_MAX)
        elif acc>=0.45: pass
        elif acc>=0.35: w *= 0.5
        else: w = max(w*0.2, WEIGHT_MIN/2)
        w_pri[k] = max(WEIGHT_MIN, min(WEIGHT_MAX, w))
        err = {k: mem.get('error_ema',{}).get(k,1.0) for k in FACTOR_KEYS}
    # Clamp error minimal supaya score tidak meledak
    scores = {k: 1.0/(max(err[k], 1e-3) + 1e-6) for k in FACTOR_KEYS}
    # Stable softmax: subtract max sebelum exp → identik matematis, no overflow
    max_score = max(scores.values()) if scores else 1.0
    exp_s = {k: math.exp((v - max_score) / SOFTMAX_TEMP) for k, v in scores.items()}
    sum_exp = sum(exp_s.values()) if sum(exp_s.values()) > 0 else 1.0
    sm = {k: v / sum_exp for k, v in exp_s.items()}
    final = {}
    for k in FACTOR_KEYS:
        bw = w_pri[k]; sw = max(0.10, sm[k])
        final[k] = max(WEIGHT_MIN, min(WEIGHT_MAX, 0.6*bw + 0.4*bw*sw*len(FACTOR_KEYS)))
    tot = sum(final.values())
    return {k: v/tot for k,v in final.items()}

def update_v12_memory(ticker, factor_signals, actual_return, volatility=0.02):
    if abs(actual_return) < 0.003:   
        return
    if ticker not in st.session_state.v12_memory:
        st.session_state.v12_memory[ticker] = {'weights':{},'accuracy':{},'error_ema':{}}
    mem = st.session_state.v12_memory[ticker]
    alpha = max(0.08, 0.20 if volatility>0.04 else (0.15 if volatility>0.02 else 0.10))
    ac = max(-1.0, min(1.0, actual_return))
    for k in FACTOR_KEYS:
        sv = max(-1.0, min(1.0, factor_signals.get(k,0.0)))
        err = abs(sv - ac)
        old = mem['error_ema'].get(k,1.0)
        mem['error_ema'][k] = old*(1-alpha) + err*alpha
    # Adaptive learning rate: cepat saat volatile, lambat saat tenang
    alpha_acc = 0.20 if volatility > 0.04 else (0.12 if volatility > 0.02 else 0.06)
    for k in FACTOR_KEYS:
        hit = 1.0 if factor_signals.get(k,0.0)*actual_return>0 else 0.0
        old_acc = mem['accuracy'].get(k,0.5)
        mem['accuracy'][k] = old_acc*(1-alpha_acc) + hit*alpha_acc   
    for k in FACTOR_KEYS:
        acc = mem['accuracy'][k]
        old_w = mem['weights'].get(k, default_weight(k,'SIDEWAYS'))
    
        # Drift proporsional terhadap seberapa jauh acc dari 0.5
        # acc = 0.75 → multiplier 1.10 ; acc = 0.90 → multiplier 1.25
        if acc >= 0.55:
            drift = min(0.25, (acc - 0.5) * 0.5)   # cap +25%
            new_w = min(old_w * (1 + drift), WEIGHT_MAX)
        elif acc <= 0.45:
            drift = min(0.25, (0.5 - acc) * 0.5)
            new_w = max(old_w * (1 - drift), WEIGHT_MIN)
        else:
            new_w = old_w
    
        mem['weights'][k] = new_w
    total_updates = mem.get('total_updates', 0) + 1
    mem['total_updates'] = total_updates
    if total_updates % 100 == 0:
        for k in FACTOR_KEYS:
            default_w = default_weight(k, 'SIDEWAYS')
            mem['weights'][k] = mem['weights'][k] * 0.85 + default_w * 0.15
    st.session_state.v12_memory[ticker] = mem
    save_v12_memory(st.session_state.v12_memory)

# ==========================================
# KONFIGURASI FILE RIWAYAT & SESSION STATE
# ==========================================
def bersihkan_untuk_json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

def simpan_riwayat(ringkasan, aksi_mode="simpan_baru", target_saham=None):
    if aksi_mode == "lewati":
        st.info("ℹ️ Riwayat tidak disimpan (Opsi 'Lewati Simpan' aktif untuk Swing Aktif).")
        return

    try:
        sheet = get_gsheet().worksheet("riwayat")
        items_to_add = ringkasan if isinstance(ringkasan, list) else [ringkasan]
        records = sheet.get_all_records()
        valid_records = [r for r in records if any(str(v).strip() for v in r.values())]
        data = list(valid_records)

        if aksi_mode == "update" and target_saham:
            saham_target_clean = str(target_saham).replace(".JK", "").strip().upper()
            for item_new in items_to_add:
                gaya_new = str(item_new.get("Gaya", "SW")).strip().upper()
                ringkasan_bersih = {k: bersihkan_untuk_json(v) for k, v in item_new.items()}
                updated = False
                for idx, record in enumerate(data):
                    s_rec = str(record.get("Saham", "")).replace(".JK", "").strip().upper()
                    g_rec = str(record.get("Gaya", "SW")).strip().upper()
                    if s_rec == saham_target_clean and g_rec == gaya_new:
                        data[idx] = ringkasan_bersih
                        updated = True
                        break
                if not updated:
                    data.insert(0, ringkasan_bersih)
        else:
            for item in reversed(items_to_add):
                ringkasan_bersih = {k: bersihkan_untuk_json(v) for k, v in item.items()}
                data.insert(0, ringkasan_bersih)

        data = data[:3000]
        if data:
            headers = list(data[0].keys())
            rows = [[row.get(h, "") for h in headers] for row in data]
            sheet.clear()
            sheet.update([headers] + rows, value_input_option='RAW')
        st.session_state.riwayat = data
        if aksi_mode == "update":
            st.success("✅ Catatan Swing Aktif berhasil diperbarui di riwayat!")
        else:
            st.success("✅ Riwayat berhasil disimpan!")
    except Exception as e:
        st.error(f"❌ Gagal menyimpan riwayat: {e}")

def muat_riwayat_dari_sheets():
    try:
        sheet = get_gsheet().worksheet("riwayat")
        records = sheet.get_all_records()
        valid_records = [r for r in records if any(str(v).strip() for v in r.values())]
        return valid_records[:3000]
    except Exception as e:
        st.error(f"❌ Gagal memuat riwayat: {e}")
        return []

def muat_riwayat_actual():
    data = {}
    # Normalisasi nilai mode dari berbagai format ke label pendek (SW/DT)
    def norm_gaya(val):
        v = str(val).strip().lower()
        if v in ('sw', 'swing'): return 'SW'
        if v in ('dt', 'daytrade', 'day_trade', 'day trade'): return 'DT'
        return val  # kembalikan apa adanya jika tidak dikenali

    try:
        sheet = get_gsheet().worksheet("riwayat_actual")
        records = sheet.get_all_records()
        for row in records:
            waktu = str(row.get('Waktu', ''))
            saham = str(row.get('Saham', ''))
            raw_gaya = row.get('Mode', '') or row.get('Gaya', '')
            gaya = norm_gaya(raw_gaya) if raw_gaya else ''

            # Isi nilai aktual
            val = {
                'Actual_High': str(row.get('Actual_High', '') or '').strip(),
                'Actual_Low': str(row.get('Actual_Low', '') or '').strip(),
                'Actual_Close': str(row.get('Actual_Close', '') or '').strip(),
                'Outcome': str(row.get('Outcome', '') or '').strip(),
                'Entry_Miss': str(row.get('Entry_Miss', '') or '').strip(),
                'Mode': gaya if gaya else '',
                'V12_Consumed': str(row.get('V12_Consumed', 'No') or 'No').strip()
            }

            if waktu and saham:
                if gaya:
                    data[(waktu, saham, gaya)] = val
                    mode_long = "swing" if gaya == "SW" else ("daytrade" if gaya == "DT" else gaya)
                    data[(waktu, saham, mode_long)] = val
                elif raw_gaya:
                    data[(waktu, saham, str(raw_gaya))] = val
                else:
                    # Legacy data tanpa spesifikasi mode
                    data[(waktu, saham)] = val
    except Exception as e:
        st.error(f"Gagal memuat actual: {e}")
    return data

def hitung_statistik_riwayat_actual(riwayat_actual):
    """Menghitung statistik Win Rate aktual dari st.session_state.riwayat_actual."""
    if not riwayat_actual or not isinstance(riwayat_actual, dict):
        return None
    
    seen_ids = set()
    total_win = 0
    total_loss = 0
    total_not_touched = 0
    
    win_sw, loss_sw = 0, 0
    win_dt, loss_dt = 0, 0
    
    for val in riwayat_actual.values():
        if not isinstance(val, dict):
            continue
        obj_id = id(val)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        
        outcome = val.get('Outcome', '')
        gaya = str(val.get('Mode', '')).upper()
        
        if outcome == 'Win':
            total_win += 1
            if gaya == 'SW':
                win_sw += 1
            elif gaya == 'DT':
                win_dt += 1
        elif outcome == 'Loss':
            total_loss += 1
            if gaya == 'SW':
                loss_sw += 1
            elif gaya == 'DT':
                loss_dt += 1
        elif outcome == 'Not Touched' or val.get('Entry_Miss') == 'Yes':
            total_not_touched += 1
            
    total_eval = total_win + total_loss
    
    eval_sw = win_sw + loss_sw
    wr_sw = (win_sw / eval_sw * 100) if eval_sw > 0 else None
    
    eval_dt = win_dt + loss_dt
    wr_dt = (win_dt / eval_dt * 100) if eval_dt > 0 else None
    
    if total_eval == 0:
        return {
            'total_eval': 0,
            'total_win': 0,
            'total_loss': 0,
            'total_not_touched': total_not_touched,
            'win_rate': None,
            'wr_sw': wr_sw,
            'eval_sw': eval_sw,
            'wr_dt': wr_dt,
            'eval_dt': eval_dt
        }
        
    win_rate = (total_win / total_eval) * 100
    
    return {
        'total_eval': total_eval,
        'total_win': total_win,
        'total_loss': total_loss,
        'total_not_touched': total_not_touched,
        'win_rate': win_rate,
        'wr_sw': wr_sw,
        'eval_sw': eval_sw,
        'wr_dt': wr_dt,
        'eval_dt': eval_dt
    }

def hitung_winrate_ticker_actual(ticker_raw, riwayat_actual):
    """Menghitung Win Rate riwayat_actual khusus untuk 1 ticker saham."""
    if not riwayat_actual or not isinstance(riwayat_actual, dict):
        return None
    
    ticker_clean = str(ticker_raw).replace(".JK", "").upper().strip()
    seen_ids = set()
    win, loss, not_touched = 0, 0, 0
    
    for key, val in riwayat_actual.items():
        if not isinstance(val, dict):
            continue
            
        saham_key = ""
        if isinstance(key, tuple) and len(key) >= 2:
            saham_key = str(key[1]).replace(".JK", "").upper().strip()
        elif isinstance(key, str):
            saham_key = str(val.get('Saham', '')).replace(".JK", "").upper().strip()
            
        if saham_key != ticker_clean:
            continue
            
        obj_id = id(val)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        
        outcome = val.get('Outcome', '')
        if outcome == 'Win':
            win += 1
        elif outcome == 'Loss':
            loss += 1
        elif outcome == 'Not Touched' or val.get('Entry_Miss') == 'Yes':
            not_touched += 1
            
    total = win + loss
    if total == 0:
        return {'win': 0, 'loss': 0, 'total': 0, 'win_rate': None, 'not_touched': not_touched}
    
    return {
        'win': win,
        'loss': loss,
        'total': total,
        'win_rate': (win / total) * 100,
        'not_touched': not_touched
    }

def hapus_riwayat_item(waktu, saham, gaya=None):
    """
    Hapus 1 item riwayat dari sheet — AMAN tanpa clear-all.
    Pakai delete_rows() untuk hapus baris spesifik.
    """
    try:
        sheet = get_gsheet().worksheet("riwayat")
        records = sheet.get_all_records()

        waktu_str = str(waktu).strip()
        saham_str = str(saham).strip()
        gaya_str = str(gaya).strip().upper() if gaya else None

        # ── Cari baris yang match (index di worksheet, mulai dari 2 karena header) ──
        rows_to_delete = []
        for i, r in enumerate(records):
            r_waktu = str(r.get('Waktu', '')).strip()
            r_saham = str(r.get('Saham', '')).strip()
            r_gaya = str(r.get('Gaya', '')).strip().upper()

            match = (r_waktu == waktu_str and r_saham == saham_str)
            if gaya_str:
                match = match and (r_gaya == gaya_str)

            if match:
                rows_to_delete.append(i + 2)  # +2: header di row 1, index mulai 2

        if not rows_to_delete:
            st.warning("⚠️ Item tidak ditemukan di riwayat.")
            return

        # ── Hapus dari BAWAH ke ATAS biar index gak shift ──
        for row_idx in sorted(rows_to_delete, reverse=True):
            sheet.delete_rows(row_idx)

        # ── Sync session_state ──
        try:
            fresh = sheet.get_all_records()
            valid = [r for r in fresh if any(str(v).strip() for v in r.values())]
            st.session_state.riwayat = valid
        except Exception:
            pass

        st.success(f"✅ {len(rows_to_delete)} item dihapus dari riwayat.")

    except Exception as e:
        st.error(f"❌ Gagal menghapus riwayat: {e}")
        
def simpan_riwayat_actual(waktu, saham, actual_data, mode="swing"):
    def norm_gaya(val):
        v = str(val).strip().lower()
        if v in ('sw', 'swing'): return 'SW'
        if v in ('dt', 'daytrade', 'day_trade', 'day trade'): return 'DT'
        return val

    try:
        sheet = get_gsheet().worksheet("riwayat_actual")
        records = sheet.get_all_records()
        headers_8 = ['Waktu', 'Saham', 'Mode', 'Actual_High', 'Actual_Low', 'Actual_Close', 'Outcome', 'Entry_Miss']
        headers_9 = headers_8 + ['V12_Consumed']

        # Pastikan sheet punya minimal 9 kolom (untuk kolom V12_Consumed di kolom I)
        if sheet.col_count < 9:
            sheet.add_cols(9 - sheet.col_count)

        # Jika belum ada data sama sekali, tulis header dulu
        if not records:
            sheet.update('A1:I1', [headers_9], value_input_option='RAW')
            records = sheet.get_all_records()
        else:
            existing_headers = list(records[0].keys())
            if 'Mode' not in existing_headers:
                sheet.update('A1:I1', [headers_9], value_input_option='RAW')
                records = sheet.get_all_records()
            elif 'V12_Consumed' not in existing_headers:
                # Header A-H sudah ada, tinggal tambah I1
                sheet.update('I1', [['V12_Consumed']], value_input_option='RAW')
                records = sheet.get_all_records()

        row_index = None
        v12_consumed = 'No'  # default untuk row baru
        target_mode_norm = norm_gaya(mode)
        for i, row in enumerate(records):
            r_mode = row.get('Mode') or row.get('Gaya') or ''
            if str(row.get('Waktu')) == str(waktu) and str(row.get('Saham')) == str(saham) and norm_gaya(r_mode) == target_mode_norm:
                row_index = i + 2
                v12_consumed = str(row.get('V12_Consumed', 'No')).strip()
                break
        new_row = [waktu, saham, mode,
                   actual_data.get('Actual_High', ''),
                   actual_data.get('Actual_Low', ''),
                   actual_data.get('Actual_Close', ''),
                   actual_data.get('Outcome', ''),
                   actual_data.get('Entry_Miss', '')]
        if row_index:
            sheet.update(f'A{row_index}:H{row_index}', [new_row], value_input_option='RAW')
        else:
            sheet.append_row(new_row, value_input_option='RAW')
        st.session_state.riwayat_actual = muat_riwayat_actual()
        if v12_consumed != 'Yes':
            integrate_actual_to_v12(waktu, saham, actual_data, mode=mode)
            # Mark V12_Consumed = Yes di kolom I
            if row_index:
                sheet.update(f'I{row_index}', [['Yes']], value_input_option='RAW')
            else:
                # Row baru = baris terakhir setelah append
                last_row = len(sheet.get_all_values())
                sheet.update(f'I{last_row}', [['Yes']], value_input_option='RAW')
    except Exception as e:
        st.error(f"Gagal menyimpan actual: {e}")

def hitung_hari_bursa(start_date, end_date):
    """Menghitung jumlah hari bursa (Senin-Jumat) antara 2 tanggal"""
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    if start_date >= end_date:
        return 0
    cur = start_date + timedelta(days=1)
    b_days = 0
    while cur <= end_date:
        if cur.weekday() < 5:
            b_days += 1
        cur += timedelta(days=1)
    return b_days

def fetch_actual_data_yfinance(saham, waktu_str):
    """
    Mengambil data High, Low, Close historis dari yfinance sejak tanggal sinyal s/d hari ini.
    Jika analisis dilakukan malam hari (setelah jam 16:00 WIB / pasar tutup),
    maka perhitungan actual data secara otomatis dimulai dari H+1 (hari bursa berikutnya).
    """
    try:
        ticker_input = saham if saham.endswith(".JK") else f"{saham}.JK"
        clean_waktu = str(waktu_str).strip()

        dt_obj = None
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d %B %Y, %H:%M WIB",
            "%d %B %Y, %H:%M:%S WIB",
            "%Y-%m-%d"
        ]

        for fmt in formats:
            try:
                dt_obj = datetime.strptime(clean_waktu, fmt)
                break
            except ValueError:
                continue

        if dt_obj is None:
            try:
                dt_part = clean_waktu.split()[0]
                dt_obj = datetime.strptime(dt_part, "%Y-%m-%d")
            except Exception:
                dt_obj = datetime.now()

        effective_date = dt_obj.date()
        # Jika analisis dibuat jam 16:00 WIB ke atas (setelah jam tutup bursa),
        # sinyal berlaku untuk hari bursa berikutnya (H+1).
        if dt_obj.hour >= 16:
            effective_date += timedelta(days=1)

        start_str = (effective_date - timedelta(days=1)).strftime("%Y-%m-%d")
        df_hist = yf.download(ticker_input, start=start_str, progress=False)
        if df_hist is None or df_hist.empty:
            return None

        if isinstance(df_hist.columns, pd.MultiIndex):
            try:
                df_hist = df_hist.xs(ticker_input, axis=1, level=1)
            except Exception:
                df_hist.columns = [c[0] for c in df_hist.columns]

        df_filtered = df_hist[df_hist.index.date >= effective_date]
        if df_filtered.empty:
            return None

        max_hi = float(df_filtered['High'].max())
        min_lo = float(df_filtered['Low'].min())
        last_cl = float(df_filtered['Close'].iloc[-1])

        return {
            'Actual_High': f"{max_hi:,.0f}".replace(",", ""),
            'Actual_Low': f"{min_lo:,.0f}".replace(",", ""),
            'Actual_Close': f"{last_cl:,.0f}".replace(",", "")
        }
    except Exception as e:
        return None

def dapatkan_sinyal_perlu_dicatat(riwayat_data, riwayat_actual):
    urgent_items = []
    active_swing_items = []
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_date = now_jkt.date()

    for r in riwayat_data:
        waktu_str = r.get('Waktu', '')
        saham = r.get('Saham', '')
        gaya = r.get('Gaya', 'SW')
        mode_actual = "swing" if gaya == "SW" else "daytrade"

        actual_data = (
            riwayat_actual.get((waktu_str, saham, gaya)) or
            riwayat_actual.get((waktu_str, saham, mode_actual)) or
            riwayat_actual.get((waktu_str, saham))
        )

        has_actual = False
        if actual_data:
            if (actual_data.get('Actual_High') or 
                actual_data.get('Actual_Low') or 
                actual_data.get('Actual_Close') or 
                actual_data.get('Outcome') or 
                actual_data.get('Entry_Miss') == 'Yes'):
                has_actual = True

        if has_actual:
            continue

        try:
            dt_sinyal = datetime.strptime(waktu_str.split()[0], "%Y-%m-%d").date()
        except:
            dt_sinyal = today_date

        b_days = hitung_hari_bursa(dt_sinyal, today_date)

        item = {
            'record': r,
            'waktu': waktu_str,
            'saham': saham,
            'gaya': gaya,
            'mode_actual': mode_actual,
            'b_days': b_days,
            'dt_sinyal': dt_sinyal
        }

        if gaya == "DT" or mode_actual == "daytrade":
            if dt_sinyal < today_date:
                item['alasan'] = f"Daytrade Sesi Sebelumnya ({waktu_str})"
                urgent_items.append(item)
        else:
            if b_days >= 7:
                item['alasan'] = f"Mencapai Batas Maksimal 7 Hari Bursa ({b_days} hari kerja)"
                urgent_items.append(item)
            elif b_days >= 1:
                item['alasan'] = f"Swing Berjalan (Hari bursa ke-{b_days})"
                active_swing_items.append(item)

    return urgent_items, active_swing_items

def dapatkan_dict_swing_aktif(riwayat_data=None, riwayat_actual=None):
    """
    Mengembalikan dict {saham_clean: item_info} untuk emiten-emiten yang memiliki
    posisi Swing (SW) aktif (belum diisi outcome actual dan belum kadaluarsa/lewat 7 hari bursa).
    """
    if riwayat_data is None:
        riwayat_data = st.session_state.get('riwayat', [])
    if riwayat_actual is None:
        riwayat_actual = st.session_state.get('riwayat_actual', {})

    active_map = {}
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_date = now_jkt.date()

    for r in riwayat_data:
        waktu_str = str(r.get('Waktu', ''))
        saham = str(r.get('Saham', '')).strip().upper()
        saham_clean = saham.replace(".JK", "")
        gaya = str(r.get('Gaya', 'SW')).strip().upper()

        if gaya not in ('SW', 'SWING'):
            continue

        mode_actual = "swing"

        actual_data = (
            riwayat_actual.get((waktu_str, saham, gaya)) or
            riwayat_actual.get((waktu_str, saham_clean, gaya)) or
            riwayat_actual.get((waktu_str, saham, mode_actual)) or
            riwayat_actual.get((waktu_str, saham_clean, mode_actual)) or
            riwayat_actual.get((waktu_str, saham)) or
            riwayat_actual.get((waktu_str, saham_clean))
        )

        has_actual = False
        if actual_data:
            if (actual_data.get('Actual_High') or 
                actual_data.get('Actual_Low') or 
                actual_data.get('Actual_Close') or 
                actual_data.get('Outcome') or 
                actual_data.get('Entry_Miss') == 'Yes'):
                has_actual = True

        if has_actual:
            continue

        try:
            dt_sinyal = datetime.strptime(waktu_str.split()[0], "%Y-%m-%d").date()
        except:
            dt_sinyal = today_date

        b_days = hitung_hari_bursa(dt_sinyal, today_date)

        if b_days < 7 and saham_clean not in active_map:
            active_map[saham_clean] = {
                'record': r,
                'waktu': waktu_str,
                'saham': saham_clean,
                'gaya': gaya,
                'b_days': b_days,
                'dt_sinyal': dt_sinyal
            }

    return active_map

def get_dip_entry(r):
    """
    Mengembalikan string harga entry ideal / dip entry (misal 'Rp 5,000') dari record riwayat.
    Jika 'Entry_Ideal_RRR2' tidak tersedia, dihitung secara otomatis dari TP dan SL.
    """
    if not r or not isinstance(r, dict):
        return ""
    val = r.get('Entry_Ideal_RRR2', '')
    if val and str(val).strip() and str(val).strip() != '?':
        val_str = str(val).strip()
        return val_str if val_str.startswith("Rp") else f"Rp {val_str}"
    
    # Fallback: Hitung dari TP_Harga/TP_Range dan SL_Harga jika belum tersimpan
    try:
        tp_str = r.get('TP_Harga') or r.get('TP_Range', '')
        sl_str = r.get('SL_Harga', '')
        
        clean_tp = str(tp_str).replace("Rp", "").replace(".", "").replace(",", "").strip()
        clean_sl = str(sl_str).replace("Rp", "").replace(".", "").replace(",", "").strip()
        
        tp_matches = re.findall(r'\d+', clean_tp)
        sl_matches = re.findall(r'\d+', clean_sl)
        
        if tp_matches and sl_matches:
            tp_val = float(tp_matches[0])
            sl_val = float(sl_matches[0])
            if tp_val > sl_val:
                dip_val = (tp_val + 2 * sl_val) / 3.0
                return f"Rp {dip_val:,.0f}"
    except Exception:
        pass
    return ""

def render_notifikasi_evaluasi_riwayat():
    riwayat_data = st.session_state.get('riwayat', [])
    riwayat_actual = st.session_state.get('riwayat_actual', {})

    if not riwayat_data:
        return

    urgent_items, active_swing_items = dapatkan_sinyal_perlu_dicatat(riwayat_data, riwayat_actual)

    if not urgent_items and not active_swing_items:
        return

    n_urgent = len(urgent_items)
    n_active = len(active_swing_items)

    st.markdown("""
        <style>
        .notif-box {
            background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
            border-left: 5px solid #a855f7;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    title_text = "🔔 <b>Pengingat Evaluasi Outcome Trading</b>"
    details = []
    if n_urgent > 0:
        details.append(f"⚠️ <b>{n_urgent} sinyal perlu dicatat</b> (Daytrade atau Swing ≥7 hari bursa)")
    if n_active > 0:
        details.append(f"⏳ <b>{n_active} Swing aktif</b> (1-6 hari bursa)")

    st.markdown(f"""
    <div class="notif-box">
        <div style="font-size:15px; font-weight:bold; color:#f472b6;">
            {title_text}
        </div>
        <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">
            {' | '.join(details)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📝 Form Evaluasi Sinyal (Quick Outcome Journal)", expanded=(n_urgent > 0)):
        tab_urgent, tab_active = st.tabs([
            f"🚨 Perlu Catat Immediate ({n_urgent})",
            f"⏳ Swing Aktif ({n_active})"
        ])

        with tab_urgent:
            if not urgent_items:
                st.success("🎉 Semua sinyal jatuh tempo sudah dicatat!")
            else:
                for idx, item in enumerate(urgent_items):
                    r = item['record']
                    waktu_key = item['waktu']
                    saham_key = item['saham']
                    gaya_key = item['gaya']
                    mode_actual = item['mode_actual']
                    alasan = item['alasan']
                    dip_entry = get_dip_entry(r)
                    dip_str = f" | 💡 Dip Entry: {dip_entry}" if dip_entry else ""

                    st.markdown(f"**📌 {saham_key} ({gaya_key}) - {waktu_key}** | `{alasan}`")
                    st.caption(f"Sinyal: {r.get('Sinyal','?')} | 🎯 Entry: {r.get('Entry_Zone','?')}{dip_str} | TP: {r.get('TP_Range','?')} | SL: Rp {r.get('SL_Harga','?')}")

                    fetch_key = f"fetch_urg_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                    form_key = f"form_urg_{idx}_{waktu_key}_{saham_key}_{gaya_key}"

                    col_auto, _ = st.columns([2, 1])
                    with col_auto:
                        if st.button(f"⚡ Fetch Otomatis Data Harga ({saham_key})", key=fetch_key):
                            fetched = fetch_actual_data_yfinance(saham_key, waktu_key)
                            if fetched:
                                st.session_state[f"hi_{fetch_key}"] = fetched['Actual_High']
                                st.session_state[f"lo_{fetch_key}"] = fetched['Actual_Low']
                                st.session_state[f"cl_{fetch_key}"] = fetched['Actual_Close']
                                msg = f"Data harga {saham_key} berhasil ditarik!"
                                if dip_entry:
                                    msg += f" (Dip Entry Target: {dip_entry})"
                                st.success(msg)
                            else:
                                st.error(f"Gagal mengambil data {saham_key} dari yfinance")

                    with st.form(key=form_key):
                        if dip_entry:
                            st.caption(f"💡 Target Dip Entry: **{dip_entry}** | 🎯 Entry Zone: **{r.get('Entry_Zone', '-')}**")
                        def_hi = st.session_state.get(f"hi_{fetch_key}", "")
                        def_lo = st.session_state.get(f"lo_{fetch_key}", "")
                        def_cl = st.session_state.get(f"cl_{fetch_key}", "")

                        c1, c2, c3 = st.columns(3)
                        actual_high = c1.text_input("Actual High", value=def_hi, placeholder="contoh: 5350")
                        actual_low = c2.text_input("Actual Low", value=def_lo, placeholder="contoh: 5050")
                        actual_close = c3.text_input("Actual Close", value=def_cl, placeholder="contoh: 5200")

                        c4, c5 = st.columns(2)
                        entry_miss = c4.checkbox("🚫 Entry Tidak Tersentuh", value=False)
                        if entry_miss:
                            outcome = "Not Touched"
                        else:
                            outcome = c5.selectbox("Outcome", ["", "Win", "Loss", "Not Touched"], format_func=lambda x: "Pilih Outcome" if x=="" else x)

                        submitted = st.form_submit_button("💾 Simpan Outcome")
                        if submitted:
                            if not entry_miss and outcome == "":
                                st.error("Pilih Outcome terlebih dahulu.")
                            else:
                                data = {
                                    'Actual_High': actual_high.strip(),
                                    'Actual_Low': actual_low.strip(),
                                    'Actual_Close': actual_close.strip(),
                                    'Outcome': outcome,
                                    'Entry_Miss': 'Yes' if entry_miss else 'No',
                                    'Mode': mode_actual
                                }
                                simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                st.success(f"✅ Outcome {saham_key} berhasil disimpan!")
                                st.rerun()
                    st.divider()

        with tab_active:
            if not active_swing_items:
                st.info("Tidak ada posisi Swing aktif (1-6 hari bursa) yang sedang berjalan.")
            else:
                for idx, item in enumerate(active_swing_items):
                    r = item['record']
                    waktu_key = item['waktu']
                    saham_key = item['saham']
                    gaya_key = item['gaya']
                    mode_actual = item['mode_actual']
                    alasan = item['alasan']
                    dip_entry = get_dip_entry(r)
                    dip_str = f" | 💡 Dip Entry: {dip_entry}" if dip_entry else ""

                    st.markdown(f"**⏳ {saham_key} ({gaya_key}) - {waktu_key}** | `{alasan}`")
                    st.caption(f"Sinyal: {r.get('Sinyal','?')} | 🎯 Entry: {r.get('Entry_Zone','?')}{dip_str} | TP: {r.get('TP_Range','?')} | SL: Rp {r.get('SL_Harga','?')}")

                    fetch_key = f"fetch_act_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                    form_key = f"form_act_{idx}_{waktu_key}_{saham_key}_{gaya_key}"

                    col_auto, _ = st.columns([2, 1])
                    with col_auto:
                        if st.button(f"⚡ Fetch Otomatis Data Harga ({saham_key})", key=fetch_key):
                            fetched = fetch_actual_data_yfinance(saham_key, waktu_key)
                            if fetched:
                                st.session_state[f"hi_{fetch_key}"] = fetched['Actual_High']
                                st.session_state[f"lo_{fetch_key}"] = fetched['Actual_Low']
                                st.session_state[f"cl_{fetch_key}"] = fetched['Actual_Close']
                                msg = f"Data harga {saham_key} berhasil ditarik!"
                                if dip_entry:
                                    msg += f" (Dip Entry Target: {dip_entry})"
                                st.success(msg)
                            else:
                                st.error(f"Gagal mengambil data {saham_key} dari yfinance")

                    with st.form(key=form_key):
                        if dip_entry:
                            st.caption(f"💡 Target Dip Entry: **{dip_entry}** | 🎯 Entry Zone: **{r.get('Entry_Zone', '-')}**")
                        def_hi = st.session_state.get(f"hi_{fetch_key}", "")
                        def_lo = st.session_state.get(f"lo_{fetch_key}", "")
                        def_cl = st.session_state.get(f"cl_{fetch_key}", "")

                        c1, c2, c3 = st.columns(3)
                        actual_high = c1.text_input("Actual High", value=def_hi, placeholder="contoh: 5350")
                        actual_low = c2.text_input("Actual Low", value=def_lo, placeholder="contoh: 5050")
                        actual_close = c3.text_input("Actual Close", value=def_cl, placeholder="contoh: 5200")

                        c4, c5 = st.columns(2)
                        entry_miss = c4.checkbox("🚫 Entry Tidak Tersentuh", value=False)
                        if entry_miss:
                            outcome = "Not Touched"
                        else:
                            outcome = c5.selectbox("Outcome", ["", "Win", "Loss", "Not Touched"], format_func=lambda x: "Pilih Outcome" if x=="" else x)

                        submitted = st.form_submit_button("💾 Simpan Outcome (Early Exit)")
                        if submitted:
                            if not entry_miss and outcome == "":
                                st.error("Pilih Outcome terlebih dahulu.")
                            else:
                                data = {
                                    'Actual_High': actual_high.strip(),
                                    'Actual_Low': actual_low.strip(),
                                    'Actual_Close': actual_close.strip(),
                                    'Outcome': outcome,
                                    'Entry_Miss': 'Yes' if entry_miss else 'No',
                                    'Mode': mode_actual
                                }
                                simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                st.success(f"✅ Outcome {saham_key} berhasil disimpan!")
                                st.rerun()
                    st.divider()

def integrate_actual_to_v12(waktu, saham, actual_data, mode="swing"):
    try:
        ticker = saham
        last_pred = load_v12_predictions(ticker, mode=mode)
        if not last_pred:
            return

        factor_signals = {}
        for k in FACTOR_KEYS:
            key = f'sig_{k}'
            if key in last_pred:
                factor_signals[k] = float(last_pred[key])
            else:
                factor_signals[k] = 0.0

        # --- 1) Update arah prediksi berdasarkan Actual Close ---
        actual_close_str = actual_data.get('Actual_Close', '')
        if actual_close_str:
            try:
                actual_close = float(str(actual_close_str).replace(",", ""))
                last_close = safe_float(last_pred.get('close_price'), 0.0)
                if last_close > 0:
                    actual_return = (actual_close - last_close) / last_close
                    actual_return = max(-1.0, min(1.0, actual_return))
                    SIGNAL_NOISE_FLOOR = 0.003   # 0.3%
                    if abs(actual_return) < SIGNAL_NOISE_FLOOR:
                        # Tidak ada sinyal riil — jangan update memory
                        pass
                    else:
                        # Volatility adaptif dari data
                        _vol = factor_signals.get('_volatility', 0.02)
                        update_v12_memory(ticker, factor_signals, actual_return, volatility=_vol)
            except:
                pass  # gagal parse → arah tidak diupdate

        # --- 2) Belajar dari Entry Miss / Not Touched ---
        # Hanya berjalan jika prediksi sebelumnya menyimpan entry_low & entry_high
        entry_low = last_pred.get('entry_low')
        entry_high = last_pred.get('entry_high')
        if entry_low is not None and entry_high is not None:
            try:
                entry_low_f = safe_float(entry_low, None)
                entry_high_f = safe_float(entry_high, None)
            except:
                entry_low_f = None
                entry_high_f = None

            if entry_low_f is not None and entry_high_f is not None and entry_low_f < entry_high_f:
                gap = None

                # --- Path A: User mengisi Actual Low → hitung gap dari data nyata ---
                actual_low_str = actual_data.get('Actual_Low', '')
                if actual_low_str:
                    try:
                        actual_low_f = float(str(actual_low_str).replace(",", ""))
                        # Jika actual low > entry_high, harga tidak pernah menyentuh zona entry
                        if actual_low_f > entry_high_f:
                            gap = actual_low_f - entry_high_f
                    except:
                        pass

                # --- Path B: User centang "Entry Tidak Tersentuh" (Entry_Miss=Yes)
                #     tanpa mengisi Actual Low → estimasi gap dari selisih close price
                #     prediksi terakhir vs entry_high (fallback konservatif) ---
                if gap is None and actual_data.get('Entry_Miss', '') == 'Yes':
                    last_close = safe_float(last_pred.get('close_price'), 0.0)
                    if last_close > entry_high_f:
                        # Harga penutupan sudah di atas entry_high → gap = selisihnya
                        gap = last_close - entry_high_f
                    else:
                        # Tidak bisa estimasi gap dengan pasti, gunakan nilai kecil
                        # agar engine tahu ada miss tapi tidak over-koreksi
                        gap = entry_high_f * 0.01  # 1% dari entry_high sebagai proxy

                if gap is not None and gap > 0:
                    mem = st.session_state.v12_memory.get(ticker, {})
                    if 'entry_error_ema' not in mem:
                        mem['entry_error_ema'] = 0.0

                    alpha = 0.2
                    mem['entry_error_ema'] = (
                        mem['entry_error_ema'] * (1 - alpha) + gap * alpha
                    )

                    st.session_state.v12_memory[ticker] = mem
                    save_v12_memory(st.session_state.v12_memory)
    except Exception as e:
        st.error(f"Gagal integrasi V12: {e}")
# ====================== API IDX ======================
@st.cache_data(ttl=86400)   # cache 1 hari
def fetch_idx_stock_list_exclude_monitoring():
    excluded_boards = {
        "pemantauankhusus", "pemantauan_khusus", "pemantauankhusus",
        "monitoring", "special_monitoring", "specialmonitoring"
    }
    endpoints = [
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?start=0&length=9999",
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?language=id-id&start=0&length=9999",
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?start=0&length=9999&exchangeBoard=&industry=&subIndustry=&search="
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.idx.co.id/",
        "X-Requested-With": "XMLHttpRequest"
    }

    all_codes = []
    seen = set()
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=25)
            if resp.status_code != 200:
                continue
            raw = resp.json()

            # Cari key yang berisi list saham
            items = None
            if isinstance(raw, list):
                items = raw
            else:
                for key in ('data', 'Data', 'result', 'Result', 'results'):
                    if key in raw:
                        items = raw[key]
                        break
                if items is None:
                    # Coba ekstrak rekursif
                    def extract(obj):
                        out = []
                        if isinstance(obj, dict):
                            if 'code' in obj or 'Code' in obj or 'KodeSaham' in obj:
                                out.append(obj)
                            for v in obj.values():
                                out.extend(extract(v))
                        elif isinstance(obj, list):
                            for v in obj:
                                out.extend(extract(v))
                        return out
                    items = extract(raw)

            if not items:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                code = item.get('Code') or item.get('code') or item.get('KodeSaham')
                if not code:
                    continue
                code = str(code).strip().upper()
                if len(code) > 6 or not code.isalnum():
                    continue

                board = item.get('BoardId') or item.get('Board') or item.get('boardId') or ""
                board_lower = str(board).lower().replace(" ", "").replace("-", "").replace("_", "")
                if board_lower in excluded_boards:
                    continue

                if code not in seen:
                    seen.add(code)
                    all_codes.append(code)

            if len(all_codes) >= 100:
                break
        except Exception as e:
            # st.write(f"Error endpoint {url}: {e}")   # debug
            continue

    return sorted(all_codes) if all_codes else None
def fetch_all_idx_stocks():
    """Ambil semua saham BEI non-Pemantauan Khusus."""
    return fetch_idx_stock_list_exclude_monitoring()
# ==========================================
# FUNGSI AI GEMINI
# ==========================================
def dapatkan_model_gemini(api_key):
    if not api_key: return None, "API key belum diisi."
    try:
        genai.configure(api_key=api_key)
        available = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available: return None, "Tidak ada model Gemini."
        for model_id in available:
            try:
                model = genai.GenerativeModel(model_id)
                model.generate_content("test", generation_config={"max_output_tokens": 1})
                return model, None
            except Exception: continue
        return None, "Model gagal digunakan."
    except Exception as e:
        return None, f"Error: {str(e)}"
def analisis_saham_dengan_ai(data_saham, riwayat, api_key, ticker=None):
    """
    Analisis saham dengan Gemini AI.
    ticker (optional): untuk fetch broker flow dari database & inject ke analysis
    """
    model, error = dapatkan_model_gemini(api_key)
    if error: return None, error
    
    # ===== FORMAT RIWAYAT (EXISTING) =====
    riwayat_text = ""
    if riwayat:
        riwayat_text = "Riwayat analisis sebelumnya (termasuk hasil aktual jika tersedia):\n"
        for r in riwayat:  # sudah difilter per emiten oleh pemanggil
            base = f"- {r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | RRR: {r['RRR']} | Rezim: {r['Rezim']}"
            # Tambahkan data aktual
            if r.get('Actual_High') or r.get('Actual_Outcome'):
                base += " | Hasil Aktual: "
                if r.get('Actual_High'):
                    base += f"High={r['Actual_High']}, "
                if r.get('Actual_Low'):
                    base += f"Low={r['Actual_Low']}, "
                if r.get('Actual_Close'):
                    base += f"Close={r['Actual_Close']}, "
                if r.get('Actual_Outcome'):
                    base += f"Outcome={r['Actual_Outcome']}"
                if r.get('Entry_Miss'):
                    base += " (Entry tidak tersentuh)"
            ai_insight = r.get("AI_Insight", "").strip()
            if ai_insight:
                short_insight = (ai_insight[:120] + "...") if len(ai_insight) > 120 else ai_insight
                base += f" | AI Insight: {short_insight}"
            riwayat_text += base + "\n"
    else:
        riwayat_text = "Belum ada riwayat sebelumnya."

    # ===== LOAD SEMUA HISTORY BROKER FLOW DARI DATABASE =====
    broksum_context = ""
    if ticker:
        try:
            history = load_broksum_history(ticker)
            if history:
                # Sort terbaru dulu
                history_sorted = sorted(
                    history,
                    key=lambda r: str(r.get('upload_date', '')),
                    reverse=True
                )

                # Ambil maks 5 terbaru
                broksum_entries = []
                for h in history_sorted[:5]:
                    buyers_list = json.loads(h.get('top_buyers', '[]')) if isinstance(h.get('top_buyers'), str) else h.get('top_buyers', [])
                    sellers_list = json.loads(h.get('top_sellers', '[]')) if isinstance(h.get('top_sellers'), str) else h.get('top_sellers', [])

                    buyers_text, buyers_sum = format_broker_list_for_ai(buyers_list)
                    sellers_text, sellers_sum = format_broker_list_for_ai(sellers_list)

                    broksum_entries.append(
                        f"**Upload {h.get('upload_date', 'N/A')}**\n"
                        f"  Status Bandarmologi: {h.get('bandarmology_status', 'N/A')}\n"
                        f"  Pembeli (Komposisi: {buyers_sum}):\n{buyers_text}\n"
                        f"  Penjual (Komposisi: {sellers_sum}):\n{sellers_text}\n"
                        f"  Summary: {h.get('summary_narrative', 'N/A')[:180]}"
                    )

                broksum_context = (
                    f"**🕵🏻‍♂️ Bandarmology (Broker Flow) — {len(history_sorted)} snapshot terakhir**\n\n"
                    + "\n\n".join(broksum_entries)
                )

                if len(history_sorted) > 1:
                    broksum_context += (
                        f"\n\n**⚠️ PENTING:** Ada {len(history_sorted)} snapshot. "
                        f"Bandingkan perubahan dominasi Bandar 🐋 vs Retail 🧑 antar waktu "
                        f"untuk mendeteksi pola akumulasi atau distribusi secara presisi."
                    )
        except Exception as e:
            pass

    prompt = f"""
Anda adalah asisten analis saham profesional & pakar Bandarmology. Berikut data analisis teknikal, fundamental, dan broker flow saham {data_saham['Saham']}:

- Harga terakhir: Rp {data_saham['Harga']}
- Sinyal saat ini: {data_saham['Sinyal']}
- Rezim Pasar: {data_saham['Rezim']}
- Sentimen Berita: {data_saham['Sentimen']}
- Risk/Reward Ratio (RRR): {data_saham['RRR']}
- Probabilitas Naik: {data_saham['Prob Naik']}
- Take Profit: +{data_saham['TP%']}%
- Stop Loss: -{data_saham['SL%']}%
- Estimasi: Rp {data_saham['Estimasi']}
- Beta terhadap IHSG: {data_saham.get('Beta', 'N/A')}
- Win Rate Backtest: {data_saham.get('WinRate', 'N/A')}
- Actual Track Record Saham Ini: {data_saham.get('Actual_WinRate_Ticker', 'Belum ada evaluasi')}
- Profit Factor Backtest: {data_saham.get('ProfitFactor', 'N/A')}
- Max Drawdown Backtest: {data_saham.get('MaxDD', 'N/A')}
- Alokasi Kelly Maks: {data_saham.get('Kelly', 'N/A')}%
- Fundamental: Market Cap: {data_saham.get('Fundamental_MC', 'N/A')}, PER: {data_saham.get('Fundamental_PER', 'N/A')}, PBV: {data_saham.get('Fundamental_PBV', 'N/A')}, ROE: {data_saham.get('Fundamental_ROE', 'N/A')}, D/E: {data_saham.get('Fundamental_DE', 'N/A')}
- Status Posisi: {data_saham.get('Status_Posisi', 'Tidak diketahui')}
- Harga Beli: {data_saham.get('Harga_Beli', 'Tidak diisi')}
- Floating P/L: {data_saham.get('Floating_PL', 'N/A')}

{broksum_context}

{riwayat_text}

Berdasarkan data di atas{' (khususnya dominasi Bandar 🐋 vs Retail 🧑 pada broker flow)' if broksum_context else ''}, berikan analisis ringkas (Bahasa Indonesia) yang mencakup:
- Makna sinyal teknikal dalam konteks pergerakan saat ini
{f'- Analisis peta akumulasi/distribusi broker (Bandar vs Retail) & implikasinya pada harga' if broksum_context else ''}
- Kekuatan dan kelemahan saham
- Risiko utama
- Rekomendasi langkah selanjutnya (buy/hold/sell) dengan alasan singkat
- Jika ada pola dari riwayat, sebutkan.
Gunakan bahasa mudah dipahami trader, maksimal 4 paragraf pendek.
"""
        # ── Retry logic untuk handle 500/503 dari Google ──
    import time as _time
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip(), None
        except Exception as e:
            err_str = str(e)
            last_error = err_str
            
            # Cek apakah error sementara (500/503/overloaded)
            is_transient = any(x in err_str for x in [
                "500", "503", "Internal error", 
                "overloaded", "temporarily", "try again"
            ])
            
            if is_transient and attempt < max_retries - 1:
                _time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            else:
                break
    
    return None, f"Gagal menghasilkan insight AI: {last_error}"

def analisis_riwayat_global(riwayat_data, riwayat_actual, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error: return None, error
    if not riwayat_data: return None, "Belum ada riwayat."
    prompt = "Berikut adalah riwayat analisis saham yang telah dilakukan (termasuk hasil aktual jika tersedia):\n\n"
    for r in riwayat_data[:30]:
        gaya_label = "📆 SW" if r.get('Gaya') == "SW" else "⏱️ DT"
        base = f"- {r['Waktu']}|{gaya_label}|{r['Saham']}|Sinyal:{r['Sinyal']}|Harga:{r['Harga']}|RRR:{r['RRR']}|Sentimen:{r['Sentimen']}|Rezim:{r['Rezim']}|TP%:{r['TP%']}%|SL%:{r['SL%']}%"

        # Tambahkan data aktual jika tersedia
        gaya = r.get('Gaya', 'SW')
        # Coba key 3 elemen (dengan gaya), fallback ke key 2 elemen (data lama)
        key_actual = (r.get('Waktu'), r.get('Saham'), gaya)
        actual = riwayat_actual.get(key_actual) or riwayat_actual.get((r.get('Waktu'), r.get('Saham')), {})
        if actual:
            base += " | Hasil Aktual: "
            details = []
            if actual.get('Actual_High'):
                details.append(f"High={actual['Actual_High']}")
            if actual.get('Actual_Low'):
                details.append(f"Low={actual['Actual_Low']}")
            if actual.get('Actual_Close'):
                details.append(f"Close={actual['Actual_Close']}")
            if actual.get('Outcome'):
                details.append(f"Outcome={actual['Outcome']}")
            if actual.get('Entry_Miss') == 'Yes':
                details.append("Entry Tidak Tersentuh")
            base += ", ".join(details)

        prompt += base + "\n"

    prompt += (
        "\nBerdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia):\n"
        "- Pola sinyal yang sering muncul\n"
        "- Saham dengan peluang terbaik menurut data (termasuk hasil aktualnya)\n"
        "- Rekomendasi perbaikan strategi\n"
        "- Insight tambahan yang berguna untuk trader\n"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks: return teks
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    teks = teks.replace('\n', '<br>')
    return teks

def evaluasi_mode_dengan_ai(res_swing, res_day, ticker_raw, api_key):
    """
    Evaluasi AI Gemini untuk membandingkan kesesuaian mode Swing Trade vs Day Trade (Hybrid Scoring).
    """
    model, error = dapatkan_model_gemini(api_key)
    if error or not model:
        return None, error

    prompt = f"""
Anda adalah analis kuantitatif pasar saham profesional. Evaluasi emiten {ticker_raw} untuk menentukan apakah lebih cocok diperdagangkan secara **Swing Trade (SW)** atau **Day Trade (DT)**.

DATA SWING TRADE:
- Sinyal: {res_swing.get('signal', 'N/A')}
- RRR: {res_swing.get('rrr', 0):.2f}
- Confidence: {res_swing.get('confidence', 0):.1f}%
- Win Rate Backtest: {res_swing.get('win_bt', 0)*100 if res_swing.get('win_bt') else 0:.1f}%

DATA DAY TRADE:
- Sinyal: {res_day.get('signal', 'N/A')}
- RRR: {res_day.get('rrr', 0):.2f}
- Confidence: {res_day.get('confidence', 0):.1f}%
- Win Rate Backtest: {res_day.get('win_bt', 0)*100 if res_day.get('win_bt') else 0:.1f}%

KONTEKS EMITEN:
- Beta IHSG: {res_swing.get('beta_ihsg', 1.0):.2f}
- Harga Terakhir: Rp {res_swing.get('harga_terakhir', 0):,.0f}

TUGAS ANDA:
Berikan evaluasi dalam format JSON murni dengan struktur persis seperti ini (tanpa tanda petik ganda di dalam string reasoning):
{{
  "swing_score": 60,
  "day_score": 80,
  "recommended_mode": "Day Trade",
  "reasoning": "Alasan singkat 1-2 kalimat tanpa tanda petik ganda."
}}
"""
    try:
        gen_config = {"response_mime_type": "application/json"}
        try:
            response = model.generate_content(prompt, generation_config=gen_config)
        except Exception:
            response = model.generate_content(prompt)

        raw_text = response.text.strip()

        # Tier 1: Direct JSON load
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict) and 'swing_score' in data:
                return data, None
        except Exception:
            pass

        # Tier 2: Extract specific JSON object using regex
        match = re.search(r'\{[^{}]*"swing_score"[^{}]*\}', raw_text, re.DOTALL)
        if not match:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str, strict=False)
                if isinstance(data, dict) and 'swing_score' in data:
                    return data, None
            except Exception:
                pass

        # Tier 3: Regex fallbacks for fields
        swing_match = re.search(r'"swing_score"\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        day_match = re.search(r'"day_score"\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw_text)

        if swing_match and day_match:
            data = {
                "swing_score": float(swing_match.group(1)),
                "day_score": float(day_match.group(1)),
                "reasoning": reason_match.group(1) if reason_match else "Evaluasi AI Gemini untuk kesesuaian mode."
            }
            return data, None

        return None, "Format JSON AI tidak dapat diparse"
    except Exception as e:
        return None, str(e)

# ==========================================
# FUNGSI BANDARMOLOGY & BROKSUM (GEMINI VISION)
# ==========================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def _call_gemini_vision_api(model, prompt, image):
    """Internal function untuk Gemini Vision call dengan retry logic."""
    return model.generate_content([prompt, image])

def analisis_broksum_gemini_vision(image, api_key):
    """
    Menganalisis screenshot Broker Summary (Broksum) / Trade Flow / Broker Flow menggunakan Gemini Vision AI.
    - Kompresi gambar untuk hemat token
    - Retry dengan exponential backoff untuk handle rate limit
    - Prioritas model: gemini-1.5-flash (cepat & murah)
    Mengembalikan dict data terstruktur (JSON) & error jika ada.
    """
    if not PIL_AVAILABLE:
        return None, "Library Pillow (PIL) belum terpasang."
    if not api_key:
        return None, "Gemini API Key belum diisi di sidebar."

    try:
        genai.configure(api_key=api_key)
        
        # ===== KOMPRESI GAMBAR (Hemat Token) =====
        compressed_image = compress_image_for_gemini(image, max_width=1280, max_height=960, quality=85)
        
        # ===== PILIH MODEL (Prioritas Flash - Gunakan yang Available) =====
        available = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        vision_candidates = [
            'gemini-3.1-flash-lite',    # ⚡ Paling cepat
            'gemini-3.5-flash-lite',    # ⚡ Cepat
            'gemini-1.5-flash-lite',    # ⚡ Lite fallback
            'gemini-3-flash',           # Regular (jika lite tidak ada)
            'gemini-3.8-flash',         # 3.8
            'gemini-3.7-flash',         # 3.7
            'gemini-2.0-flash',         # Fallback 2.0
            'gemini-1.5-flash'          # Fallback 1.5
        ]
        
        selected_model_name = None
        for cand in vision_candidates:
            if cand in available:
                selected_model_name = cand
                break
        if not selected_model_name:
            if available:
                selected_model_name = available[0]
            else:
                return None, "Tidak ada model Gemini yang mendukung eksekusi saat ini."

        model = genai.GenerativeModel(selected_model_name)

        prompt = """
Anda adalah pakar Bandarmology & Pasar Modal Indonesia (BEI/IDX).
Tugas Anda: Analisis screenshot Broker Summary (Broksum) / Broker Flow / Trade Flow ini dengan SANGAT AKURAT.

Ekstrak seluruh informasi tabel dan berikan analisis terstruktur dalam format JSON MURNI tanpa teks di luar JSON:
{
  "ticker": "KODE_SAHAM (contoh: BBRI, tulis N/A jika tidak terlihat)",
  "periode": "TANGGAL / PERIODE (contoh: 09 Sep 2026, tulis N/A jika tidak terlihat)",
  "bandarmology_status": "Big Accumulation / Normal Accumulation / Neutral / Normal Distribution / Big Distribution",
  "foreign_flow_status": "Net Buy / Net Sell / Neutral / N/A",
  "summary_narrative": "Penjelasan 2-3 kalimat: broker mana yang dominan (sertakan indikasi Bandar vs Retail jika terlihat), konsentrasi top 1 vs top 3, dan implikasi harga.",
  "top_buyers": [
    {
      "broker": "YP",
      "volume_lot": 15000,
      "value_idr": 1500000000,
      "avg_price": 1250,
      "freq": 250,
      "avg_lot_per_freq": 60
    }
  ],
  "top_sellers": [
    {
      "broker": "AK",
      "volume_lot": 20000,
      "value_idr": 2000000000,
      "avg_price": 1260,
      "freq": 85,
      "avg_lot_per_freq": 235
    }
  ]
}

ATURAN EKSTRAKSI:
1. Kolom "freq" adalah frekuensi/jumlah transaksi broker (biasanya ada di kolom "Freq" atau "F").
   Kalau tidak terlihat di screenshot, tulis null.
2. Kolom "avg_lot_per_freq" = volume_lot / freq (rata-rata lot per transaksi).
   Kalau freq null, tulis null.
3. Konversikan angka Milyar (B) / Juta (M/K) ke nilai penuh (contoh: 1.5B = 1500000000).
4. Ambil hingga 5-10 broker pembeli & penjual yang terlihat di screenshot.
5. Kembalikan HANYA JSON yang valid, tanpa teks lain.
"""

        # ===== CALL GEMINI DENGAN RETRY (Exponential Backoff) =====
        try:
            response = _call_gemini_vision_api(model, prompt, compressed_image)
        except Exception as e:
            return None, f"Error Gemini Vision (after 3 retries): {str(e)}"
        
        raw_text = response.text.strip()

        # Pembersihan JSON
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        # Regex fallback jika JSON mengandung karakter ilegal
        try:
            parsed = json.loads(raw_text, strict=False)
            return parsed, None
        except Exception:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0), strict=False)
                return parsed, None
            return {"summary_narrative": raw_text, "bandarmology_status": "Raw AI Response"}, None

    except Exception as e:
        return None, f"Error Gemini Vision: {str(e)}"


# ==========================================
# FUNGSI BANDARMOLOGY & BROKSUM (OCR METODE LOCAL)
# ==========================================
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False


def analisis_broksum_ocr(image):
    """
    Menganalisis screenshot Broker Summary (Broksum) / Trade Flow menggunakan OCR (EasyOCR / PyTesseract).
    Metode offline ini hemat kuota dan tidak menggunakan API Key Gemini.
    """
    if not PIL_AVAILABLE:
        return None, "Library Pillow (PIL) belum terpasang."

    extracted_text = ""
    engine_used = ""

    # Strategy 1: Try EasyOCR if available
    if EASYOCR_AVAILABLE:
        try:
            reader = easyocr.Reader(['id', 'en'], gpu=False)
            img_np = np.array(image.convert('RGB'))
            results = reader.readtext(img_np, detail=0)
            extracted_text = "\n".join(results)
            engine_used = "EasyOCR"
        except Exception:
            extracted_text = ""

    # Strategy 2: Try PyTesseract if available
    if not extracted_text and PYTESSERACT_AVAILABLE:
        try:
            extracted_text = pytesseract.image_to_string(image)
            engine_used = "PyTesseract"
        except Exception:
            extracted_text = ""

    if not extracted_text:
        return None, "Library OCR (EasyOCR / PyTesseract) belum terpasang di server. Anda dapat mengisi data manual di bawah."

    # Parse extracted text to identify broker codes and numerical values
    known_brokers = {
        "YP", "BK", "ZP", "AK", "KZ", "NI", "GR", "RX", "PD", "CC", "CP", "DX", "AZ", "DR",
        "LG", "IF", "OD", "XC", "EP", "YU", "XL", "AI", "DB", "IU", "TP", "HD", "YJ", "CS",
        "KK", "MG", "SQ", "XA", "LS", "BD", "AT", "AN", "OD", "YU", "GA", "RG", "CD"
    }

    buyers = []
    sellers = []
    lines = extracted_text.split('\n')
    is_seller = False

    for line in lines:
        u_line = line.upper()
        if "SELL" in u_line or "SELLER" in u_line or "NET SELL" in u_line:
            is_seller = True
        
        found_brks = re.findall(r'\b([A-Z]{2})\b', u_line)
        numbers = re.findall(r'\b(\d+[\d\.,]*)\b', line)

        for brk in found_brks:
            if brk in known_brokers or len(found_brks) == 1:
                val = 0
                if numbers:
                    try:
                        val = int(numbers[0].replace('.', '').replace(',', ''))
                    except Exception:
                        val = 0
                item = {"broker": brk, "volume_lot": val, "value_idr": val * 100, "avg_price": 0.0}
                if is_seller:
                    sellers.append(item)
                else:
                    buyers.append(item)

    tot_b = sum(b["volume_lot"] for b in buyers)
    tot_s = sum(s["volume_lot"] for s in sellers)

    if tot_b > tot_s * 1.5:
        status = "Big Accumulation"
    elif tot_b > tot_s * 1.1:
        status = "Normal Accumulation"
    elif tot_s > tot_b * 1.5:
        status = "Big Distribution"
    elif tot_s > tot_b * 1.1:
        status = "Normal Distribution"
    else:
        status = "Neutral"

    narrative = f"Hasil OCR ({engine_used}): Terbaca {len(buyers)} Buyer, {len(sellers)} Seller. Status: {status}."

    res_json = {
        "bandarmology_status": status,
        "foreign_flow_status": "Neutral",
        "summary_narrative": narrative,
        "top_buyers": buyers[:5],
        "top_sellers": sellers[:5]
    }

    return res_json, None


def render_broker_summary_and_aggregate_ui(buyers_list, sellers_list):
    """Render Top Buyers/Sellers & Agregat Bandar vs Retail section."""
    if not buyers_list and not sellers_list:
        return

    # Pastikan setiap broker terisi kategori & icon
    for b in buyers_list:
        if isinstance(b, dict) and (not b.get('kategori') or not b.get('kategori_icon')):
            b['kategori'], b['kategori_icon'] = klasifikasi_broker(b.get('broker'), b.get('volume_lot'), b.get('freq'))
    for s in sellers_list:
        if isinstance(s, dict) and (not s.get('kategori') or not s.get('kategori_icon')):
            s['kategori'], s['kategori_icon'] = klasifikasi_broker(s.get('broker'), s.get('volume_lot'), s.get('freq'))

    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown("**🟢 Top Buyers:**")
        for b in buyers_list:
            if not isinstance(b, dict): continue
            kode = b.get('broker', '')
            vol  = b.get('volume_lot', 0) or 0
            frq  = b.get('freq')
            icon = b.get('kategori_icon', '')
            kat  = b.get('kategori', '')
            frq_str = f" · Freq {frq}" if frq else ""
            st.caption(
                f"- {icon} **{kode}**: "
                f"{vol:,.0f} lot{frq_str} — *{kat}*"
            )
    with col_s:
        st.markdown("**🔴 Top Sellers:**")
        for s in sellers_list:
            if not isinstance(s, dict): continue
            kode = s.get('broker', '')
            vol  = s.get('volume_lot', 0) or 0
            frq  = s.get('freq')
            icon = s.get('kategori_icon', '')
            kat  = s.get('kategori', '')
            frq_str = f" · Freq {frq}" if frq else ""
            st.caption(
                f"- {icon} **{kode}**: "
                f"{vol:,.0f} lot{frq_str} — *{kat}*"
            )

    # ═══════════════════════════════════════════════════════
    # AGREGAT — Bandar vs Retail (dari Top broker)
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**📊 Agregat Bandar vs Retail**")
    st.caption(
        "⚠️ Angka di bawah **hanya dari broker yang terlihat di screenshot** "
        "(biasanya Top 5-10) — bukan total market. Gunakan untuk melihat "
        "*proporsi* bandar vs retail, bukan volume absolut."
    )

    # ── Buy Side ──
    bandar_buy = sum((b.get('volume_lot') or 0) for b in buyers_list if isinstance(b, dict) and b.get('kategori') == 'Bandar')
    retail_buy = sum((b.get('volume_lot') or 0) for b in buyers_list if isinstance(b, dict) and b.get('kategori') == 'Retail')
    mixed_buy  = sum((b.get('volume_lot') or 0) for b in buyers_list if isinstance(b, dict) and b.get('kategori') == 'Mixed')

    # ── Sell Side ──
    bandar_sell = sum((s.get('volume_lot') or 0) for s in sellers_list if isinstance(s, dict) and s.get('kategori') == 'Bandar')
    retail_sell = sum((s.get('volume_lot') or 0) for s in sellers_list if isinstance(s, dict) and s.get('kategori') == 'Retail')
    mixed_sell  = sum((s.get('volume_lot') or 0) for s in sellers_list if isinstance(s, dict) and s.get('kategori') == 'Mixed')

    col_ab, col_as = st.columns(2)
    with col_ab:
        st.markdown("**🟢 Buy Side**")
        st.metric("🐋 Bandar (Buy)", f"{bandar_buy:,.0f} lot")
        st.metric("🧑 Retail (Buy)", f"{retail_buy:,.0f} lot")
        if mixed_buy > 0:
            st.caption(f"⚖️ Mixed: {mixed_buy:,.0f} lot")

    with col_as:
        st.markdown("**🔴 Sell Side**")
        st.metric("🐋 Bandar (Sell)", f"{bandar_sell:,.0f} lot")
        st.metric("🧑 Retail (Sell)", f"{retail_sell:,.0f} lot")
        if mixed_sell > 0:
            st.caption(f"⚖️ Mixed: {mixed_sell:,.0f} lot")

    # ── Net + Insight Narasi ──
    net_bandar = bandar_buy - bandar_sell
    net_retail = retail_buy - retail_sell

    if net_bandar > 0 and net_retail < 0:
        insight_icon = "🟢"
        insight = ("<b>Bandar akumulasi, retail distribusi</b> — sinyal bullish. "
                   "Bandar sedang menyerap supply dari retail.")
    elif net_bandar < 0 and net_retail > 0:
        insight_icon = "🔴"
        insight = ("<b>Bandar distribusi, retail akumulasi</b> — hati-hati. "
                   "Bandar sedang melepas barang ke retail (kemungkinan puncak).")
    elif net_bandar > 0 and net_retail > 0:
        insight_icon = "⚖️"
        insight = "<b>Kedua pihak net buy</b> — minat beli kuat, tapi perlu konfirmasi arah lanjut."
    elif net_bandar < 0 and net_retail < 0:
        insight_icon = "⚠️"
        insight = "<b>Kedua pihak net sell</b> — tekanan jual kuat, waspadai koreksi lanjut."
    else:
        insight_icon = "⚖️"
        insight = "<b>Net flow seimbang</b> — pasar belum ada dominasi jelas."

    if net_bandar > 0:
        net_color = "#10b981"
    elif net_bandar < 0:
        net_color = "#ef4444"
    else:
        net_color = "#94a3b8"

    st.markdown(f"""<div style="background:{net_color}12; border-left:4px solid {net_color}; border-radius:8px; padding:12px 16px; margin-top:12px; color:#cbd5e1; font-size:13px; line-height:1.6;">
<div style="font-weight:600; color:{net_color}; font-size:14px; margin-bottom:6px;">{insight_icon} Net Bandar: {net_bandar:+,.0f} lot · Net Retail: {net_retail:+,.0f} lot</div>
<div>{insight}</div>
</div>""", unsafe_allow_html=True)


def render_broksum_scan_ui(api_key="", key_prefix="broksum"):
    st.markdown("### 📸 Scan Broker Summary (Broksum)")
    st.caption("Upload screenshot Broksum (Stockbit, IPOT, HOTS, dll) untuk dianalisis & tersimpan ke database.")

    # ========== INPUT TICKER ==========
    st.markdown("**Ticker Saham** (wajib untuk menyimpan ke database)")
    ticker_input = st.text_input(
        "Masukkan kode saham (contoh: BBCA, GOTO, ASII)",
        key=f"{key_prefix}_ticker_input",
        placeholder="BBCA"
    ).strip().upper()

    uploaded_file = st.file_uploader(
        "Pilih Foto / Screenshot Broksum",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"{key_prefix}_file_uploader"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview Broksum", use_container_width=True)

            result_key  = f"{key_prefix}_result"
            error_key   = f"{key_prefix}_error"
            source_key  = f"{key_prefix}_source"

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_gemini = st.button("🤖 Scan AI Gemini (Akurat)", key=f"{key_prefix}_btn_gemini", use_container_width=True)
            with col_btn2:
                btn_ocr = st.button("⚡ Scan Offline (OCR)", key=f"{key_prefix}_btn_ocr", use_container_width=True)

            if btn_gemini:
                if not api_key:
                    st.error("⚠️ Gemini API Key belum diisi di sidebar.")
                else:
                    with st.spinner("🧠 Gemini Vision sedang membaca tabel Broksum... [Retry enabled]"):
                        res_json, err = analisis_broksum_gemini_vision(image, api_key)
                    if err:
                        st.session_state[error_key] = err
                    else:
                        res_json = enrich_broker_kategori(res_json)
                        st.session_state[result_key] = res_json
                        st.session_state[error_key]  = None
                        st.session_state[source_key] = "gemini"

            elif btn_ocr:
                with st.spinner("🔍 Membaca screenshot Broksum via OCR..."):
                    res_json, err = analisis_broksum_ocr(image)
                if err:
                    st.session_state[error_key] = err
                else:
                    res_json = enrich_broker_kategori(res_json)
                    st.session_state[result_key] = res_json
                    st.session_state[error_key]  = None
                    st.session_state[source_key] = "ocr"

            res_json = st.session_state.get(result_key)
            err      = st.session_state.get(error_key)
            source   = st.session_state.get(source_key, "")

            # ═══ ENRICH: Klasifikasi Bandar vs Retail ═══
            if res_json:
                for side_key in ["top_buyers", "top_sellers"]:
                    for item in res_json.get(side_key, []):
                        kode = item.get("broker", "")
                        vol  = item.get("volume_lot", 0) or 0
                        frq  = item.get("freq")
                        kategori, icon = klasifikasi_broker(kode, vol, frq)
                        item["kategori"] = kategori
                        item["kategori_icon"] = icon

            if res_json:
                st.success(f"✅ **Status:** {res_json.get('bandarmology_status', 'N/A')}")
                if res_json.get("summary_narrative"):
                    st.info(f"📝 {res_json.get('summary_narrative')}")

                # ========== SAVE TO DATABASE ==========
                st.divider()
                col_save_1, col_save_2 = st.columns([2, 1])
                with col_save_1:
                    if not ticker_input:
                        st.warning("⚠️ Masukkan ticker terlebih dahulu untuk menyimpan ke database.")
                    else:
                        st.info(f"💾 Siap simpan ke database untuk **{ticker_input}**")
                with col_save_2:
                    btn_save = st.button("💾 Simpan ", key=f"{key_prefix}_btn_save", use_container_width=True)
                    
                if btn_save:
                    if not ticker_input:
                        st.error("❌ Ticker tidak boleh kosong!")
                    else:
                        with st.spinner(f"💾 Menyimpan data {ticker_input} ke Google Sheets..."):
                            success = save_broksum_data(ticker_input, res_json, source=source)

                        if success:
                            # ▼▼▼ TAMBAH INI ▼▼▼
                            with st.spinner(f"📊 Mengambil foreign flow IDX untuk {ticker_input}..."):
                                ff_ok = save_foreign_flow_snapshot(ticker_input)
                            # ▲▲▲ ▲▲▲

                            if ff_ok:
                                st.success(f"✅ Broksum + Foreign Flow IDX **{ticker_input}** tersimpan!")
                            else:
                                st.warning(
                                    f"✅ Broksum **{ticker_input}** tersimpan! "
                                    f"⚠️ Foreign flow IDX gagal diambil sekarang (kemungkinan rate limit). "
                                    f"Data akan di-nambal otomatis oleh cron jam 19:30 WIB."
                                )

                            st.session_state[f"{key_prefix}_result"] = None

            elif err:
                st.warning(f"⚠️ {err}")

        except Exception as e_img:
            st.error(f"Gagal memuat gambar: {e_img}")

    with st.expander("📝 Input Manual Broksum", expanded=False):
        st.caption("Isi data Top Buyers & Top Sellers secara manual.")
        n_broker = st.number_input("Jumlah broker per sisi", min_value=1, max_value=10, value=5, key=f"{key_prefix}_n_broker")

        st.markdown("**🟢 Top Buyers**")
        buyers_manual = []
        for idx in range(int(n_broker)):
            ca, cb = st.columns([2, 3])
            brk = ca.text_input("", key=f"{key_prefix}_b_brk_{idx}", placeholder=f"Broker {idx+1}", label_visibility="collapsed")
            vol = cb.number_input("", key=f"{key_prefix}_b_vol_{idx}", min_value=0, value=0, label_visibility="collapsed")
            if brk.strip():
                buyers_manual.append({"broker": brk.strip(), "volume_lot": int(vol)})

        st.markdown("**🔴 Top Sellers**")
        sellers_manual = []
        for idx in range(int(n_broker)):
            ca, cb = st.columns([2, 3])
            brk = ca.text_input("", key=f"{key_prefix}_s_brk_{idx}", placeholder=f"Broker {idx+1}", label_visibility="collapsed")
            vol = cb.number_input("", key=f"{key_prefix}_s_vol_{idx}", min_value=0, value=0, label_visibility="collapsed")
            if brk.strip():
                sellers_manual.append({"broker": brk.strip(), "volume_lot": int(vol)})

        bandarmology_status = st.selectbox(
            "Status Bandarmologi",
            ["Akumulasi", "Distribusi", "Sideways/Tidak Jelas", "Mixed"],
            key=f"{key_prefix}_manual_status"
        )

        if st.button("✅ Simpan Data Manual", key=f"{key_prefix}_manual_submit", use_container_width=True):
            if buyers_manual or sellers_manual:
                st.session_state[f"{key_prefix}_result"] = {
                    "top_buyers": buyers_manual,
                    "top_sellers": sellers_manual,
                    "bandarmology_status": bandarmology_status,
                    "summary_narrative": f"Input manual: Status {bandarmology_status}."
                }
                st.session_state[f"{key_prefix}_error"] = None
                st.success("✅ Data manual berhasil disimpan.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# BANDARMOLOGY – DATA LOADER & CHART BUILDERS (HYBRID TIME-SERIES)
# ═══════════════════════════════════════════════════════════════

# ---------- PRICE DATA HELPERS (yfinance) ----------
@st.cache_data(ttl=120, show_spinner=False)
def _load_intraday_price_data(ticker):
    """
    Ambil data intraday dari yfinance, prioritaskan 1m.
    Fallback: 5m → 15m → 30m → 60m kalau 1m tidak tersedia / terlalu sparse.
    Return: (df, interval_label)
    """
    t = ticker.upper().strip()
    if not t.endswith(".JK"):
        t = f"{t}.JK"

    # Urutan prioritas: 1m dulu (paling detail), lalu turun
    intervals = ["1m", "2m", "5m", "15m", "30m", "60m"]
    # yfinance: 1m hanya boleh period <= 7d
    period_map = {
        "1m": "5d", "2m": "5d", "5m": "5d",
        "15m": "5d", "30m": "5d", "60m": "5d"
    }

    for interval in intervals:
        try:
            df = yf.download(
                t,
                period=period_map[interval],
                interval=interval,
                progress=False,
                prepost=False
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Konversi timezone ke Jakarta
            try:
                if df.index.tz is not None:
                    df = df.tz_convert("Asia/Jakarta")
            except Exception:
                pass

            # Ambil hari terakhir yang punya data
            last_date = df.index[-1].date()
            df_last = df[df.index.date == last_date].copy()

            if df_last.empty:
                continue

            # Minimal 30 bar untuk 1m (sesi IDX 1 hari ~ 330 menit),
            # minimal 20 bar untuk interval lain
            min_bars = 30 if interval == "1m" else 20
            if len(df_last) < min_bars:
                # Coba hari sebelumnya untuk 1m (jaga-jaga kalau sesi baru mulai)
                if interval == "1m":
                    unique_dates = sorted(set(df.index.date), reverse=True)
                    for d in unique_dates:
                        df_candidate = df[df.index.date == d].copy()
                        if len(df_candidate) >= 50:
                            return df_candidate, interval
                continue

            return df_last, interval

        except Exception:
            continue

    return None, None


@st.cache_data(ttl=300, show_spinner=False)
def _load_daily_price_data(ticker, period="1mo"):
    """Ambil data harian dari yfinance."""
    t = ticker.upper().strip()
    if not t.endswith(".JK"):
        t = f"{t}.JK"
    try:
        df = yf.download(t, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


# ---------- DATA LOADER (broksum snapshot) ----------
def load_bandarmology_data(ticker):
    """Load & normalisasi broksum history untuk 1 ticker."""
    try:
        history = load_broksum_history(ticker)
    except Exception:
        return None
    if not history:
        return None

    history_sorted = sorted(history, key=lambda r: str(r.get('upload_date', '')), reverse=True)
    latest = history_sorted[0]

    def _norm(lst):
        out = []
        for it in _parse_broker_list(lst):
            if not isinstance(it, dict):
                continue
            code = str(it.get('broker', '')).strip().upper()
            if not code:
                continue
            vol = safe_float(it.get('volume_lot', 0))
            frq = it.get('freq')
            kat = it.get('kategori')
            icon = it.get('kategori_icon')
            if not kat or not icon:
                kat, icon = klasifikasi_broker(code, vol, frq)
            out.append({
                'broker': code,
                'volume_lot': vol,
                'value_idr': safe_float(it.get('value_idr', 0)),
                'avg_price': safe_float(it.get('avg_price', 0)),
                'freq': frq,
                'kategori': kat,
                'kategori_icon': icon,
                'category': BROKER_TYPES.get(code, 'Domestic')
            })
        return out

    return {
        'ticker': ticker.upper().replace('.JK', ''),
        'history': history_sorted,
        'latest': latest,
        'buyers': _norm(latest.get('top_buyers', '[]')),
        'sellers': _norm(latest.get('top_sellers', '[]')),
        'upload_date': latest.get('upload_date', 'N/A'),
        'bandarmology_status': latest.get('bandarmology_status', 'N/A'),
        'summary_narrative': latest.get('summary_narrative', ''),
    }

# ---------- CHART 1: BROKER FLOW (TIME-SERIES) ----------
def build_broker_flow_chart(data):
    """
    Broker Flow time-series: harga intraday + garis net flow kumulatif per broker.
    Flow didistribusi dari snapshot broksum mengikuti pola volume × arah bar.
    """
    if not data:
        return None, None

    df, interval = _load_intraday_price_data(data['ticker'])
    if df is None or df.empty:
        return None, None

    df = df.copy()
    df['direction'] = np.where(df['Close'] >= df['Open'], 1.0, -1.0)
    df['delta_vol'] = df['Volume'].astype(float) * df['direction']

    raw_cum = df['delta_vol'].cumsum().values
    if len(raw_cum) == 0:
        return None, None
    final_raw = raw_cum[-1] if abs(raw_cum[-1]) > 0 else 1.0

    buyers = sorted(data['buyers'], key=lambda x: x['volume_lot'], reverse=True)[:4]
    sellers = sorted(data['sellers'], key=lambda x: x['volume_lot'], reverse=True)[:4]

    df['time'] = df.index.strftime("%H:%M")

    fig = go.Figure()

    # Harga (secondary axis)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Close'], mode="lines", name="Price",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>Price</b>: %{y:,.0f}<extra></extra>"
    ))

    buyer_colors = ["#10b981", "#06b6d4", "#3b82f6", "#a855f7"]
    seller_colors = ["#ef4444", "#f97316", "#eab308", "#ec4899"]

    # Buyer flows (kumulatif ke atas)
    for idx, b in enumerate(buyers):
        target = b['volume_lot']
        flow_cum = raw_cum / abs(final_raw) * target
        label = get_broker_label(b['broker'])
        custom_text = [
            f"{v/1e9:.2f}B Lot" if abs(v) >= 1e9 else
            f"{v/1e6:.2f}M Lot" if abs(v) >= 1e6 else
            f"{v/1e3:.1f}K Lot" if abs(v) >= 1e3 else
            f"{v:,.0f} Lot" for v in flow_cum
        ]
        fig.add_trace(go.Scatter(
            x=df['time'], y=flow_cum, mode="lines", name=f"Accum {label}",
            customdata=custom_text,
            line=dict(color=buyer_colors[idx % len(buyer_colors)], width=2),
            hovertemplate=f"<b>{label}</b>: %{{customdata}}<extra></extra>"
        ))

    # Seller flows (kumulatif ke bawah)
    for idx, s in enumerate(sellers):
        target = s['volume_lot']
        flow_cum = -raw_cum / abs(final_raw) * target
        label = get_broker_label(s['broker'])
        custom_text = [
            f"{v/1e9:.2f}B Lot" if abs(v) >= 1e9 else
            f"{v/1e6:.2f}M Lot" if abs(v) >= 1e6 else
            f"{v/1e3:.1f}K Lot" if abs(v) >= 1e3 else
            f"{v:,.0f} Lot" for v in flow_cum
        ]
        fig.add_trace(go.Scatter(
            x=df['time'], y=flow_cum, mode="lines", name=f"Dist {label}",
            customdata=custom_text,
            line=dict(color=seller_colors[idx % len(seller_colors)], width=2),
            hovertemplate=f"<b>{label}</b>: %{{customdata}}<extra></extra>"
        ))

    tick_vals = df['time'].tolist()[::max(1, len(df) // 12)]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        height=420, margin=dict(l=10, r=10, t=45, b=10),
        dragmode=False, hovermode="x unified",
        hoverdistance=100, spikedistance=100,
        title=dict(
            text=f"Broker Flow ({interval}) – {data['ticker']} • snapshot {data['upload_date'][:10]}",
            font=dict(size=13, color='#e0e0e0'), x=0.01, xanchor='left'
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=True, gridcolor="#262626", type="category",
            tickmode="array", tickvals=tick_vals,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="#64748b", spikedash="dot"
        ),
        yaxis=dict(
            title="Net Flow (Lot)", showgrid=True, gridcolor="#262626",
            zeroline=True, zerolinecolor="#525252"
        ),
        yaxis2=dict(
            title="Harga", showgrid=False, overlaying="y", side="right"
        )
    )
    return fig, df

def render_trade_flow_spectrum_bar(stats):
    """Render Trade Flow Spectrum Bar (Net Dist <---> Net Acc) dengan marker posisi."""
    if not stats:
        return
    
    marker_pos = stats.get('marker_pos', 50.0)
    status_lbl = stats.get('status_lbl', 'Netral')
    status_clr = stats.get('status_clr', '#94a3b8')
    ratio_pct  = stats.get('net_ratio_pct', 0.0)

    st.markdown(f"""<div style="background:#131722; border:1px solid #262626; border-radius:10px; padding:12px 16px; margin:10px 0 12px 0; font-family:-apple-system, sans-serif;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="color:#94a3b8; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;"></span>
<span style="color:{status_clr}; font-size:12px; font-weight:700;">{status_lbl} ({ratio_pct:+.1f}%)</span>
</div>
<div style="position:relative; height:18px; border-radius:9px; background:linear-gradient(90deg, #ef4444 0%, #dc2626 25%, #451a03 50%, #15803d 75%, #10b981 100%); padding:2px; box-shadow:inset 0 1px 3px rgba(0,0,0,0.6);">
<div style="position:absolute; left:20%; top:0; bottom:0; width:1px; background:rgba(0,0,0,0.4);"></div>
<div style="position:absolute; left:40%; top:0; bottom:0; width:1px; background:rgba(0,0,0,0.4);"></div>
<div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:rgba(255,255,255,0.3);"></div>
<div style="position:absolute; left:60%; top:0; bottom:0; width:1px; background:rgba(0,0,0,0.4);"></div>
<div style="position:absolute; left:80%; top:0; bottom:0; width:1px; background:rgba(0,0,0,0.4);"></div>
<div style="position:absolute; left:{marker_pos:.1f}%; top:-3px; bottom:-3px; width:4px; margin-left:-2px; background:#a855f7; border-radius:2px; box-shadow:0 0 10px #c084fc, 0 0 4px #a855f7; z-index:10;"></div>
</div>
<div style="display:flex; justify-content:space-between; color:#64748b; font-size:10px; margin-top:6px; font-weight:600;">
<span style="color:#f87171;">Net Dist</span>
<span style="color:#64748b;">Netral (0%)</span>
<span style="color:#34d399;">Net Acc</span>
</div>
</div>""", unsafe_allow_html=True)

# ---------- CHART 2: TRADE FLOW (TIME-SERIES) ----------
def build_trade_flow_chart(data):
    """
    Trade Flow refined:
    - Pakai typical price (H+L+C)/3, bukan Close saja
    - Direction dari posisi Close di dalam range bar (bukan binari)
    - Net value = Volume × typical × close_position_signal
    """
    if not data:
        return None, None, None

    df, interval = _load_intraday_price_data(data['ticker'])
    if df is None or df.empty:
        return None, None, None

    df = df.copy()

    # Typical price
    df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3.0

    # Close position dalam range bar (0 = di low, 1 = di high)
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    df['close_pos'] = ((df['Close'] - df['Low']) / rng).fillna(0.5).clip(0, 1)

    # Signal -1 s/d +1
    df['signal'] = df['close_pos'] * 2 - 1

    # Net value (smooth)
    df['net_value'] = df['Volume'].astype(float) * df['typical'] * df['signal']
    df['net_buy'] = df['net_value'].where(df['net_value'] > 0, 0)
    df['net_sell'] = df['net_value'].where(df['net_value'] < 0, 0)
    df['time'] = df.index.strftime("%H:%M")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['time'], y=df['net_buy'], name="Net Buy",
        marker_color="#10b981",
        hovertemplate="<b>Net Buy</b>: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df['time'], y=df['net_sell'], name="Net Sell",
        marker_color="#ef4444",
        hovertemplate="<b>Net Sell</b>: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Close'], mode="lines", name="Price",
        line=dict(color="#0284c7", width=2), yaxis="y2",
        hovertemplate="<b>Price</b>: %{y:,.0f}<extra></extra>"
    ))

    tick_vals = df['time'].tolist()[::max(1, len(df) // 12)]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        height=420, margin=dict(l=10, r=10, t=45, b=10), barmode="relative",
        dragmode=False, hovermode="x unified",
        hoverdistance=100, spikedistance=100,
        title=dict(
            text=f"Trade Flow ({interval}) – {data['ticker']}",
            font=dict(size=13, color='#e0e0e0'), x=0.01, xanchor='left'
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=True, gridcolor="#262626", type="category",
            tickmode="array", tickvals=tick_vals,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="#64748b", spikedash="dot"
        ),
        yaxis=dict(
            title="Value (Rp)", showgrid=True, gridcolor="#262626",
            zeroline=True, zerolinecolor="#525252"
        ),
        yaxis2=dict(
            title="Harga", showgrid=False, overlaying="y", side="right"
        )
    )

    tot_buy = float(df['net_buy'].sum())
    tot_sell = abs(float(df['net_sell'].sum()))
    tot_flow = tot_buy + tot_sell
    net_val = tot_buy - tot_sell

    net_ratio = (net_val / tot_flow) if tot_flow > 0 else 0.0
    marker_pos = max(2.0, min(98.0, (net_ratio + 1.0) / 2.0 * 100.0))

    if net_ratio > 0.25:
        status_lbl, status_clr = "Big Accumulation 🚀", "#10b981"
    elif net_ratio > 0.05:
        status_lbl, status_clr = "Accumulation 🟢", "#34d399"
    elif net_ratio >= -0.05:
        status_lbl, status_clr = "Netral ⚖️", "#94a3b8"
    elif net_ratio >= -0.25:
        status_lbl, status_clr = "Distribution 🔴", "#f87171"
    else:
        status_lbl, status_clr = "Big Distribution 🚨", "#ef4444"

    stats = {
        'tot_buy': tot_buy,
        'tot_sell': tot_sell,
        'net_val': net_val,
        'net_ratio': net_ratio,
        'net_ratio_pct': net_ratio * 100.0,
        'marker_pos': marker_pos,
        'status_lbl': status_lbl,
        'status_clr': status_clr
    }

    return fig, df, stats

@st.cache_data(ttl=1800, show_spinner=False)   # cache 30 menit
def _fetch_idx_foreign_flow(ticker, days=30):
    """
    Scrape Foreign Flow harian dari IDX Trading Summary.
    Return: DataFrame [date, foreign_buy, foreign_sell, net_foreign] atau None.
    """
    ticker_clean = str(ticker).upper().replace(".JK", "").strip()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.idx.co.id/",
        "X-Requested-With": "XMLHttpRequest",
    }

    rows = []
    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
    max_iter = days + 20   # buffer weekend + libur bursa

    for i in range(max_iter):
        if len(rows) >= days:
            break
        d = today - timedelta(days=i)
        if d.weekday() >= 5:   # skip Sabtu/Minggu
            continue

        date_str = d.strftime("%Y-%m-%d")
        date_alt = d.strftime("%Y%m%d")

        # Coba 2 format tanggal endpoint (beda versi IDX pakai beda format)
        endpoints = [
            f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
            f"?length=9999&start=0&date={date_str}",
            f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
            f"?length=9999&start=0&date={date_alt}",
        ]

        items = None
        for url in endpoints:
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code != 200:
                    continue
                try:
                    payload = r.json()
                except Exception:
                    continue
                # Struktur: {"data": [...]} atau langsung list
                if isinstance(payload, dict):
                    items = payload.get("data") or payload.get("Data") or []
                elif isinstance(payload, list):
                    items = payload
                if items:
                    break
            except Exception:
                continue

        if not items:
            continue

        # Cari baris yang match ticker
        for it in items:
            if not isinstance(it, dict):
                continue
            code = str(
                it.get("StockCode") or it.get("KodeSaham") or ""
            ).upper().strip()
            if code != ticker_clean:
                continue

            def _f(*keys):
                """Ambil nilai float dari beberapa kandidat field."""
                for k in keys:
                    v = it.get(k)
                    if v not in (None, "", "N/A", "-"):
                        try:
                            return float(str(v).replace(",", ""))
                        except Exception:
                            continue
                return 0.0

            fb = _f(
                "ForeignBuy", "ForeignBuyValue", "Foreign_Buy",
                "ForeignBuyIDR", "ForeignBuyRp", "ForeignBuyValueIDR"
            )
            fs = _f(
                "ForeignSell", "ForeignSellValue", "Foreign_Sell",
                "ForeignSellIDR", "ForeignSellRp", "ForeignSellValueIDR"
            )

            # Kalau nilainya kecil banget (< 1e8), kemungkinan dalam lot → konversi ke Rupiah
            close_px = _f("Close", "Previous", "ClosePrice", "Price")
            if 0 < fb < 1e8 and close_px > 0:
                fb *= 100 * close_px
            if 0 < fs < 1e8 and close_px > 0:
                fs *= 100 * close_px

            rows.append({
                "date": d,
                "foreign_buy": abs(fb),
                "foreign_sell": abs(fs),
                "net_foreign": fb - fs,
            })
            break

    if not rows:
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
    df = df[(df["foreign_buy"] > 0) | (df["foreign_sell"] > 0)]
    return df if not df.empty else None

# ---------- CHART 3: FOREIGN FLOW (DAILY TIME-SERIES) ----------
def build_foreign_flow_chart(data):
    """
    Foreign Flow — redesign:
    - Baca dari sheet foreign_flow_history (dari cron IDX)
    - Kalau kosong, coba fetch IDX 1 hari langsung (fallback)
    - Return (fig, df, stats) atau (None, None, None)
    """
    if not data:
        return None, None, None

    ticker = str(data.get('ticker', '')).upper().replace('.JK', '').strip()
    if not ticker:
        return None, None, None

    # ── 1. Baca history dari sheet ──
    df = load_foreign_flow_history(ticker, days=30)

    # ── 2. Kalau kosong, coba fetch IDX 1 hari (defensive: wrap try-except) ──
    if df is None or df.empty:
        items = None
        try:
            items = _fetch_idx_all_stock_summary()
        except Exception:
            # IDX rate limit / server down → skip fallback, anggap tidak ada data
            items = None

        if items:
            for it in items:
                if not isinstance(it, dict):
                    continue
                if str(it.get("StockCode", "")).upper() != ticker:
                    continue

                def _f(k):
                    v = it.get(k)
                    if v in (None, "", "N/A", "-"):
                        return 0.0
                    try:
                        return float(str(v).replace(",", ""))
                    except Exception:
                        return 0.0

                fb = _f("ForeignBuy")
                fs = _f("ForeignSell")
                close = _f("Close")
                fb_rp = fb * close if close > 0 else 0.0
                fs_rp = fs * close if close > 0 else 0.0
                date_str = str(it.get("Date") or "")[:10]

                df = pd.DataFrame([{
                    "date": date_str,
                    "close": close,
                    "foreign_buy": fb_rp,
                    "foreign_sell": fs_rp,
                    "net_foreign": fb_rp - fs_rp,
                }])
                break

    if df is None or df.empty:
        return None, None, None

    # ── 3. Stats untuk card ──
    latest = df.iloc[-1]
    stats = {
        'fb': float(latest['foreign_buy']),
        'fs': float(latest['foreign_sell']),
        'net': float(latest['net_foreign']),
        'date': str(latest['date']),
        'days': len(df),
    }

    # ── 4. Kalau < 7 hari → tampilkan info chart ──
    if len(df) < 7:
        fig = go.Figure()
        fig.add_annotation(
            text=f"📊 Butuh minimal 7 hari data<br>"
                 f"<span style='font-size:11px;color:#94a3b8;'>"
                 f"Saat ini: {len(df)} hari — cron IDX akan accumulate otomatis"
                 f"</span>",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='#cbd5e1'),
            align='center',
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(
                text=f"Net Foreign Flow – {ticker}",
                font=dict(size=13, color='#e0e0e0'), x=0.01, xanchor='left'
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig, df, stats

    # ── 5. Build chart net foreign + price line ──
    df_chart = df.copy()
    df_chart['date_str'] = pd.to_datetime(df_chart['date']).dt.strftime("%d %b")

    bar_colors = ['#10b981' if v >= 0 else '#ef4444' for v in df_chart['net_foreign']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_chart['date_str'], y=df_chart['net_foreign'], name="Net F Buy",
        marker_color=bar_colors,
        customdata=[['Net F Buy' if v >= 0 else 'Net F Sell'] for v in df_chart['net_foreign']],
        hovertemplate="<b>%{customdata[0]}</b>: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_chart['date_str'], y=df_chart['close'], mode="lines", name="Price",
        line=dict(color="#0284c7", width=2), yaxis="y2",
        hovertemplate="<b>Price</b>: %{y:,.0f}<extra></extra>"
    ))

    date_range = f"{df_chart['date_str'].iloc[0]} – {df_chart['date_str'].iloc[-1]}"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        height=380, margin=dict(l=10, r=10, t=45, b=10), barmode="relative",
        dragmode=False, hovermode="x unified",
        hoverdistance=100, spikedistance=100,
        title=dict(
            text=f"Net Foreign Flow – {ticker} <span style='font-size:11px;color:#64748b;'>({date_range})</span>",
            font=dict(size=13, color='#e0e0e0'), x=0.01, xanchor='left'
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="center", x=0.5, font=dict(size=11, color="#94a3b8")
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#262626", type="category",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="#64748b", spikedash="dot"
        ),
        yaxis=dict(
            title="Net Foreign (Rp)", showgrid=True, gridcolor="#262626",
            zeroline=True, zerolinecolor="#525252",
            tickformat=".2s",
        ),
        yaxis2=dict(
            title="Harga", showgrid=False, overlaying="y", side="right"
        )
    )

    return fig, df, stats

def _ipf_allocate(row_sums, col_sums, iterations=20):
    """
    Iterative Proportional Fitting.
    Return: matrix X[i][j] s.t. sum_j X[i][j] ≈ row_sums[i] dan
    sum_i X[i][j] ≈ col_sums[j].
    """
    n = len(row_sums)
    m = len(col_sums)
    if n == 0 or m == 0:
        return []

    total_row = sum(row_sums)
    total_col = sum(col_sums)
    if total_row <= 0 or total_col <= 0:
        return [[0.0] * m for _ in range(n)]

    total = min(total_row, total_col)
    row_target = [r * total / total_row for r in row_sums]
    col_target = [c * total / total_col for c in col_sums]

    # Init semua cell = 1
    X = [[1.0] * m for _ in range(n)]

    for _ in range(iterations):
        # Scale rows
        for i in range(n):
            s = sum(X[i])
            if s > 0:
                f = row_target[i] / s
                for j in range(m):
                    X[i][j] *= f
        # Scale cols
        for j in range(m):
            s = sum(X[i][j] for i in range(n))
            if s > 0:
                f = col_target[j] / s
                for i in range(n):
                    X[i][j] *= f

    return X

# ---------- CHART 4: BROKER DISTRIBUTION (SANKEY) ----------
def build_broker_sankey(data):
    """Broker Distribution Sankey – buyer → seller, dengan payload Value + Volume."""
    if not data:
        return None

    buyers = data.get('buyers', [])[:8]
    sellers = data.get('sellers', [])[:8]
    if not buyers and not sellers:
        return None

    # ═══════════════════════════════════════════════════════
    # HELPER: Harga per lembar & nilai Rupiah
    # ═══════════════════════════════════════════════════════
    def _ppl(item):
        """Harga per lembar (price per lot / 100)."""
        try:
            v = float(item.get('value_idr', 0) or 0)
            vol = float(item.get('volume_lot', 0) or 0)
            if v > 0 and vol > 0:
                return v / (vol * 100)
            ap = float(item.get('avg_price', 0) or 0)
            if ap > 0:
                return ap
        except Exception:
            pass
        return 0.0

    def _val(item, fallback_price=1000.0):
        """Nilai Rupiah total."""
        try:
            v = float(item.get('value_idr', 0) or 0)
            if v > 0:
                return v
            vol = float(item.get('volume_lot', 0) or 0)
            p = _ppl(item)
            if p <= 0:
                p = fallback_price
            return vol * 100 * p
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════
    # HITUNG HARGA GLOBAL (fallback)
    # ═══════════════════════════════════════════════════════
    all_prices = []
    for it in (buyers + sellers):
        p = _ppl(it)
        if p > 0:
            all_prices.append(p)
    global_price = (sum(all_prices) / len(all_prices)) if all_prices else 1000.0

    # ═══════════════════════════════════════════════════════
    # TOTAL BUY / SELL
    # ═══════════════════════════════════════════════════════
    total_buy = sum(float(b.get('volume_lot', 0) or 0) for b in buyers)
    total_sell = sum(float(s.get('volume_lot', 0) or 0) for s in sellers)

    # ═══════════════════════════════════════════════════════
    # BUILD NODES
    # ═══════════════════════════════════════════════════════
    buyer_nodes = [f"{b.get('broker', '??')} (Buy)" for b in buyers]
    seller_nodes = [f"{s.get('broker', '??')} (Sell)" for s in sellers]
    node_idx = {n: i for i, n in enumerate(buyer_nodes + seller_nodes)}

    # ═══════════════════════════════════════════════════════
    # BUILD LINKS
    # ═══════════════════════════════════════════════════════
    sources = []
    targets = []
    vol_values = []
    val_values = []
    link_colors = []

    category_rgb = {
        "Domestic": "168, 85, 247",
        "BUMN": "16, 185, 129",
        "Foreign": "239, 68, 68"
    }

    # ── Hitung alokasi optimal pakai IPF ──
    row_sums = [float(b.get('volume_lot', 0) or 0) for b in buyers]
    col_sums = [float(s.get('volume_lot', 0) or 0) for s in sellers]
    allocation = _ipf_allocate(row_sums, col_sums, iterations=20)

    # ── Precompute value per lot untuk tiap buyer & seller ──
    buyer_vpl = []   # value per lot
    for i, b in enumerate(buyers):
        bv = row_sums[i]
        buyer_vpl.append(_val(b, global_price) / bv if bv > 0 else 0)

    seller_vpl = []
    for j, s in enumerate(sellers):
        sv = col_sums[j]
        seller_vpl.append(_val(s, global_price) / sv if sv > 0 else 0)

    # ── Build links dari alokasi IPF ──
    for i, b in enumerate(buyers):
        if row_sums[i] <= 0:
            continue
        for j, s in enumerate(sellers):
            if col_sums[j] <= 0:
                continue

            flow_vol = allocation[i][j] if (i < len(allocation) and j < len(allocation[i])) else 0.0
            if flow_vol <= 0:
                continue

            b_vpl = buyer_vpl[i]
            s_vpl = seller_vpl[j]
            if b_vpl > 0 and s_vpl > 0:
                avg_vpl = (b_vpl + s_vpl) / 2
            elif b_vpl > 0:
                avg_vpl = b_vpl
            elif s_vpl > 0:
                avg_vpl = s_vpl
            else:
                avg_vpl = global_price * 100

            flow_val = flow_vol * avg_vpl

            sources.append(node_idx[f"{b.get('broker', '??')} (Buy)"])
            targets.append(node_idx[f"{s.get('broker', '??')} (Sell)"])
            vol_values.append(flow_vol)
            val_values.append(flow_val)

            rgb = category_rgb.get(b.get('category', 'Domestic'), "148, 163, 184")
            link_colors.append(f"rgba({rgb}, 0.55)")

    if not vol_values:
        return None

    # ═══════════════════════════════════════════════════════
    # FORMATTER
    # ═══════════════════════════════════════════════════════
    def _fmt_vol(v):
        try:
            v = float(v)
        except Exception:
            return "0"
        if v >= 1e6:
            return f"{v/1e6:,.2f}M"
        return f"{v:,.0f}"

    def _fmt_val(v):
        try:
            v = float(v)
        except Exception:
            return "0"
        if v >= 1e12:
            return f"{v/1e12:.2f}T"
        if v >= 1e9:
            return f"{v/1e9:.2f}B"
        if v >= 1e6:
            return f"{v/1e6:,.0f}M"
        if v >= 1e3:
            return f"{v/1e3:,.0f}K"
        return f"{v:,.0f}"

    # ═══════════════════════════════════════════════════════
    # BUILD LABELS & COLORS
    # ═══════════════════════════════════════════════════════
    labels_vol = []
    for b in buyers:
        labels_vol.append(f"{b.get('broker', '??')} ({_fmt_vol(b.get('volume_lot', 0))})")
    for s in sellers:
        labels_vol.append(f"{s.get('broker', '??')} ({_fmt_vol(s.get('volume_lot', 0))})")

    labels_val = []
    for b in buyers:
        labels_val.append(f"{b.get('broker', '??')} ({_fmt_val(_val(b, global_price))})")
    for s in sellers:
        labels_val.append(f"{s.get('broker', '??')} ({_fmt_val(_val(s, global_price))})")

    cat_hex = {
        "Domestic": "#a855f7",
        "BUMN": "#10b981",
        "Foreign": "#ef4444"
    }
    node_colors = []
    for b in buyers:
        node_colors.append(cat_hex.get(b.get('category', 'Domestic'), "#94a3b8"))
    for s in sellers:
        node_colors.append(cat_hex.get(s.get('category', 'Domestic'), "#94a3b8"))

    # ═══════════════════════════════════════════════════════
    # BUILD FIGURE
    # ═══════════════════════════════════════════════════════
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16,
            thickness=12,
            line=dict(color="#121212", width=1),
            label=labels_vol,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=vol_values,
            color=link_colors
        )
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        height=450,
        margin=dict(l=5, r=5, t=40, b=5),
        hovermode=False,
        title=dict(
            text=f"Broker Distribution – {data.get('ticker', '?')}",
            font=dict(size=13, color='#e0e0e0'),
            x=0.01,
            xanchor='left'
        ),
        font=dict(size=11, color="#94a3b8"),
        meta={
            'mode_toggle': {
                'volume': {
                    'link_values': vol_values,
                    'node_labels': labels_vol,
                },
                'value': {
                    'link_values': val_values,
                    'node_labels': labels_val,
                }
            },
            'n_buyers': len(buyers),
            'n_sellers': len(sellers),
        }
    )

    return fig


# ---------- TAB RENDERER ----------
def display_bandarmology_tab(ticker):
    """Render section Bandarmology lengkap (4 chart realtime) di tab."""
    data = load_bandarmology_data(ticker)

    if not data:
        st.info(
            f"📭 Belum ada data Bandarmology untuk **{ticker}**. "
            "Upload screenshot Broksum via sidebar → 📸 Scan Broksum untuk mulai tracking."
        )
        return

    st.markdown(f"### 🕵🏻‍♂️ Bandarmology – {data['ticker']}")
    st.caption(
        f"Snapshot: **{data['upload_date']}** | Status: **{data['bandarmology_status']}** "
        f"| ℹ️ Flow intraday = interpolasi dari snapshot broksum "
    )
    if data.get('summary_narrative'):
        st.info(f"📝 {data['summary_narrative']}")

    # ═══ TOP BROKERS & AGREGAT BANDAR vs RETAIL ═══
    render_broker_summary_and_aggregate_ui(data.get('buyers', []), data.get('sellers', []))
    st.divider()

    # ═══════════════════════════════════════════════
    # CHART 1: BROKER FLOW
    # ═══════════════════════════════════════════════
    st.markdown("#### 1. Broker Flow")
    fig1, df1 = build_broker_flow_chart(data)
    if fig1 is not None and df1 is not None and len(df1) > 0:
        render_plotly_realtime(fig1, height=420)
    else:
        st.caption("(Data harga intraday tidak tersedia dari yfinance)")

    st.divider()

    # ═══════════════════════════════════════════════
    # CHART 2: TRADE FLOW
    # ═══════════════════════════════════════════════
    st.markdown("#### 2. Trade Flow")
    fig2, df2, stats2 = build_trade_flow_chart(data)
    if fig2 is not None and df2 is not None and len(df2) > 0:
        render_plotly_realtime(fig2, height=420)
        render_trade_flow_spectrum_bar(stats2)
    else:
        st.caption("(Data harga intraday tidak tersedia dari yfinance)")

    st.divider()

    # ═══════════════════════════════════════════════
    # CHART 3: FOREIGN FLOW
    # ═══════════════════════════════════════════════
    st.markdown("#### 3. Foreign Flow")
    fig3, df3, stats3 = build_foreign_flow_chart(data)

    # ── CARD ala Stockbit ──
    if stats3:
        fb_str = fmt_money(stats3['fb'])
        fs_str = fmt_money(stats3['fs'])
        net_str = fmt_money(stats3['net'])
        net_color = "#10b981" if stats3['net'] >= 0 else "#ef4444"
        net_sign = "+" if stats3['net'] >= 0 else ""
        tanggal = stats3.get('date', '')

        st.markdown(f"""
        <div style="background:#1a1d24; border-radius:8px; padding:14px 18px; margin:8px 0 4px 0; border:1px solid #262626; display:flex; justify-content:space-around; align-items:center; font-family:-apple-system, sans-serif;">
            <div style="text-align:center;">
                <div style="color:#94a3b8; font-size:11px; margin-bottom:4px;">F Buy</div>
                <div style="color:#10b981; font-size:18px; font-weight:600;">{fb_str}</div>
            </div>
            <div style="color:#334155; font-size:18px;">|</div>
            <div style="text-align:center;">
                <div style="color:#94a3b8; font-size:11px; margin-bottom:4px;">F Sell</div>
                <div style="color:#ef4444; font-size:18px; font-weight:600;">{fs_str}</div>
            </div>
            <div style="color:#334155; font-size:18px;">|</div>
            <div style="text-align:center;">
                <div style="color:#94a3b8; font-size:11px; margin-bottom:4px;">Net F</div>
                <div style="color:{net_color}; font-size:18px; font-weight:600;">{net_sign}{net_str}</div>
            </div>
        </div>
        <div style="text-align:center; color:#64748b; font-size:10px; margin-bottom:12px; font-style:italic;">
            📊 Sumber: IDX resmi (data per {tanggal})
        </div>
        """, unsafe_allow_html=True)

    if fig3 is not None:
        render_plotly_realtime(fig3, height=380)
    else:
        st.caption("(Data foreign flow tidak tersedia — tunggu cron IDX akumulasi data)")

    st.divider()

    # ═══════════════════════════════════════════════
    # CHART 4: SANKEY (TIDAK PAKAI EMBED)
    # ═══════════════════════════════════════════════
    st.markdown("#### 4. Broker Distribution ")
    fig4 = build_broker_sankey(data)
    if fig4:
        render_sankey_interactive(fig4, height=520)
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 12px; margin-top: 4px; color: #94a3b8;">
            <div><span style="color: #a855f7; font-size: 14px;">■</span> Domestic</div>
            <div><span style="color: #10b981; font-size: 14px;">■</span> BUMN</div>
            <div><span style="color: #ef4444; font-size: 14px;">■</span> Foreign</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("(Tidak ada data buyer/seller yang cukup untuk diagram )")

    # ═══════════════════════════════════════════════
    # RIWAYAT UPLOAD
    # ═══════════════════════════════════════════════
    if len(data['history']) > 1:
        with st.expander(f"📜 Riwayat Upload ({len(data['history'])} entri)"):
            for h in data['history'][:10]:
                st.caption(
                    f"• {h.get('upload_date', 'N/A')} — "
                    f"{h.get('bandarmology_status', 'N/A')}"
                )

# ==========================================
# KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro v2", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

if "sheets_initialized" not in st.session_state:
    init_sheets()
    st.session_state.sheets_initialized = True

if 'v12_memory' not in st.session_state:
    st.session_state.v12_memory = load_v12_memory()

if "riwayat" not in st.session_state:
    st.session_state.riwayat = muat_riwayat_dari_sheets()
if "riwayat_actual" not in st.session_state:
    st.session_state.riwayat_actual = muat_riwayat_actual()

st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8892b0; }
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border: 1px solid #374151; }
    .stButton>button:hover { background-color: #374151; border-color: #00ffcc; }
    h1, h2, h3 { color: #f3f4f6; }
    .translated { color: #cbd5e1; font-size: 13px; }
    .source { color: #6b7280; font-size: 11px; }
    .summary-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 20px; margin: 10px 0; border: 1px solid #334155;
    }
    .action-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 5px solid #00ffcc;
    }
    .section-title { color: #00ffcc; font-size: 18px; font-weight: bold; margin-bottom: 12px; }
    .summary-item { color: #cbd5e1; font-size: 15px; margin-bottom: 8px; }
    .fundamental-table { width: 100%; border-collapse: collapse; color: #cbd5e1; }
    .fundamental-table td { padding: 6px 12px; border-bottom: 1px solid #334155; }
    .fundamental-table td:first-child { color: #8892b0; width: 180px; }
    .ai-insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 20px; margin: 15px 0;
        border-left: 5px solid #8b5cf6; color: #cbd5e1; font-size: 15px; line-height: 1.6;
    }
    .ai-insight-card h3 { color: #a78bfa; margin-top: 0; font-size: 20px; }
    .ai-insight-card p { margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def _get_broksum_for_date(ticker, date_str):
    """
    Ambil broksum dari sheet broksum_history untuk ticker + tanggal.
    Match exact dulu, fallback ±2 hari.
    Return: dict {status, upload_date, narrative} atau None.
    """
    try:
        sheet = get_gsheet().worksheet("broksum_history")
        records = sheet.get_all_records()
        ticker_clean = str(ticker).upper().replace(".JK", "").strip()

        def _parse(rec):
            try:
                return {
                    'status': rec.get('bandarmology_status', 'N/A'),
                    'upload_date': rec.get('upload_date', ''),
                    'narrative': rec.get('summary_narrative', ''),
                }
            except Exception:
                return None

        # ── Exact match ──
        for r in records:
            if str(r.get('ticker', '')).upper() != ticker_clean:
                continue
            if str(r.get('upload_date', '')).startswith(date_str):
                parsed = _parse(r)
                if parsed:
                    return parsed

        # ── Fallback: ±2 hari ──
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
            for delta in [1, -1, 2, -2]:
                alt = (target + timedelta(days=delta)).strftime("%Y-%m-%d")
                for r in records:
                    if str(r.get('ticker', '')).upper() != ticker_clean:
                        continue
                    if str(r.get('upload_date', '')).startswith(alt):
                        parsed = _parse(r)
                        if parsed:
                            parsed['matched_offset'] = delta
                            return parsed
        except Exception:
            pass

        return None
    except Exception:
        return None
# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 QuantRisk Pro")
    
    st.caption("⚙️ Sistem akan menganalisis **Swing (harian)** dan **Daytrade (intraday)** secara otomatis.")
    
    st.markdown("Masukkan kode saham IHSG untuk analisis lengkap.")
    ticker_raw = st.text_input("🔍 Kode Saham", value="BBRI", placeholder="Contoh: BBRI, TLKM, BMRI").upper().strip()
    if ticker_raw and not ticker_raw.endswith(".JK"):
        ticker_input = f"{ticker_raw}.JK"
    else:
        ticker_input = ticker_raw

    # ── Load Gemini API Key lebih awal (dibutuhkan untuk Scan Broksum di bawah) ──
    def _get_api_key_early():
        try:
            return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
        return st.session_state.get("gemini_api_key", "")

    if not st.session_state.get("gemini_api_key"):
        st.session_state.gemini_api_key = _get_api_key_early()

    # ── 📸 Scan Broksum (dipindah ke sini — tepat di bawah input ticker) ──
    with st.expander("📸 Scan Broksum (Gemini AI / OCR)", expanded=False):
        render_broksum_scan_ui(
            api_key=st.session_state.get("gemini_api_key", ""),
            key_prefix="sb_broksum"
        )

    # --- Cek Swing Aktif untuk Ticker ---
    ticker_clean = ticker_raw.replace(".JK", "").strip().upper()
    dict_active_swings = dapatkan_dict_swing_aktif()
    aksi_simpan_mode = "simpan_baru"

    if ticker_clean in dict_active_swings:
        active_info = dict_active_swings[ticker_clean]
        st.warning(
            f"⏳ **{ticker_clean} memiliki Swing Aktif!**\n\n"
            f"Entry tanggal: `{active_info['waktu']}` (Hari bursa ke-{active_info['b_days']})\n"
            f"Status outcome belum diisi."
        )
        pilihan_aksi = st.radio(
            "📋 Tindakan Penyimpanan Riwayat:",
            [
                "🛡️ Lewati Simpan (Hanya Lihat Analisis)",
                "🔄 Update Entry Swing Aktif",
                "➕ Simpan Setup Baru (Re-entry / Add Lot)"
            ],
            index=0,
            key="pilihan_aksi_riwayat_active"
        )
        if "Lewati" in pilihan_aksi:
            aksi_simpan_mode = "lewati"
        elif "Update" in pilihan_aksi:
            aksi_simpan_mode = "update"
        else:
            aksi_simpan_mode = "simpan_baru"
    
    st.session_state['aksi_simpan_mode'] = aksi_simpan_mode

    # --- Input Harga Manual ---
    harga_manual = st.text_input("💵 Harga Pasar Saat Ini (opsional)", placeholder="Kosongkan jika pakai harga data")
    if harga_manual:
        try:
            harga_terakhir_manual = float(harga_manual.replace(",",""))
        except:
            st.error("Format harga salah")
            harga_terakhir_manual = None
    else:
        harga_terakhir_manual = None
    # Letakkan sebelum tombol ANALISIS, misal setelah harga_manual
    sudah_beli = st.checkbox("🟢 Saya sudah punya posisi di saham ini", value=False)
    # ---- Tambahan input harga beli ----
    harga_beli_float = None
    if sudah_beli:
        harga_beli_str = st.text_input("💰 Harga Beli Rata‑rata (opsional)", placeholder="Kosongkan jika tidak tahu")
        if harga_beli_str:
            try:
                harga_beli_float = float(harga_beli_str.replace(",", ""))
            except:
                st.error("Format harga beli salah")

    # ---- Pengaturan Fee Broker ----
    with st.expander("⚙️ Fee Broker (Beli & Jual)", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fee_beli_pct = st.number_input("Fee Beli (%)", min_value=0.0, max_value=2.0, value=0.15, step=0.05, key="fee_beli_pct")
        with col_f2:
            fee_jual_pct = st.number_input("Fee Jual (%)", min_value=0.0, max_value=2.0, value=0.25, step=0.05, key="fee_jual_pct")



    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 ANALISIS", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache dibersihkan!")
    #==================== SCANNER SAHAM IDX ====================
    st.markdown("---")
    st.subheader("🔍 Scanner Saham IDX")
    mode_scan = st.selectbox(
        "Pilih Mode Scan:",
        ["Cepat (LQ45)", "Papan Utama", "Komprehensif (Utama + Pengembangan)", "Full IDX", "Auto-Fetch (API BEI)"],
        index=0, key="mode_scan"
    )
    likuiditas_min = st.number_input(
        "Filter Likuiditas Minimum (Rp/hari, rata2 20 hari)",
        min_value=0, value=300_000_000, step=100_000_000, key="likuiditas_min"
    )
    hide_active_swings = st.checkbox(
        "🚫 Sembunyikan emiten dengan Swing Aktif",
        value=False,
        key="hide_active_swings",
        help="Jika dicentang, saham yang posisi swing-nya masih aktif di riwayat tidak akan ditampilkan di hasil scan."
    )
    ai_rerank = st.checkbox("Sertakan Scanner AI Re-Rank (Top 15 kandidat teknikal)", value=False, key="ai_rerank")
    if ai_rerank:
        st.caption("+15-30 detik. Hemat kuota Gemini gratis: HANYA 1 panggilan API dibatch utk semua kandidat + cache harian per-saham.")
    
    scan_btn = st.button("🔍 SCAN SAHAM", use_container_width=True)

    # ---------- HELPER RENDER CARD PER MODE (SIDE-BY-SIDE) ----------
    def render_mode_card(r, mode_title, mode_icon, container, idx_key):
        with container:
            if not r:
                st.markdown(
                    '<div style="color:#64748b; font-size:11px; font-style:italic; '
                    'padding:10px; text-align:center; background:#0f172a; '
                    'border-radius:8px; border:1px dashed #334155;">'
                    'Data tidak tersedia</div>',
                    unsafe_allow_html=True
                )
                return
    
            sinyal = r.get('Sinyal', '?')
            if "STRONG BUY" in sinyal:
                sig_color, sig_icon = "#10b981", "🔥"
            elif "BUY" in sinyal:
                sig_color, sig_icon = "#84cc16", "⚡"
            elif "HOLD" in sinyal:
                sig_color, sig_icon = "#3b82f6", "⏸️"
            else:
                sig_color, sig_icon = "#ef4444", "🚨"
    
            mode_color = "#a855f7" if "Swing" in mode_title else "#06b6d4"
    
            # ── Header mode + signal badge ──
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {sig_color}18 0%, {sig_color}06 100%);
                border-left: 4px solid {sig_color};
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 8px;
            ">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="color:{mode_color}; font-size:10px; font-weight:700; 
                        letter-spacing:1.2px; text-transform:uppercase;">
                        {mode_icon} {mode_title}
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size:18px; line-height:1;">{sig_icon}</span>
                    <span style="color:{sig_color}; font-size:13px; font-weight:700;">
                        {sinyal.replace('🔥', '').replace('⚡', '').strip()}
                    </span>
                </div>
                <div style="color:#94a3b8; font-size:10px; margin-top:4px;">
                    Score <b style="color:#e2e8f0;">{r.get('Score','?')}</b> · 
                    RRR <b style="color:#e2e8f0;">{r.get('RRR','?')}</b> · 
                    Conf <b style="color:#e2e8f0;">{r.get('Confidence','?')}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
            # ── Harga beli + floating P/L ──
            harga_beli_r = r.get('Harga_Beli', '')
            if harga_beli_r:
                floating_pl = r.get('Floating_PL', '')
                try:
                    pl_val = float(str(floating_pl).replace('%', '').replace('+', ''))
                except Exception:
                    pl_val = 0
                pl_color = "#10b981" if pl_val > 0 else ("#ef4444" if pl_val < 0 else "#94a3b8")
    
                st.markdown(f"""
                <div style="background:#1e293b; border-radius:6px; padding:8px 10px; 
                    margin-bottom:8px; border-left:3px solid {pl_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">
                                Beli
                            </div>
                            <div style="color:#e2e8f0; font-size:11px; font-weight:600;">
                                Rp {harga_beli_r}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">
                                Float P/L
                            </div>
                            <div style="color:{pl_color}; font-size:13px; font-weight:700;">
                                {floating_pl}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
            # ── Grid teknikal 2x2 ──
            entry_zone = r.get('Entry_Zone', '')
            dip_entry  = get_dip_entry(r)
            tp_range   = r.get('TP_Range', '')
            sl_harga   = r.get('SL_Harga', '')
    
            cells = []
            if entry_zone:
                cells.append(("🎯", "Entry", entry_zone, "#00ffcc"))
            if dip_entry:
                cells.append(("💡", "Dip Entry", dip_entry, "#facc15"))
            if tp_range:
                cells.append(("📈", "TP", tp_range, "#10b981"))
            if sl_harga:
                cells.append(("🛑", "SL", f"Rp {sl_harga}", "#ef4444"))
    
            if cells:
                html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-bottom:8px;">'
                for icon, label, value, color in cells:
                    html += (
                        f'<div style="background:#1e293b; border-radius:5px; padding:6px 8px; '
                        f'border-left:2px solid {color};">'
                        f'<div style="color:#94a3b8; font-size:8px; text-transform:uppercase;">'
                        f'{icon} {label}</div>'
                        f'<div style="color:{color}; font-size:10px; font-weight:600; '
                        f'margin-top:2px; word-break:break-word;">'
                        f'{value}</div>'
                        f'</div>'
                    )
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)
    
            # ── Status actual / outcome ──
            waktu_key = r.get('Waktu','')
            saham_key = r.get('Saham','')
            gaya_key = r.get('Gaya', 'SW')
            mode_actual = "swing" if gaya_key == "SW" else "daytrade"
            actual_data = (
                st.session_state.riwayat_actual.get((waktu_key, saham_key, gaya_key)) or
                st.session_state.riwayat_actual.get((waktu_key, saham_key, mode_actual)) or
                st.session_state.riwayat_actual.get((waktu_key, saham_key))
            )
    
            has_actual = False
            if actual_data:
                if (actual_data.get('Actual_High') or 
                    actual_data.get('Actual_Low') or 
                    actual_data.get('Actual_Close') or 
                    actual_data.get('Outcome') or 
                    actual_data.get('Entry_Miss') == 'Yes'):
                    has_actual = True
    
            if has_actual:
                outcome = actual_data.get('Outcome', '')
                entry_miss = actual_data.get('Entry_Miss') == 'Yes'
                
                if entry_miss or outcome == 'Not Touched':
                    out_icon, out_color, out_label = "⚪", "#94a3b8", "NOT TOUCHED"
                elif outcome == 'Win':
                    out_icon, out_color, out_label = "🏆", "#10b981", "WIN"
                elif outcome == 'Loss':
                    out_icon, out_color, out_label = "💔", "#ef4444", "LOSS"
                else:
                    out_icon, out_color, out_label = "❓", "#64748b", "PENDING"
    
                hi = actual_data.get('Actual_High', '-') or '-'
                lo = actual_data.get('Actual_Low', '-') or '-'
                cl = actual_data.get('Actual_Close', '-') or '-'
    
                st.markdown(f"""
                <div style="background:{out_color}10; border:1px solid {out_color}40; 
                    border-radius:6px; padding:8px 10px; margin-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px;">
                        <span style="font-size:14px;">{out_icon}</span>
                        <span style="color:{out_color}; font-size:10px; font-weight:700; 
                            letter-spacing:1px;">{out_label}</span>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px;">
                        <div style="text-align:center;">
                            <div style="color:#94a3b8; font-size:8px;">H</div>
                            <div style="color:#e2e8f0; font-size:10px; font-weight:600;">{hi}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#94a3b8; font-size:8px;">L</div>
                            <div style="color:#e2e8f0; font-size:10px; font-weight:600;">{lo}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#94a3b8; font-size:8px;">C</div>
                            <div style="color:#e2e8f0; font-size:10px; font-weight:600;">{cl}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
            else:
                # ── Belum ada actual → cuma placeholder ──
                st.markdown("""
                <div style="background:#1e293b; border-radius:6px; padding:6px 10px; 
                    font-size:10px; color:#94a3b8; text-align:center; 
                    border:1px dashed #334155; margin-bottom:6px;">
                    ⏳ Outcome belum dicatat · <i>Cek Quick Outcome di atas</i>
                </div>
                """, unsafe_allow_html=True)
    
            # ── Tombol Hapus (full width, di bawah) ──
            del_key = f"del_{idx_key}_{waktu_key}_{saham_key}_{gaya_key}"
            if st.button("🗑️ Hapus dari Riwayat", 
                         key=del_key, 
                         use_container_width=True,
                         help="Hapus entri ini dari riwayat"):
                hapus_riwayat_item(waktu_key, saham_key, gaya=gaya_key)
                st.rerun()
    # ---------- RIWAYAT ANALISIS (dengan Search & Paginasi) ----------
    st.subheader("📜 Riwayat Analisis")
    
    if "riwayat_page" not in st.session_state:
        st.session_state.riwayat_page = 0
    if "prev_search" not in st.session_state:
        st.session_state.prev_search = ""
    
    render_notifikasi_evaluasi_riwayat()
    
    search_query = st.text_input("🔎 Cari Saham", key="search_riwayat", placeholder="Ketik kode saham...")
    
    if search_query != st.session_state.prev_search:
        st.session_state.riwayat_page = 0
        st.session_state.prev_search = search_query
    
    riwayat_data = st.session_state.riwayat if st.session_state.riwayat else []
    if search_query:
        riwayat_data = [r for r in riwayat_data if search_query.lower() in r.get('Saham', '').lower()]
    
    # ---------- Toggle tampilan per hari ----------
    group_by_day = st.checkbox("📅 Kelompokkan per Hari", value=True)

    if group_by_day:
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in riwayat_data:
            tgl = r.get('Waktu', '')[:10]
            if tgl:
                grouped[tgl].append(r)
        sorted_days = sorted(grouped.keys(), reverse=True)
        items_per_page = 5
        total_items = len(sorted_days)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

        # ── Pagination ──
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Sebelumnya", disabled=(st.session_state.riwayat_page == 0), key="prev_day"):
                    st.session_state.riwayat_page = max(0, st.session_state.riwayat_page - 1)
            with col2:
                st.markdown(
                    f"<div style='text-align:center; color:#8892b0; font-size:12px;'>"
                    f"Hal. {st.session_state.riwayat_page+1} / {total_pages}</div>",
                    unsafe_allow_html=True
                )
            with col3:
                if st.button("Selanjutnya ▶", disabled=(st.session_state.riwayat_page >= total_pages - 1), key="next_day"):
                    st.session_state.riwayat_page = min(total_pages - 1, st.session_state.riwayat_page + 1)

        start_idx = st.session_state.riwayat_page * items_per_page
        end_idx = start_idx + items_per_page
        display_days = sorted_days[start_idx:end_idx]

        if display_days:
            for day in display_days:
                entries = grouped[day]
                session_map = defaultdict(dict)
                for r in entries:
                    s_key = (r.get('Waktu', ''), r.get('Saham', ''))
                    gaya = r.get('Gaya', 'SW')
                    session_map[s_key][gaya] = r

                # ── Label tanggal Indonesia ──
                try:
                    dt_obj = datetime.strptime(day, "%Y-%m-%d")
                    day_map = {
                        "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
                        "Thursday": "Kamis", "Friday": "Jumat",
                        "Saturday": "Sabtu", "Sunday": "Minggu"
                    }
                    day_label = f"{day_map.get(dt_obj.strftime('%A'), dt_obj.strftime('%A'))}, {dt_obj.strftime('%d %b %Y')}"
                except Exception:
                    day_label = day

                # ── Expander per hari ──
                n_sesi = len(session_map)
                with st.expander(f"📅 {day_label}  ·  {n_sesi} sesi analisis", expanded=False):
                    for s_idx, ((waktu, saham), modes) in enumerate(session_map.items()):
                        r_sw = modes.get('SW')
                        r_dt = modes.get('DT')
                        harga_val = r_sw.get('Harga') if r_sw else (r_dt.get('Harga') if r_dt else '?')

                        _sig = (r_sw or r_dt or {}).get('Sinyal', '')
                        if "STRONG BUY" in _sig:
                            _border = "#10b981"
                        elif "BUY" in _sig:
                            _border = "#84cc16"
                        elif "HOLD" in _sig:
                            _border = "#3b82f6"
                        else:
                            _border = "#ef4444"

                        waktu_short = waktu.split()[1] if len(waktu.split()) > 1 else waktu

                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:8px;
                            padding:10px 14px; margin-bottom:10px;
                            border-left:4px solid {_border};">
                            <div style="display:flex; justify-content:space-between;
                                align-items:center; flex-wrap:wrap; gap:8px;">
                                <div>
                                    <span style="color:#f3f4f6; font-size:15px;
                                        font-weight:700;">{saham}</span>
                                    <span style="color:#94a3b8; font-size:11px;
                                        margin-left:8px;">@ Rp {harga_val}</span>
                                </div>
                                <div style="color:#64748b; font-size:10px;">
                                    🕒 {waktu_short}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_sw, col_dt = st.columns(2)
                        render_mode_card(r_sw, "Swing", "📆", col_sw, f"g_{day}_{s_idx}")
                        render_mode_card(r_dt, "Daytrade", "⏱️", col_dt, f"g_{day}_{s_idx}")
                        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            st.caption(
                f"📋 Menampilkan {start_idx+1}-{min(end_idx, total_items)} dari {total_items} hari"
                + (f" (hasil pencarian '{search_query}')" if search_query else "")
            )
        else:
            if search_query:
                st.caption(f"❌ Tidak ada riwayat cocok dengan '{search_query}'.")
            else:
                st.caption("Belum ada riwayat.")
    else:
        # ---------- Tampilan flat — card style compact + AI Summary ----------
        items_per_page = 10
        total_items = len(riwayat_data)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

        # ── Pagination ──
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Sebelumnya", disabled=(st.session_state.riwayat_page == 0), key="prev_flat"):
                    st.session_state.riwayat_page = max(0, st.session_state.riwayat_page - 1)
            with col2:
                st.markdown(
                    f"<div style='text-align:center; color:#8892b0; font-size:12px;'>"
                    f"Hal. {st.session_state.riwayat_page+1} / {total_pages}</div>",
                    unsafe_allow_html=True
                )
            with col3:
                if st.button("Selanjutnya ▶", disabled=(st.session_state.riwayat_page >= total_pages - 1), key="next_flat"):
                    st.session_state.riwayat_page = min(total_pages - 1, st.session_state.riwayat_page + 1)

        start_idx = st.session_state.riwayat_page * items_per_page
        end_idx = start_idx + items_per_page
        display_riwayat = riwayat_data[start_idx:end_idx]

        if display_riwayat:
            for idx, r in enumerate(display_riwayat):
                # ── Tentukan warna signal ──
                sinyal = r.get('Sinyal', '?')
                if "STRONG BUY" in sinyal:
                    sig_color, sig_icon, sig_label = "#10b981", "🔥", "STRONG BUY"
                elif "BUY" in sinyal:
                    sig_color, sig_icon, sig_label = "#84cc16", "⚡", "BUY"
                elif "HOLD" in sinyal:
                    sig_color, sig_icon, sig_label = "#3b82f6", "⏸️", "HOLD"
                else:
                    sig_color, sig_icon, sig_label = "#ef4444", "🚨", "AVOID"

                gaya = r.get('Gaya', '?')
                gaya_label = "DT" if gaya == "DT" else ("SW" if gaya == "SW" else "?")
                gaya_icon = "⏱️" if gaya == "DT" else "📆"

                saham_key = r.get('Saham', '?')
                harga = r.get('Harga', '?')
                waktu = r.get('Waktu', '?')
                score_val = r.get('Score', '?')
                gaya_color = "#06b6d4" if gaya == "DT" else "#a855f7"

                # ── Expander title (compact) ──
                expander_title = (
                    f"{sig_icon} {saham_key} @ Rp {harga}  ·  "
                    f"{sig_label} ({gaya_icon}{gaya_label})  ·  Score: {score_val}"
                )

                with st.expander(expander_title, expanded=False):
                    # ── Header card ticker ──
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1a1d24 0%,#0f1116 100%);
                        border-radius:12px; padding:14px 18px; margin-bottom:10px;
                        border-left:5px solid {sig_color};
                        border-top:1px solid #262626; border-right:1px solid #262626;
                        border-bottom:1px solid #262626;">
                        <div style="display:flex; justify-content:space-between;
                            align-items:flex-start; flex-wrap:wrap; gap:10px;">
                            <div>
                                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                                    <span style="color:{gaya_color}; font-size:10px; font-weight:700;
                                        letter-spacing:1.2px; text-transform:uppercase;">
                                        {gaya_icon} {gaya_label}
                                    </span>
                                </div>
                                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                                    <span style="color:#f3f4f6; font-size:22px;
                                        font-weight:700; letter-spacing:0.5px;">{saham_key}</span>
                                    <span style="color:#94a3b8; font-size:13px;">@ Rp {harga}</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="color:{sig_color}; font-size:15px; font-weight:700;">
                                    {sig_icon} {sig_label}
                                </div>
                                <div style="color:#64748b; font-size:10px; margin-top:2px;">
                                    🕒 {waktu}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Score / RRR / Confidence ──
                    st.markdown(f"""
                    <div style="background:#1e293b; border-radius:8px; padding:10px 14px;
                        margin-bottom:10px;">
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;">
                            <div style="text-align:center;">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                    letter-spacing:0.5px;">Score</div>
                                <div style="color:#00ffcc; font-size:14px; font-weight:700;
                                    margin-top:2px;">{r.get('Score','?')}</div>
                            </div>
                            <div style="text-align:center; border-left:1px solid #334155;
                                border-right:1px solid #334155;">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                    letter-spacing:0.5px;">RRR</div>
                                <div style="color:#e2e8f0; font-size:14px; font-weight:700;
                                    margin-top:2px;">{r.get('RRR','?')}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                    letter-spacing:0.5px;">Confidence</div>
                                <div style="color:#e2e8f0; font-size:14px; font-weight:700;
                                    margin-top:2px;">{r.get('Confidence','?')}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ═══ 📝 AI SUMMARY ═══
                    riwayat_date = waktu[:10] if waktu else ""
                    _bs = _get_broksum_for_date(saham_key, riwayat_date) if riwayat_date else None
                    _narrative = (_bs or {}).get('narrative', '').strip()
                    if _narrative:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#a855f712 0%,#1e293b 100%);
                            border-left:4px solid #a855f7; border-radius:8px;
                            padding:10px 14px; margin-bottom:10px;">
                            <div style="color:#a855f7; font-size:10px; font-weight:700;
                                letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">
                                📝 AI Summary
                            </div>
                            <div style="color:#cbd5e1; font-size:11px; line-height:1.6;
                                font-style:italic;">
                                "{_narrative}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Coppock + Regime ──
                    coppock = r.get('Coppock', '?')
                    if "Turning Up" in coppock:
                        cop_icon, cop_color = "🔼", "#10b981"
                    elif "Rising" in coppock:
                        cop_icon, cop_color = "📈", "#84cc16"
                    else:
                        cop_icon, cop_color = "📉", "#ef4444"

                    regime = r.get('Rezim', '?')
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;
                        margin-bottom:10px;">
                        <div style="background:#1e293b; border-radius:6px; padding:8px 12px;
                            border-left:2px solid {cop_color};">
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                letter-spacing:0.5px;">Coppock</div>
                            <div style="color:{cop_color}; font-size:12px; font-weight:600;
                                margin-top:2px;">{cop_icon} {coppock}</div>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; padding:8px 12px;
                            border-left:2px solid #a855f7;">
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                letter-spacing:0.5px;">Regime</div>
                            <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                margin-top:2px;">{regime}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Estimasi ──
                    est_netral = r.get('Estimasi_Netral', '?')
                    est_sinyal = r.get('Estimasi_Sinyal', '?')
                    ret_netral = r.get('Est_Return', '?')
                    ret_sinyal = r.get('Est_Return_Sinyal', '?')
                    st.markdown(f"""
                    <div style="background:#1e293b; border-radius:8px; padding:10px 14px;
                        margin-bottom:10px;">
                        <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:8px;">📊 Estimasi Harga</div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                            <div style="text-align:center;">
                                <div style="color:#64748b; font-size:9px; margin-bottom:3px;">Netral</div>
                                <div style="color:#e2e8f0; font-size:13px; font-weight:600;">{est_netral}</div>
                                <div style="color:#94a3b8; font-size:10px; margin-top:2px;">{ret_netral}</div>
                            </div>
                            <div style="text-align:center; border-left:1px solid #334155;">
                                <div style="color:#64748b; font-size:9px; margin-bottom:3px;">🎯 Sinyal</div>
                                <div style="color:#00ffcc; font-size:13px; font-weight:600;">{est_sinyal}</div>
                                <div style="color:#10b981; font-size:10px; margin-top:2px;">{ret_sinyal}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── TP & SL ──
                    tp_val = r.get('TP_Harga') or r.get('TP_Range', '?')
                    sl_val = r.get('SL_Harga', '?')
                    tp_label = "TP Sesi Berikutnya" if gaya == "DT" else "TP Besok"
                    sl_label = "SL Sesi Berikutnya" if gaya == "DT" else "SL Besok"
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;
                        margin-bottom:10px;">
                        <div style="background:#1e293b; border-radius:8px; padding:10px 12px;
                            border-left:3px solid #10b981;">
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                letter-spacing:0.5px;">📈 {tp_label}</div>
                            <div style="color:#10b981; font-size:14px; font-weight:700;
                                margin-top:4px;">{tp_val}</div>
                        </div>
                        <div style="background:#1e293b; border-radius:8px; padding:10px 12px;
                            border-left:3px solid #ef4444;">
                            <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                letter-spacing:0.5px;">🛑 {sl_label}</div>
                            <div style="color:#ef4444; font-size:14px; font-weight:700;
                                margin-top:4px;">Rp {sl_val}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Entry & Dip ──
                    entry_zone_val = r.get('Entry_Zone', '?')
                    dip_entry_val = get_dip_entry(r)
                    if (entry_zone_val and entry_zone_val != '?') or dip_entry_val:
                        st.markdown(f"""
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;
                            margin-bottom:10px;">
                            <div style="background:#1e293b; border-radius:8px; padding:10px 12px;
                                border-left:3px solid #00ffcc;">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                    letter-spacing:0.5px;">🎯 Entry Zone</div>
                                <div style="color:#00ffcc; font-size:12px; font-weight:600;
                                    margin-top:4px;">{entry_zone_val if entry_zone_val else '-'}</div>
                            </div>
                            <div style="background:#1e293b; border-radius:8px; padding:10px 12px;
                                border-left:3px solid #facc15;">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                                    letter-spacing:0.5px;">💡 Dip Entry (RRR 1:2.0)</div>
                                <div style="color:#facc15; font-size:12px; font-weight:600;
                                    margin-top:4px;">{dip_entry_val if dip_entry_val else '-'}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Indikator teknikal ──
                    rsi = r.get('RSI', '?')
                    rsi_status = r.get('RSI_Status', '')
                    vol_surge = r.get('Vol_Surge', '?')
                    vs_status = r.get('VS_Status', '')
                    zscore = r.get('ZScore', '?')
                    zs_status = r.get('ZS_Status', '')
                    trend = r.get('Trend_Consistency', '?')

                    def _ind_color(status):
                        s = str(status).lower()
                        if "overbought" in s or "high" in s or "tinggi" in s:
                            return "#ef4444"
                        if "oversold" in s or "low" in s or "rendah" in s:
                            return "#10b981"
                        return "#94a3b8"

                    st.markdown(f"""
                    <div style="background:#1e293b; border-radius:8px; padding:10px 14px;
                        margin-bottom:10px;">
                        <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:8px;">📊 Indikator Teknikal</div>
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px;">
                            <div style="text-align:center;">
                                <div style="color:#64748b; font-size:8px;">RSI-14</div>
                                <div style="color:#e2e8f0; font-size:12px; font-weight:700;
                                    margin-top:2px;">{rsi}</div>
                                <div style="color:{_ind_color(rsi_status)}; font-size:8px;
                                    margin-top:1px;">{rsi_status}</div>
                            </div>
                            <div style="text-align:center; border-left:1px solid #334155;
                                border-right:1px solid #334155;">
                                <div style="color:#64748b; font-size:8px;">Vol Surge</div>
                                <div style="color:#e2e8f0; font-size:12px; font-weight:700;
                                    margin-top:2px;">{vol_surge}</div>
                                <div style="color:{_ind_color(vs_status)}; font-size:8px;
                                    margin-top:1px;">{vs_status}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#64748b; font-size:8px;">Z-Score</div>
                                <div style="color:#e2e8f0; font-size:12px; font-weight:700;
                                    margin-top:2px;">{zscore}</div>
                                <div style="color:{_ind_color(zs_status)}; font-size:8px;
                                    margin-top:1px;">{zs_status}</div>
                            </div>
                            <div style="text-align:center; border-left:1px solid #334155;">
                                <div style="color:#64748b; font-size:8px;">Trend</div>
                                <div style="color:#e2e8f0; font-size:12px; font-weight:700;
                                    margin-top:2px;">{trend}</div>
                                <div style="color:#94a3b8; font-size:8px; margin-top:1px;">Cons.</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Beta / Momentum / Likuiditas ──
                    beta = r.get('Beta', '?')
                    momentum = r.get('Momentum', '?')
                    likuiditas = r.get('Likuiditas', '?')
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px;
                        margin-bottom:10px;">
                        <div style="background:#1e293b; border-radius:6px; padding:8px 10px;">
                            <div style="color:#94a3b8; font-size:8px; text-transform:uppercase;
                                letter-spacing:0.5px;">Beta</div>
                            <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                margin-top:2px;">{beta}</div>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; padding:8px 10px;">
                            <div style="color:#94a3b8; font-size:8px; text-transform:uppercase;
                                letter-spacing:0.5px;">Momentum 5D</div>
                            <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                margin-top:2px;">{momentum}</div>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; padding:8px 10px;">
                            <div style="color:#94a3b8; font-size:8px; text-transform:uppercase;
                                letter-spacing:0.5px;">Likuiditas</div>
                            <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                margin-top:2px;">{likuiditas}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Status Posisi ──
                    if r.get('Status_Posisi', '') == 'Sudah Beli':
                        harga_beli_r = r.get('Harga_Beli', '')
                        floating_pl = r.get('Floating_PL', '')
                        try:
                            pl_val = float(str(floating_pl).replace('%', '').replace('+', ''))
                        except Exception:
                            pl_val = 0
                        pl_color = "#10b981" if pl_val > 0 else ("#ef4444" if pl_val < 0 else "#94a3b8")
                        st.markdown(f"""
                        <div style="background:#1e293b; border-radius:8px; padding:10px 14px;
                            margin-bottom:10px; border-left:3px solid {pl_color};">
                            <div style="display:flex; justify-content:space-between;
                                align-items:center; flex-wrap:wrap; gap:8px;">
                                <div>
                                    <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">
                                        💰 Harga Beli</div>
                                    <div style="color:#e2e8f0; font-size:13px; font-weight:600;
                                        margin-top:2px;">Rp {harga_beli_r}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">
                                        Floating P/L</div>
                                    <div style="color:{pl_color}; font-size:15px; font-weight:700;
                                        margin-top:2px;">{floating_pl}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Cek actual data ──
                    waktu_key = r.get('Waktu','')
                    gaya_key = r.get('Gaya', 'SW')
                    mode_actual = "swing" if gaya_key == "SW" else "daytrade"
                    actual_data = (
                        st.session_state.riwayat_actual.get((waktu_key, saham_key, gaya_key)) or
                        st.session_state.riwayat_actual.get((waktu_key, saham_key, mode_actual)) or
                        st.session_state.riwayat_actual.get((waktu_key, saham_key))
                    )

                    has_actual = False
                    if actual_data:
                        if (actual_data.get('Actual_High') or actual_data.get('Actual_Low') or
                            actual_data.get('Actual_Close') or actual_data.get('Outcome') or
                            actual_data.get('Entry_Miss') == 'Yes'):
                            has_actual = True

                    if has_actual:
                        outcome = actual_data.get('Outcome', '')
                        entry_miss = actual_data.get('Entry_Miss') == 'Yes'
                        if entry_miss or outcome == 'Not Touched':
                            oi, oc, ol = "⚪", "#94a3b8", "NOT TOUCHED"
                        elif outcome == 'Win':
                            oi, oc, ol = "🏆", "#10b981", "WIN"
                        elif outcome == 'Loss':
                            oi, oc, ol = "💔", "#ef4444", "LOSS"
                        else:
                            oi, oc, ol = "❓", "#64748b", "PENDING"

                        hi = actual_data.get('Actual_High', '-') or '-'
                        lo = actual_data.get('Actual_Low', '-') or '-'
                        cl = actual_data.get('Actual_Close', '-') or '-'

                        st.markdown(f"""
                        <div style="background:{oc}10; border:1px solid {oc}40;
                            border-radius:8px; padding:10px 14px; margin-bottom:6px;">
                            <div style="display:flex; align-items:center; gap:8px;
                                margin-bottom:8px;">
                                <span style="font-size:16px;">{oi}</span>
                                <span style="color:{oc}; font-size:11px; font-weight:700;
                                    letter-spacing:1px;">{ol}</span>
                            </div>
                            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px;">
                                <div style="text-align:center;">
                                    <div style="color:#94a3b8; font-size:9px;">High</div>
                                    <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                        margin-top:2px;">{hi}</div>
                                </div>
                                <div style="text-align:center; border-left:1px solid {oc}30;
                                    border-right:1px solid {oc}30;">
                                    <div style="color:#94a3b8; font-size:9px;">Low</div>
                                    <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                        margin-top:2px;">{lo}</div>
                                </div>
                                <div style="text-align:center;">
                                    <div style="color:#94a3b8; font-size:9px;">Close</div>
                                    <div style="color:#e2e8f0; font-size:11px; font-weight:600;
                                        margin-top:2px;">{cl}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:#1e293b; border-radius:6px; padding:8px 12px;
                            font-size:10px; color:#94a3b8; text-align:center;
                            border:1px dashed #334155; margin-bottom:6px;">
                            ⏳ Outcome belum dicatat · <i>Cek Quick Outcome di atas</i>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── AI Insight ──
                    ai = r.get("AI_Insight", "").strip()
                    if ai:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#8b5cf615 0%,#1e293b 100%);
                            border-left:3px solid #8b5cf6; border-radius:6px;
                            padding:8px 12px; margin-bottom:8px; font-size:10px;
                            color:#cbd5e1; line-height:1.5;">
                            💡 <b>AI Insight:</b> {ai[:200]}{'...' if len(ai) > 200 else ''}
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Tombol Hapus ──
                    hapus_key = f"hapus_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                    if st.button("🗑️ Hapus dari Riwayat", key=hapus_key,
                                 use_container_width=True):
                        hapus_riwayat_item(waktu_key, saham_key, gaya=gaya_key)
                        st.rerun()

            st.caption(
                f"📋 Menampilkan {start_idx+1}-{min(end_idx, total_items)} dari {total_items} riwayat"
                + (f" (hasil pencarian '{search_query}')" if search_query else "")
            )
        else:
            if search_query:
                st.caption(f"❌ Tidak ada riwayat cocok dengan '{search_query}'.")
            else:
                st.caption("Belum ada riwayat.")    
    
    st.markdown("---")
    st.subheader("🧠 AI (Gemini)")
    def get_api_key():
        try: return st.secrets["GEMINI_API_KEY"]
        except Exception: pass
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key: return env_key
        return st.session_state.get("gemini_api_key", "")

    api_key_loaded = get_api_key()
    st.session_state.gemini_api_key = api_key_loaded

    if api_key_loaded:
        st.success("🟢 Gemini API Key Terhubung")
    else:
        st.warning("⚠️ Gemini API Key belum ada di Secrets / ENV.")
    ai_riwayat_btn = st.button("📊 Analisis Riwayat dgn AI", use_container_width=True)
    if st.button("🗑️ Hapus Semua Riwayat"):
        try:
            sheet = get_gsheet().worksheet("riwayat")
            sheet.clear()
            st.session_state.riwayat = []
            st.success("Riwayat dihapus!")
        except Exception as e:
            st.error(f"Gagal menghapus riwayat: {e}")

    # ---------- KALENDER BURSA ----------
    st.markdown("---")
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_str = now_jkt.strftime("%Y-%m-%d")
    today_day = now_jkt.strftime("%A")
    current_hour, current_minute = now_jkt.hour, now_jkt.minute
    current_year = now_jkt.strftime("%Y")
    st.subheader(f"📅 Kalender Bursa {current_year}")
    libur_bursa = {
        "2025-01-01": "Tahun Baru Masehi", "2025-01-29": "Tahun Baru Imlek", "2025-03-14": "Hari Suci Nyepi",
        "2025-04-18": "Wafat Yesus Kristus", "2025-05-01": "Hari Buruh", "2025-05-29": "Kenaikan Yesus Kristus",
        "2025-05-30": "Hari Raya Waisak", "2025-06-06": "Idul Adha", "2025-06-27": "Tahun Baru Islam",
        "2025-08-17": "Hari Kemerdekaan", "2025-09-05": "Maulid Nabi", "2025-12-25": "Hari Raya Natal",
        "2026-01-01": "Tahun Baru Masehi", "2026-02-17": "Tahun Baru Imlek", "2026-03-03": "Hari Suci Nyepi",
        "2026-04-03": "Wafat Yesus Kristus", "2026-05-01": "Hari Buruh", "2026-05-14": "Kenaikan Yesus Kristus",
        "2026-05-15": "Hari Raya Waisak", "2026-05-25": "Idul Adha", "2026-06-15": "Tahun Baru Islam",
        "2026-08-17": "Hari Kemerdekaan", "2026-08-24": "Maulid Nabi", "2026-12-25": "Hari Raya Natal",
    }
    def dalam_jam_perdagangan(hour, minute):
        sesi1 = (hour == 9 and minute >= 0) or (10 <= hour < 12) or (hour == 12 and minute == 0)
        sesi2 = (hour == 13 and minute >= 30) or (hour == 14) or (hour == 15 and minute == 0)
        return sesi1 or sesi2
    if today_str in libur_bursa: st.warning(f"Hari ini bursa **TUTUP**: {libur_bursa[today_str]}")
    elif today_day in ["Saturday", "Sunday"]: st.warning("Hari ini **AKHIR PEKAN**, bursa tutup.")
    elif dalam_jam_perdagangan(current_hour, current_minute): st.success("Bursa **TERBUKA** (Sesi 1: 09:00-12:00, Sesi 2: 13:30-15:00 WIB)")
    else: st.info("Bursa **TUTUP** (di luar jam perdagangan).")
    st.caption("Libur dalam 2 minggu ke depan:")
    future_libur = []
    for date_str, desc in libur_bursa.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        delta = (dt.date() - now_jkt.date()).days
        if 0 < delta <= 14: future_libur.append(f"- {dt.strftime('%d %b')}: {desc}")
    if future_libur:
        for item in future_libur: st.caption(item)
    else: st.caption("Tidak ada libur dalam 2 minggu.")
    st.markdown("---")
    st.caption("Data dari Yahoo Finance. Bukan rekomendasi investasi.")
    # ==================== FUNGSI DATA & INDIKATOR ====================
@st.cache_data(ttl=60)
def load_stock_data(ticker, period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, prepost=True, actions=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ▼▼▼ FIX: Drop baris yang Close-nya NaN ▼▼▼
    try:
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
    except Exception:
        pass
    # ▲▲▲
    
    return df

@st.cache_data(ttl=60)
def load_ihsg_data(period="2y", interval="1d"):
    df = yf.download("^JKSE", period=period, interval=interval, prepost=True, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ▼ FIX: Drop baris NaN ▼
    try:
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
    except Exception:
        pass
    
    return df
@st.cache_data(ttl=3600)
def get_daftar_saham(mode):
    """Mengembalikan list kode saham (tanpa .JK) berdasarkan mode scan."""
    # Daftar statis fallback (contoh, kamu bisa lengkapi sendiri)
    lq45 = ["AADI", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ASII", "BBCA", "BBNI",
            "BBRI", "BBTN", "BMRI", "BRPT", "BUMI", "CPIN", "CUAN", "DEWA", "EMTK", "ESSA",
            "EXCL", "GOTO", "HRTA", "ICBP", "INCO", "INDF", "INDY", "INKP", "ISAT", "ITMG",
            "JPFA", "KLBF", "MAPI", "MBMA", "MDKA", "MEDC", "NCKL", "PGAS", "PGEO", "PTBA",
            "SCMA", "TLKM", "UNTR", "UNVR", "WIFI"]
    
    papan_utama = lq45 + ["AALI", "ABMM", "ACES", "ADHI", "AISA", "ALDO", "AMAG", "APLN", "ARNA", "ARTO",
                          "ASGR", "ASRI", "ASSA", "AUTO", "BACA", "BALI", "BAYU", "BBHI", "BBMD", "BBYB",
                          "BCAP", "BDMN", "BEST", "BFIN", "BGTG", "BINA", "BIRD", "BISI", "BJBR", "BJTM",
                          "BKSL", "BMTR", "BNGA", "BNII", "BNLI", "BRMS", "BSDE", "BSIM", "BSSR", "BTPN",
                          "BUDI", "BVIC", "BWPT", "BYAN", "CASS", "CFIN", "CITA", "CMNP", "CTRA", "DILD",
                          "DKFT", "DLTA", "DMAS", "DNET", "DSNG", "DSSA", "ELSA", "ENRG", "EPMT", "ERAA",
                          "FISH", "GEMS", "GGRM", "GJTL", "GZCO", "HERO", "HEXA", "HMSP", "HRUM", "IMAS",
                          "IMPC", "INPC", "INTP", "ISSP", "JIHD", "JKON", "JRPT", "JSMR", "JSPT", "JTPE",
                          "KBLI", "KIJA", "KKGI", "KPIG", "LPCK", "LPKR", "LPPF", "LSIP", "LTLS", "MAIN",
                          "MAYA", "MBSS", "MCOR", "MEGA", "MERK", "MIDI", "MIKA", "MLBI", "MLPL", "MNCN",
                          "MPMX", "MTDL", "MTLA", "MYOR", "NISP", "NOBU", "PADI", "PALM", "PANS", "PNBN",
                          "PNIN", "PNLF", "PTPP", "PTRO", "PWON", "RAJA", "RALS", "SAME", "SGRO", "SIDO",
                          "SILO", "SIMP", "SMAR", "SMBR", "SMDR", "SMGR", "SMRA", "SMSM", "SRTG", "SSIA",
                          "SSMS", "TBIG", "TBLA", "TINS", "TKIM", "TMAS", "TOBA", "TOTL", "TOTO", "TOWR",
                          "TPIA", "TPMA", "TRIM", "TSPC", "ULTJ", "UNIC", "VICO", "WIIM", "WINS", "WTON",
                          "SHIP", "POWR", "PRDA", "BRIS", "PORT", "CARS", "CLEO", "WOOD", "MARK", "PSSI",
                          "MORA", "PBID", "IPCM", "BTPS", "SPTO", "HEAL", "TUGU", "MSIN", "MAPA", "IPCC",
                          "FILM", "PANI", "GOOD", "SKRN", "BOLA", "KOTA", "HDIT", "KEEN", "TEBE", "KEJU",
                          "PSGO", "UCID", "GLVA", "AMAR", "DMND", "SAMF", "SGER", "BBSI", "VICI", "TAPG",
                          "MASB", "BMHS", "MCOL", "MTEL", "CMRY", "STAA", "TLDN", "MTMH", "TRGU", "HATM",
                          "JARR", "ELPI", "PRAY", "CBUT", "MKTR", "OMED", "SUNI", "BDKR", "SMIL", "MAHA",
                          "ERAL", "BREN", "MSTI", "ALII", "GOLF", "DAAZ", "MDIY", "DGWG", "CBDK", "BLOG",
                          "YUPI", "MDLA", "RAAM", "JECX", "BACH", "RMKE", "AVIA", "DRMA", "AGRO"]
                        
    
    pengembangan = papan_utama + ["ABDA", "AKPI", "AKSI", "AMFG", "AMIN", "ANJT", "APEX", "APIC", "APII", "APLI",
                                  "ARGO", "ARII", "ARTA", "ASBI", "ASDM", "ASJT", "ASRM", "ATIC", "BABP", "BAJA",
                                  "BAPA", "BBKP", "BBLD", "BBRM", "BCIC", "BCIP", "BIPI", "BIPP", "BKDP", "BKSW",
                                  "BMAS", "BMSR", "BNBA", "BNBR", "BOLT", "BPFI", "BPII", "BRAM", "BRNA", "BTON",
                                  "BUKK", "BULL", "BUVA", "CEKA", "CENT", "CINT", "CLPI", "CPRO", "CSAP", "CTBN",
                                  "CTTH", "DART", "DEFI", "DGIK", "DNAR", "DOID", "DPNS", "DSFI", "DVLA", "DYAN",
                                  "ECII", "EKAD", "EMDE", "ERTX", "ESTI", "FAST", "FMII", "FORU", "FPNI", "GDST",
                                  "GDYR", "GEMA", "GIAA", "GMTD", "GOLD", "GPRA", "GSMF", "GTBO", "GWSA", "HDFA",
                                  "IATA", "ICON", "IGAR", "IKBI", "IMJS", "INAI", "INCI", "INDR", "INDS", "INDX",
                                  "INPP", "INTD", "IPOL", "ITMA", "JAWA", "JECC", "KAEF", "KBLM", "KBLV", "KDSI",
                                  "KICI", "KOBX", "KONI", "KOPI", "KRAS", "LAPD", "LEAD", "LINK", "LION", "LMPI",
                                  "LPGI", "LPIN", "LPLI", "LPPS", "LRNA", "MBAP", "MBTO", "MDIA", "MDLN", "META",
                                  "MGNA", "MICE", "MITI", "MKPI", "MLPT", "MMLP", "MRAT", "MREI", "MSKY", "MYOH",
                                  "NELY", "NIKL", "NIRO", "NRCA", "OKAS", "OMRE", "PANR", "PDES", "PEGE", "PGLI",
                                  "PICO", "PJAA", "PKPK", "PNBS", "PSAB", "PSDN", "PSKT", "PTIS", "PTSN", "PTSP",
                                  "PUDP", "PYFA", "RANC", "RBMS", "RDTX", "RELI", "RICY", "RIGS", "RODA", "ROTI",
                                  "RUIS", "SAFE", "SCCO", "SDMU", "SDPC", "SDRA", "SHID", "SIPD", "SKBM", "SKLT",
                                  "SMDM", "SMMA", "SMMT", "SOCI", "SPMA", "SQMI", "SRAJ", "SRSN", "SSTM", "STAR",
                                  "STTP", "SULI", "TALF", "TBMS", "TCID", "TGKA", "TIFA", "TIRA", "TMPO", "TRIS",
                                  "TRST", "TRUS", "UNIT", "VINS", "VOKS", "VRNA", "WAPO", "WEHA", "WOMF", "YPAS",
                                  "YULE", "CASA", "DAYA", "DPUM", "IDPR", "JGLE", "KINO", "OASA", "PBSA", "BOGA",
                                  "MINA", "CSIS", "FIRE", "KMTR", "HOKI", "MPOW", "MDKI", "BELL", "KIOS", "GMFI",
                                  "MTWI", "MCAS", "PPRE", "WEGE", "DWGL", "JMAS", "CAMP", "LCKM", "HELI", "GHON",
                                  "DFAM", "NICK", "PRIM", "TRUK", "PZZA", "TNCA", "TCPI", "RISE", "BPTR", "NFCX",
                                  "MGRO", "LAND", "MOLI", "CITY", "SAPX", "SURE", "MPRO", "YELO", "CAKK", "SATU",
                                  "POLA", "DIVA", "LUCK", "SOTS", "ZONE", "PEHA", "BEEF", "POLI", "CLAY", "NATO",
                                  "JAYA", "COCO", "JAST", "FITT", "CCSI", "SFAN", "POLU", "KJEN", "ITIC", "PAMG",
                                  "BLUE", "EAST", "LIFE", "FUJI", "INOV", "SMKL", "TFAS", "GGRP", "OPMS", "NZIA",
                                  "SLIS", "IRRA", "DMMX", "WOWS", "ESIP", "REAL", "IFII", "PMJS", "CSRA", "INDO",
                                  "AMOR", "TRIN", "PTPW", "TAMA", "IKAN", "RONY", "CSMI", "BBSS", "BHAT", "EPAC",
                                  "UANG", "PGUN", "TRJA", "SCNP", "KMDS", "PURI", "SOHO", "HOMI", "ROCK", "ENZO",
                                  "ATAP", "BANK", "WMUU", "EDGE", "UNIQ", "SNLK", "ZYRX", "NPGF", "ADCP", "HOPE",
                                  "TRUE", "LABA", "ARCI", "NICL", "UVCR", "HAIS", "OILS", "GPSO", "RSGK", "SBMA",
                                  "CMNT", "GTSI", "KUAS", "BOBA", "DEPO", "BINO", "TAYS", "SEMA", "ASLC", "NETV",
                                  "ENAK", "NTBK", "BIKE", "WIRG", "SICO", "GOTO", "ASHA", "SWID", "ARKO", "CHEM",
                                  "DEWI", "AXIO", "KRYA", "GULA", "TOOL", "BUAH", "CRAB", "MEDS", "COAL", "BELI",
                                  "BSBK", "PDPP", "KDTN", "ZATA", "MMIX", "PADA", "VTNY", "ELIT", "BEER", "CBPE",
                                  "CBRE", "WINE", "PEVE", "LAJU", "FWCT", "IRSX", "VAST", "HALO", "FUTR", "PTMP",
                                  "TRON", "NSSS", "GTRA", "JATI", "TYRE", "MPXL", "KLAS", "MAXI", "VKTR", "CRSN",
                                  "INET", "RMKO", "CNMA", "FOLK", "GRIA", "PPRI", "CYBR", "MUTU", "HUMI", "RSCH",
                                  "BABY", "IOTF", "KOCI", "PTPS", "STRK", "KOKA", "RGAS", "IKPM", "AYAM", "SURI",
                                  "ASLI", "GRPH", "SMGA", "UNTD", "TOSK", "MPIX", "MKAP", "LIVE", "HYGN", "BAIK",
                                  "VISI", "AREA", "MHKI", "ATLA", "DATA", "SOLA", "BATR", "PART", "ISEA", "BLES",
                                  "GUNA", "LABS", "DOSS", "NEST", "VERN", "BOAT", "NAIK", "KSIX", "RATU", "YOII",
                                  "HGII", "BRRC", "OBAT", "MINE", "ASPR", "PSAT", "COIN", "CDIA", "MERI", "KAQI",
                                  "FORE", "DKHH", "AYLS", "DADA", "ASPI", "ESTA", "BESS", "AMAN", "CARE", "PIPA",
                                  "NCKL", "AWAN", "DOOH", "CGAS", "NICE", "MSJA", "SMLE", "ACRO", "WIFI", "FAPA",
                                  "DCII", "KETR", "DGNS", "UFOE", "CHEK", "PMUI", "EMAS", "PJHB", "RLCO", "SUPA",
                                  "WBSA", "JELI", "EMMI", "PRDL", "RANS", "OBMD", "NASI", "BSML", "ADMF", "ADMG",
                                  "AGII", "AGRS", "AHAP", "AIMS", "PNSE"]
    akselerasi_ekonomi = ["CASH", "SOFA", "PPGL", "PLAN", "LFLO", "LUCY", "MGLV", "IPAC", "FLMC", "RUNS",
                          "IDEA", "WGSH", "SMKM", "NANO", "IBOS", "OLIV", "RCCC", "AMMS", "EURO", "KLIN",
                          "NINE", "ISAP", "SOUL", "BMBL", "NAYZ", "PACK", "CHIP", "KING", "HAJJ", "RELF",
                          "GRPM", "WIDI", "HBAT", "LMAX", "MSIE", "AEGS", "LOPI", "UDNG", "MEJA", "SPRE",
                          "MANG", "BUKA"]
    pemantauan_khusus = ["ABBA", "ACST", "ADES", "AKKU", "ALKA", "ALMI", "ALTO", "ARTI", "ASMI",
                         "BATA", "BEKS", "BHIT", "BIKA", "BIMA", "BLTA", "BLTZ", "BSWD", "BTEK", "BTEL",
                         "CANI", "CMPP", "CNKO", "COWL", "DUTI", "ELTY", "ETWA", "FASW", "GAMA", "GLOB",
                         "GOLL", "HADE", "HITS", "HOME", "HOTL", "IBFN", "IBST", "IIKP", "IKAI", "INAF",
                         "INRU", "INTA", "KARW", "KBRI", "KIAS", "KOIN", "KREN", "LCGP", "LMAS", "LMSH",
                         "MAGP", "MDRN", "MFMI", "MIRA", "MLIA", "MPPA", "MTFN", "MTSM", "MYTX", "OCAP",
                         "PBRX", "PLAS", "PLIN", "RIMO", "SCPI", "SIMA", "SKYB", "SMCB", "SMRU", "SONA",
                         "SRIL", "SUGI", "SUPR", "TARA", "TAXI", "TELE", "TFCO", "TIRT", "TRAM", "TRIL",
                         "TRIO", "UNSP", "VIVA", "WICO", "WIKA", "WSKT", "ZBRA", "MARI", "MKNT", "MTRA",
                         "INCF", "WSBP", "TAMU", "TGRA", "TOPS", "ARMY", "MAPB", "MABA", "NASA", "ZINC",
                         "PCAR", "BOSS", "JSKY", "INPS", "TDPM", "SWAT", "POLL", "NUSA", "ANDI", "DIGI",
                         "HKMU", "DUCK", "SOSS", "DEAL", "URBN", "FOOD", "MTPS", "CPRI", "HRME", "POSA",
                         "KAYU", "IPTV", "ENVY", "ARKA", "BAPI", "PURE", "SINI", "IFSH", "PGJO", "PURA",
                         "SBAT", "KBAG", "CBMF", "TECH", "TOYS", "PNGO", "PTDU", "PMMP", "BEBS", "FIMP",
                         "BAUT", "WINR", "RAFI", "KKES", "HILL", "SAGE", "TGUK", "RGAS", "PTMR", "MENN",
                         "WMPP", "IPPE", "POLY", "POOL", "PPRO"]
    # Set untuk filter cepat
    pemantauan_khusus_set = set(pemantauan_khusus)
    # Gabungan dasar semua emiten statis (non-khusus)
    base_non_khusus = list(dict.fromkeys(pengembangan + akselerasi_ekonomi))
    # Full IDX = non-Pemantauan Khusus
    full_idx_static_non_khusus = [
        c for c in base_non_khusus
        if c not in pemantauan_khusus_set
    ]
    # Auto-Fetch = SEMUA emiten (termasuk Pemantauan Khusus)
    full_idx_static_all = list(dict.fromkeys(
        base_non_khusus + pemantauan_khusus
    ))
    if mode == "Cepat (LQ45)":
        return lq45
    elif mode == "Papan Utama":
        return papan_utama
    elif mode == "Komprehensif (Utama + Pengembangan)":
        return pengembangan
    elif mode == "Full IDX":
        api_codes = fetch_all_idx_stocks()
        if api_codes:
            combined = list(dict.fromkeys(api_codes + full_idx_static_non_khusus))
            combined = [c for c in combined if c not in pemantauan_khusus_set]
            return combined
        return full_idx_static_non_khusus
    elif mode == "Auto-Fetch (API BEI)":
        api_codes = fetch_all_idx_stocks()
        if api_codes:
            combined = list(dict.fromkeys(api_codes + full_idx_static_all))
            # jangan filter apa pun, semua emiten diikutsertakan
            return combined[:1000]
        return full_idx_static_all
@st.cache_data(ttl=30)
def get_realtime_price(ticker):
    """Ambil harga real-time terpisah dari bar historis, untuk override kalau bar terakhir stale."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        return {
            "last_price": fi.get("last_price"),
            "market_state": fi.get("market_state") if hasattr(fi, "get") else None,
        }
    except Exception:
        return None

def cek_kesegaran_data(df_ihsg_preview, now_jkt, max_lag_minutes=20):
    """Return (is_stale, lag_minutes)."""
    if df_ihsg_preview.empty:
        return True, None
    last_bar_time = df_ihsg_preview.index[-1]
    if last_bar_time.tzinfo is None:
        last_bar_time = pytz.timezone("Asia/Jakarta").localize(last_bar_time)
    else:
        last_bar_time = last_bar_time.astimezone(pytz.timezone("Asia/Jakarta"))
    lag = (now_jkt - last_bar_time).total_seconds() / 60
    return lag > max_lag_minutes, lag
    
def compute_adx_series(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up>down)&(up>0), up, 0.0)
    minus_dm = np.where((down>up)&(down>0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di-minus_di)/(plus_di+minus_di))*100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def get_google_news_rss(query_str, num=5, days_back=7):
    """Ambil berita dari Google News RSS, difilter hanya N hari terakhir dan diurutkan terbaru."""
    if not RSS_AVAILABLE: return [], "RSS tidak tersedia"
    try:
        # Tambah filter after: agar Google hanya return berita terbaru
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        query_with_date = f"{query_str} after:{cutoff}"
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_with_date)}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        news = []
        for e in feed.entries[:num * 2]:  # ambil lebih banyak dulu untuk difilter
            published = e.get('published', '')
            published_parsed = e.get('published_parsed')  # struct_time
            news.append({
                'title': e.get('title', '').strip(),
                'summary': re.sub('<[^<]+?>', '', e.get('summary', '')),
                'source': 'Google News',
                'published': published,
                'published_ts': time.mktime(published_parsed) if published_parsed else 0
            })

        # Sort terbaru di atas
        news.sort(key=lambda x: x['published_ts'], reverse=True)
        return news[:num], None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)  # cache 10 menit (lebih fresh)
def get_headlines_for_ticker(ticker):
    """Ambil maks 3 judul berita terbaru dari Google News RSS."""
    try:
        news, _ = get_google_news_rss(f"{ticker} saham", num=3, days_back=7)
        return [n['title'] for n in news] if news else ["(tidak ada berita terbaru)"]
    except:
        return ["(gagal mengambil berita)"]
def get_ipot_news(query, num=5):
    """Ambil berita dari Ipotnews berdasarkan kata kunci."""
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f"https://www.ipotnews.com/?q={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        news = []
        
        # Cari semua elemen <a> dan filter yang textnya cukup panjang (judul berita)
        for a in soup.find_all('a'):
            t = a.get_text(strip=True)
            # Biasanya judul berita panjang, kita ambil yang > 30 karakter
            if len(t) > 30 and 'berita' not in t.lower() and 'indopremier' not in t.lower():
                # Pastikan judul unik
                if not any(n['title'] == t for n in news):
                    news.append({'title': t, 'summary': '', 'source': 'Ipotnews'})
                if len(news) >= num:
                    break
        return news, None
    except Exception as e:
        return [], str(e)

def get_yahoo_search_news(query_str, num=5, days_back=7):
    try:
        items = yf.Search(query_str).news or []
        cutoff_ts = time.time() - (days_back * 86400)
        news = []
        for item in items[:num * 2]:
            inner = item.get('content') or item
            title = inner.get('title') or inner.get('shortTitle') or inner.get('headline') or ''
            summary = inner.get('summary') or inner.get('longSummary') or inner.get('description') or ''
            pub_ts = inner.get('providerPublishTime') or 0  # unix timestamp
            # Filter hanya berita dalam N hari terakhir
            if title and (pub_ts == 0 or pub_ts >= cutoff_ts):
                news.append({
                    'title': title,
                    'summary': summary,
                    'source': 'Yahoo Search',
                    'published': datetime.fromtimestamp(pub_ts).strftime('%d %b %Y') if pub_ts else '',
                    'published_ts': pub_ts
                })
        # Sort terbaru di atas
        news.sort(key=lambda x: x['published_ts'], reverse=True)
        return news[:num], None
    except:
        return [], "Yahoo Search gagal"

def filter_relevant(news_list, ticker):
    keywords = [ticker.lower(),'saham','ihsg','bei','idx']
    filtered = [n for n in news_list if any(k in (n['title']+n['summary']).lower() for k in keywords)]
    return filtered if filtered else news_list

# ═══════════════════════════════════════════════════════════════
# IDX FIN-LEXICON — Kamus Pasar Modal Indonesia
# Menggantikan ketergantungan penuh pada VADER (kamus bahasa Inggris umum)
# yang tidak mengenali istilah seperti PKPU, suspensi, buyback, UMA, dll.
# ═══════════════════════════════════════════════════════════════
IDX_FIN_LEXICON = {
    # ── Katalis Positif ──
    "dividen jumbo": +0.85,
    "dividen spesial": +0.80,
    "dividen interim": +0.70,
    "kenaikan dividen": +0.72,
    "buyback saham": +0.75,
    "buy back": +0.65,
    "buyback": +0.65,
    "tender offer": +0.80,
    "akuisisi": +0.60,
    "merger": +0.55,
    "kontrak baru": +0.72,
    "proyek baru": +0.65,
    "lonjakan laba": +0.82,
    "laba bersih meningkat": +0.78,
    "laba melonjak": +0.80,
    "pendapatan meningkat": +0.68,
    "rights issue standby buyer": +0.62,
    "standby buyer": +0.60,
    "ekspansi kapasitas": +0.58,
    "ekspansi bisnis": +0.55,
    "kemitraan strategis": +0.60,
    "pembelian kembali": +0.65,
    "restrukturisasi berhasil": +0.60,
    "pelunasan utang": +0.65,
    "upgrade rating": +0.70,
    "kenaikan target harga": +0.68,
    # ── Katalis Negatif ──
    "suspensi perdagangan": -0.92,
    "suspensi": -0.85,
    "dihentikan perdagangannya": -0.88,
    "penghentian perdagangan": -0.88,
    "pkpu": -0.92,
    "penundaan kewajiban pembayaran utang": -0.90,
    "pailit": -0.95,
    "kepailitan": -0.95,
    "bangkrut": -0.95,
    "default obligasi": -0.92,
    "gagal bayar": -0.88,
    "wanprestasi": -0.85,
    "repo gagal": -0.87,
    "pengunduran diri direksi mendadak": -0.78,
    "mundur direksi": -0.72,
    "pengunduran diri direktur": -0.70,
    "pengunduran diri": -0.55,
    "rugi bersih membengkak": -0.82,
    "rugi bersih meningkat": -0.78,
    "rugi bersih": -0.72,
    "kerugian meningkat": -0.70,
    "pendapatan turun": -0.60,
    "laba tergerus": -0.65,
    "uma": -0.72,  # Unusual Market Activity
    "unusual market activity": -0.72,
    "reverse stock split": -0.68,
    "pemecahan saham terbalik": -0.68,
    "right issue tanpa standby": -0.55,
    "dilusi saham": -0.58,
    "penambahan modal tanpa hmetd": -0.65,
    "pmthmetd": -0.62,
    "gugatan": -0.62,
    "gugatan hukum": -0.68,
    "investigasi otoritas": -0.75,
    "sanksi ojk": -0.80,
    "pembekuan": -0.78,
    "delisting": -0.92,
    "force majeure": -0.60,
}

def _score_lexicon_idxfin(text_lower: str) -> tuple[float, bool]:
    """
    Cek apakah ada kata dari IDX_FIN_LEXICON dalam teks.
    Return (skor_rata_rata, ada_match).
    Jika multiple match → rata-rata tertimbang (kata lebih panjang = bobot lebih tinggi).
    """
    matches = []
    for phrase, score in IDX_FIN_LEXICON.items():
        if phrase in text_lower:
            matches.append((len(phrase), score))
    if not matches:
        return 0.0, False
    # Bobot proporsional terhadap panjang frasa (frasa panjang lebih spesifik)
    total_w = sum(l for l, _ in matches)
    weighted = sum(l * s for l, s in matches) / total_w
    return float(np.clip(weighted, -1.0, 1.0)), True

def analyze_sentiment_weighted(news_items, translator):
    """
    Hitung rata-rata sentimen tertimbang dari daftar berita.

    Upgrade dari VADER-only ke IDX Fin-Lexicon blended:
    - Jika ada kata IDX_FIN_LEXICON dalam teks → 60% lexicon + 40% VADER
    - Jika tidak ada → VADER saja (backward-compatible)
    - Bobot per berita = 1/(i+1) (berita terbaru lebih berat)
    """
    if not SENTIMENT_AVAILABLE or not news_items:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    total_w, w_sum = 0, 0
    for i, item in enumerate(news_items):
        text = f"{item['title']}. {item['summary']}" if item['summary'] else item['title']
        text_for_lexicon = text.lower()

        # ── IDX Fin-Lexicon check (bahasa Indonesia, sebelum translate) ──
        lex_score, has_lex = _score_lexicon_idxfin(text_for_lexicon)

        # ── Terjemahkan untuk VADER (bahasa Inggris) ──
        if any(ord(c) > 127 for c in text) and translator:
            try:
                text = translator.translate(text)
            except Exception:
                pass
        vader_score = analyzer.polarity_scores(text)['compound']

        # ── Blend: lexicon override jika ditemukan kata IDX ──
        if has_lex:
            # 60% lexicon (sangat spesifik konteks IDX) + 40% VADER
            final_score = 0.6 * lex_score + 0.4 * vader_score
        else:
            final_score = vader_score

        weight = 1 / (i + 1)
        w_sum += final_score * weight
        total_w += weight
    return w_sum / total_w if total_w > 0 else 0.0

def estimate_theta_ou(close_series):
    log_price = np.log(close_series.dropna())
    log_lag = log_price.shift(1).dropna()
    diff = log_price.diff().dropna()
    common_idx = diff.index.intersection(log_lag.index)
    if len(common_idx)<20: return 0.05
    y = diff.loc[common_idx].values
    X = np.vstack([np.ones(len(common_idx)), log_lag.loc[common_idx].values]).T
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    theta = -coeff[1] if coeff[1]<0 else 0.05
    return theta
def robust_std(series):
    """Robust standard deviation berbasis MAD (Median Absolute Deviation)"""
    arr = np.array(series)
    if len(arr) < 4:
        return np.std(arr, ddof=0) if len(arr) > 1 else 0.001
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return mad * 1.4826
REGIME_INFO = {
    "Strong Bullish 🚀": "Tren naik kuat dengan momentum tinggi.",
    "Bullish 📈": "Tren naik stabil. Kondisi sehat untuk akumulasi.",
    "Panic Sell 🚨": "Penurunan tajam, sering oversold.",
    "Bearish 🔻": "Tren turun terkendali.",
    "Early Recovery 🔄": "Harga di atas EMA20 tapi EMA20 < EMA50.",
    "Distribution 📉": "Harga di bawah EMA20, EMA20 > EMA50.",
    "Konsolidasi Tren ↔️": "Trending namun harga bolak-balik di EMA.",
    "Bullish Accumulation 🏗️": "Sideways dengan harga > EMA.",
    "Bearish Accumulation 🧊": "Sideways di bawah EMA.",
    "Sideways Bias Naik ↗️": "Sideways cenderung naik.",
    "Sideways Bias Turun ↘️": "Sideways cenderung turun.",
    "Sideways Normal ↔️": "Sideways moderat, tunggu katalis."
}
def score_stock_tech(df_stock, ticker, ihsg_data):
    """
    Menghitung skor teknikal + metrik lengkap untuk scanner V12.
    Mengembalikan dictionary hasil atau None jika data tidak cukup.
    """
    try:
        closes = df_stock['Close'].values
        highs = df_stock['High'].values
        lows = df_stock['Low'].values
        volumes = df_stock['Volume'].values
        last_price = float(closes[-1])
        if last_price <= 0:
            return None

        ihsg_closes = ihsg_data['Close'].values
        if len(closes) < 65 or len(ihsg_closes) < 65:
            return None

        # Gunakan 60 bar terakhir untuk perhitungan
        n = min(60, len(closes) - 1, len(ihsg_closes) - 1)
        s_adj = closes[-n-1:]
        i_adj = ihsg_closes[-n-1:]
        sT = len(s_adj) - 1
        iT = len(i_adj) - 1
        if sT < 20 or iT < 20:
            return None

        s_ret = np.diff(s_adj) / s_adj[:-1]
        i_ret = np.diff(i_adj) / i_adj[:-1]

        # --- Beta & IHSG return 5 hari ---
        i_ret5 = (i_adj[iT] - i_adj[iT-5]) / i_adj[iT-5] if iT >= 5 else 0.0
        common_len = min(len(s_ret), len(i_ret))
        cov = np.cov(s_ret[:common_len], i_ret[:common_len])[0,1]
        var_i = np.var(i_ret[:common_len])
        beta = (cov / var_i).clip(-3, 3) if var_i > 1e-8 else 1.0
        beta_norm = np.clip(beta * i_ret5 / 0.05, -1, 1)

        # --- Momentum combo (3/5/10) ---
        mom3 = (s_adj[sT] - s_adj[sT-3]) / s_adj[sT-3] if sT >= 3 else 0.0
        mom5 = (s_adj[sT] - s_adj[sT-5]) / s_adj[sT-5] if sT >= 5 else 0.0
        mom10 = (s_adj[sT] - s_adj[sT-10]) / s_adj[sT-10] if sT >= 10 else 0.0
        mom_combo = mom3*0.50 + mom5*0.30 + mom10*0.20
        mom_norm = np.clip(mom_combo / 0.05, -1, 1)

        # --- Coppock Curve ---
        copp_std, copp_prev = coppock_curve(s_adj, 14, 11, 10)
        copp_fast, copp_fast_prev = coppock_curve(s_adj, 6, 4, 5)
        copp_rising = copp_std > copp_prev
        copp_fast_rising = copp_fast > copp_fast_prev
        is_turning_up = copp_rising and copp_prev <= 0.0
        is_turning_down = not copp_rising and copp_prev >= 0.0
        fast_align = 1.08 if (copp_fast_rising == copp_rising) else 0.92

        if is_turning_up:
            copp_dir_base = 1.0
        elif copp_std > 0 and copp_rising:
            copp_dir_base = 0.70
        elif copp_std <= 0 and copp_rising:
            copp_dir_base = 0.40
        elif is_turning_down:
            copp_dir_base = -1.0
        elif copp_std > 0 and not copp_rising:
            copp_dir_base = -0.30
        else:
            copp_dir_base = -0.70
        copp_norm = np.clip(copp_dir_base * fast_align, -1, 1)

        # Label Coppock
        if is_turning_up:
            copp_label = "🔼 Turning Up"
        elif copp_std > 0 and copp_rising:
            copp_label = "↑ Rising+"
        elif copp_std <= 0 and copp_rising:
            copp_label = "↑ Recovering"
        elif is_turning_down:
            copp_label = "🔽 Turning Down"
        else:
            copp_label = "↓ Bearish"

        # --- Mean Reversion (Z-Score) ---
        sigma20 = robust_std(s_ret[-20:])
        sma20 = np.mean(s_adj[-20:])
        z_score_val = (last_price - sma20) / (sigma20 * sma20 + 1e-9)
        mr_norm = np.clip(-z_score_val / 0.05, -1, 1)

        # --- RSI ---
        rsi_ch = s_ret[-14:]
        gains = np.mean(rsi_ch[rsi_ch > 0]) if np.any(rsi_ch > 0) else 0.0
        losses = -np.mean(rsi_ch[rsi_ch < 0]) if np.any(rsi_ch < 0) else 1e-6
        rsi_val = 100.0 - (100.0 / (1.0 + gains / (losses + 1e-9)))
        if rsi_val < 25: rsi_norm = 0.90
        elif rsi_val < 35: rsi_norm = 0.55
        elif rsi_val < 45: rsi_norm = 0.20
        elif rsi_val < 55: rsi_norm = -0.10
        elif rsi_val < 65: rsi_norm = -0.35
        elif rsi_val < 75: rsi_norm = -0.55
        else: rsi_norm = -0.80

        # --- Volume Surge ---
        vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-20:].mean()
        vol5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
        vol_surge = np.clip((vol5 / (vol_ma20 + 1) - 1.0), -1, 1)

        # --- Breakout bonus ---
        res20 = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs)
        breakout_bonus = 0.10 if (last_price > res20 * 0.995 and vol_surge > 0.3) else 0.0

        # --- Tech Score ---
        tech_score = (mom_norm*0.30 + copp_norm*0.28 + beta_norm*0.17 +
                      mr_norm*0.10 + vol_surge*0.08 + rsi_norm*0.07 +
                      breakout_bonus)
        tech_score = np.clip(tech_score, -1.0, 1.0)

        # --- Sinyal ---
        if tech_score > 0.42: signal = "STRONG BUY ▲▲"
        elif tech_score > 0.18: signal = "BUY ▲"
        elif tech_score > 0.05: signal = "WEAK BUY ▲"
        elif tech_score < -0.42: signal = "STRONG SELL ▼▼"
        elif tech_score < -0.18: signal = "SELL ▼"
        else: signal = "NEUTRAL →"

        # --- Regime ---
        ema20_ihsg = pd.Series(i_adj).ewm(span=20, adjust=False).mean().iloc[-1]
        sma20_ihsg = np.mean(i_adj[-20:])
        risk_on = ema20_ihsg > sma20_ihsg and i_ret5 > 0
        fast_vc = np.std(s_ret[-3:]) / (np.std(s_ret[-20:]) + 1e-9)
        if not risk_on and fast_vc >= 1.2: regime = "PANIC"
        elif risk_on and fast_vc >= 1.0: regime = "VOL UP"
        elif risk_on: regime = "BULLISH"
        else: regime = "BEARISH"

        # --- Estimasi return & TP/SL ---
        alpha = np.mean(s_ret) - beta * np.mean(i_ret)
        mu_est = np.clip(beta * i_ret5 + alpha + mom_combo * 0.15, -0.04, 0.04)

        # --- Metrik tambahan ---
        # Bollinger %B
        std20 = np.std(s_adj[-20:])
        upper_bb = sma20 + 2*std20
        lower_bb = sma20 - 2*std20
        bb_pct = np.clip((last_price - lower_bb) / (upper_bb - lower_bb + 1e-9), 0, 1)

        # Trend Consistency
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else ema20
        trend_searah = 0
        if len(closes) >= 6:
            for i in range(1, 6):
                if ((closes[-i] > closes[-i-1]) and (ema20 > ema50)) or \
                   ((closes[-i] < closes[-i-1]) and (ema20 < ema50)):
                    trend_searah += 1
            trend_consistency = trend_searah / 5 * 100
        else:
            trend_consistency = 50.0

        # Entry Zone sederhana (berbasis pivot & fraksi BEI)
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3.0
        s1 = 2 * pivot - highs[-1]
        entry_low = fraksi_bei(min(s1, last_price * (1 - sigma20)))
        entry_high = fraksi_bei(last_price)
        if entry_low >= entry_high:
            step = fraksi_step(last_price)
            entry_low = fraksi_bei(entry_high - step)

        # Take Profit Est (di atas harga pasar, dibulatkan ke fraksi BEI)
        tp_dist_pct = max(0.02, max(mu_est, 0.01) + 0.8 * sigma20)
        tp_est = fraksi_bei(last_price * (1 + tp_dist_pct))
        if tp_est <= last_price:
            step = fraksi_step(last_price)
            tp_est = fraksi_bei(last_price + 2 * step)

        # Stop Loss Est (selalu di bawah entry_low & last_price, dibulatkan ke fraksi BEI)
        sl_dist_pct = max(0.02, 1.5 * sigma20)
        sl_est = fraksi_bei(entry_low * (1 - sl_dist_pct))
        if sl_est >= entry_low:
            step = fraksi_step(entry_low)
            sl_est = fraksi_bei(entry_low - 2 * step)

        # Likuiditas
        avg_value = np.mean(volumes[-20:] * closes[-20:])
        if avg_value >= 1e9:
            likuiditas_str = f"Rp {avg_value/1e9:.2f} M/hari"
        elif avg_value >= 1e6:
            likuiditas_str = f"Rp {avg_value/1e6:.0f} Jt/hari"
        else:
            likuiditas_str = f"Rp {avg_value:,.0f}"

        # Risk/Reward
        risk = last_price - sl_est
        reward = tp_est - last_price
        rrr = reward / risk if risk > 0 else 0.0

        # Confidence
        confidence = min(0.99, 0.5 + abs(tech_score) * 0.5)

        return {
            "ticker": ticker,
            "techScore": tech_score,
            "signal": signal,
            "muEst": mu_est,
            "coppockLabel": copp_label,
            "lastPrice": last_price,
            "tpEst": tp_est,
            "slEst": sl_est,
            "regime": regime,
            "rsi": rsi_val,
            "beta": beta,
            "momScore": mom_combo,
            "isCoppockTurningUp": is_turning_up,
            "volSurge": vol_surge,
            "zScore": z_score_val,
            "bbPct": bb_pct,
            "trendConsistency": trend_consistency,
            "entryLow": entry_low,
            "entryHigh": entry_high,
            "likuiditas": likuiditas_str,
            "rrr": rrr,
            "confidence": confidence
        }

    except Exception as e:
        # st.write(f"Error scoring {ticker}: {e}")  # untuk debugging
        return None
def generate_regime_insight(regime, adx, ofi_raw, ihsg_cond):
    base = REGIME_INFO.get(regime, "Rezim tidak terdefinisi.")
    notes = []

    # Analisis OFI (Enhanced dengan shadow-weighted volume)
    if ofi_raw > 0.5:
        notes.append("🔹 OFI sangat positif → akumulasi agresif, bullish kuat.")
    elif ofi_raw > 0.2:
        notes.append("🔹 OFI moderat positif → akumulasi bertahap, bias bullish.")
    elif ofi_raw > -0.2:
        notes.append("🔹 OFI netral/fluktuasi → pasar balance, indecision.")
    elif ofi_raw > -0.5:
        notes.append("🔹 OFI moderat negatif → distribusi bertahap, bias bearish.")
    else:
        notes.append("🔹 OFI sangat negatif → distribusi agresif, tekanan jual kuat.")

    # Analisis ADX (selalu tampil)
    if adx > 40:
        notes.append("🔹 ADX > 40 → tren sangat kuat, tapi waspadai kejenuhan.")
    elif adx < 20:
        notes.append("🔹 ADX rendah → pasar sedang konsolidasi, breakout mungkin terjadi.")
    else:
        notes.append(f"🔹 ADX {adx:.1f} → kekuatan tren moderat.")   # ← tambahan

    # Tambahan untuk RISK-ON / RISK-OFF
    if "RISK-ON" in ihsg_cond:
        notes.append("🔹 Sentimen pasar luas mendukung (RISK-ON).")
    elif "RISK-OFF" in ihsg_cond:
        notes.append("🔹 Sentimen pasar luas sedang defensif (RISK-OFF).")
    else:
        notes.append(f"🔹 Sentimen pasar luas: {ihsg_cond}")

    if notes:
        return base + " " + " ".join(notes)
    return base
# ==================== FUNGSI ANALISIS UTAMA ====================
def analyze_stock(ticker_input, harga_manual, sudah_beli, harga_beli_float, is_daytrade, v12_mem=None, fee_beli_pct=0.15, fee_jual_pct=0.25):
    """
    Menjalankan analisis lengkap untuk satu mode (swing/daytrade).
    Mengembalikan dictionary hasil atau None jika data tidak cukup.
    """
    # ------------------------------------------------------------------
    # 1. AMBIL DATA
    # ------------------------------------------------------------------
    bars_per_day_map = {"5m": 54, "15m": 18, "30m": 9, "60m": 5}

    if is_daytrade:
        actual_interval = "5m"
        df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "15m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "30m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "60m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        df_ihsg = load_ihsg_data(period="5d", interval="5m")
        df_daily = load_stock_data(ticker_input, period="1mo", interval="1d")
    else:
        actual_interval = "1d"
        df = load_stock_data(ticker_input, period="2y", interval="1d")
        df_ihsg = load_ihsg_data(period="2y", interval="1d")
        df_daily = df

    if df.empty:
        return None

    # ── VALIDASI: pastikan Close terakhir bukan NaN ──
    try:
        _last_close = float(df['Close'].iloc[-1])
        if math.isnan(_last_close) or _last_close <= 0:
            return None
    except Exception:
        return None

    # Cek 20 bar terakhir minimal 10 valid
    try:
        _valid_count = df['Close'].tail(20).dropna().shape[0]
        if _valid_count < 10:
            return None
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 1.5. MULTI-TIMEFRAME (MTF) ANCHOR
    # ------------------------------------------------------------------
    is_mtf_bullish = True
    mtf_status_text = "N/A"
    
    try:
        if is_daytrade:
            df_anchor = load_stock_data(ticker_input, period="1y", interval="1d")
            anchor_name = "Daily"
        else:
            df_anchor = load_stock_data(ticker_input, period="3y", interval="1wk")
            anchor_name = "Weekly"
            
        if df_anchor is not None and not df_anchor.empty and len(df_anchor) >= 50:
            df_anchor = df_anchor.copy()
            df_anchor['EMA20'] = df_anchor['Close'].ewm(span=20, adjust=False).mean()
            df_anchor['EMA50'] = df_anchor['Close'].ewm(span=50, adjust=False).mean()
            
            ema12 = df_anchor['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_anchor['Close'].ewm(span=26, adjust=False).mean()
            df_anchor['MACD'] = ema12 - ema26
            df_anchor['MACD_Signal'] = df_anchor['MACD'].ewm(span=9, adjust=False).mean()
            df_anchor['MACD_Hist'] = df_anchor['MACD'] - df_anchor['MACD_Signal']
            
            last_close = float(df_anchor['Close'].iloc[-1])
            last_ema20 = float(df_anchor['EMA20'].iloc[-1])
            last_ema50 = float(df_anchor['EMA50'].iloc[-1])
            last_macd_hist = float(df_anchor['MACD_Hist'].iloc[-1])
            
            bull_score = 0
            if last_close > last_ema20: bull_score += 1
            if last_close > last_ema50: bull_score += 1
            if last_macd_hist > 0: bull_score += 1
            
            is_mtf_bullish = (bull_score >= 2)
            mtf_status_text = f"Tren {anchor_name}: {'Bullish ✅' if is_mtf_bullish else 'Bearish ⚠️'} (Score {bull_score}/3)"
    except Exception as e:
        mtf_status_text = f"Error MTF: {str(e)}"
    except Exception:
        return None

    # ------------------------------------------------------------------
    # 1.6. VSA (VOLUME SPREAD ANALYSIS) & MARKING CLOSE DETECTOR
    # ------------------------------------------------------------------
    is_marking_close = False
    is_no_demand = False
    is_stopping_volume = False
    vsa_status_text = "VSA: Normal"

    try:
        # ── a. Marking Close: fetch 5m intraday untuk cek 10 menit terakhir ──
        try:
            df_5m_today = load_stock_data(ticker_input, period="1d", interval="5m")
            if df_5m_today is not None and not df_5m_today.empty and len(df_5m_today) >= 5:
                # Ambil 2 bar terakhir (≈ 15:50–16:00 WIB)
                last2 = df_5m_today.tail(2)
                vol_last2  = last2['Volume'].mean()
                vol_avg5m  = df_5m_today['Volume'].mean()
                close_last = float(last2['Close'].iloc[-1])
                close_ref  = float(df_5m_today.iloc[-3]['Close'])   # harga sebelum 2 bar terakhir
                if close_ref > 0:
                    move_pct = (close_last - close_ref) / close_ref * 100
                    # Marking close: naik > 1.5% dengan volume < 30% rata-rata
                    if move_pct > 1.5 and vol_avg5m > 0 and vol_last2 < vol_avg5m * 0.30:
                        is_marking_close = True
        except Exception:
            pass

        # ── b. VSA pada bar terakhir data utama (Daily atau Intraday) ──
        if len(df) >= 21:
            recent = df.tail(21).copy()
            spread  = recent['High'] - recent['Low']
            avg_spread = spread.iloc[:-1].mean()
            avg_vol    = recent['Volume'].iloc[:-1].mean()

            last_bar   = recent.iloc[-1]
            last_spread = float(last_bar['High'] - last_bar['Low'])
            last_vol    = float(last_bar['Volume'])
            last_close  = float(last_bar['Close'])
            last_open   = float(last_bar['Open'])
            last_low    = float(last_bar['Low'])
            last_high   = float(last_bar['High'])
            last_rng    = last_high - last_low

            # No Demand Bar: harga naik, spread sempit, volume sepi
            if (last_close > last_open and
                avg_spread > 0 and last_spread < avg_spread * 0.70 and
                avg_vol > 0 and last_vol < avg_vol * 0.80):
                is_no_demand = True

            # Stopping Volume: harga turun tapi volume meledak, close di atas mid candle
            mid_candle = last_low + last_rng / 2 if last_rng > 0 else last_close
            if (last_close < last_open and
                avg_vol > 0 and last_vol > avg_vol * 1.8 and
                last_close >= mid_candle):
                is_stopping_volume = True

        # ── c. Rangkum status VSA ──
        vsa_tags = []
        if is_marking_close:    vsa_tags.append("⚠️ Marking Close Detected")
        if is_no_demand:        vsa_tags.append("🔴 No Demand (False Breakout Risk)")
        if is_stopping_volume:  vsa_tags.append("🟢 Stopping Volume (Potential Reversal)")
        vsa_status_text = " | ".join(vsa_tags) if vsa_tags else "✅ VSA: Normal"

    except Exception as e:
        vsa_status_text = f"VSA Error: {str(e)}"

    # ------------------------------------------------------------------
    # 2. PERHITUNGAN DASAR
    # ------------------------------------------------------------------
    harga_terakhir_asli = float(df['Close'].iloc[-1])
    harga_terakhir = harga_terakhir_manual if harga_terakhir_manual else harga_terakhir_asli

    floating_pl_pct = None
    if sudah_beli and harga_beli_float and harga_beli_float > 0:
        floating_pl_pct = (harga_terakhir - harga_beli_float) / harga_beli_float * 100

    returns = df['Close'].pct_change().dropna()
    if len(returns) < 20:
        return None

    # ------------------------------------------------------------------
    # 3. INDIKATOR TEKNIKAL
    # ------------------------------------------------------------------
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = compute_adx_series(df)

    if is_daytrade:
        df['Mom5D'] = df['Close'].pct_change(10) * 100   # 10 bar intraday
    else:
        df['Mom5D'] = df['Close'].pct_change(5) * 100    # 5 hari

    df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

    # OFI - Original (Simple)
    df['Delta'] = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    df['Cumulative_OFI'] = df['Delta'].cumsum()
    df['OFI_raw'] = df['Delta'] / df['Volume'].rolling(20).mean().fillna(1)

    # OFI - Enhanced (Shadow-based Volume Allocation)
    # Upper shadow = High - Close (rejection at top)
    # Lower shadow = Open - Low (support testing)
    df['Upper_Shadow'] = df['High'] - df['Close']
    df['Lower_Shadow'] = df['Open'] - df['Low']
    df['Range'] = df['High'] - df['Low'] + 0.0001  # Avoid division by zero
    
    # Shadow ratio (0-1): larger shadow = more rejection/support
    df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Range']
    df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Range']
    
    # Weighted Delta: reduce volume if candle has large shadow (less conviction)
    df['Delta_Enhanced'] = np.where(
        df['Close'] > df['Open'],
        # Bullish candle: reduce by upper shadow (rejeksi di atas)
        df['Volume'] * (1 - df['Upper_Shadow_Ratio']),
        # Bearish candle: reduce by lower shadow (support di bawah)
        -df['Volume'] * (1 - df['Lower_Shadow_Ratio'])
    )
    
    df['Cumulative_OFI_Enhanced'] = df['Delta_Enhanced'].cumsum()
    df['OFI_Enhanced'] = df['Cumulative_OFI_Enhanced'] / (df['Volume'].rolling(20).mean().fillna(1))

    # VWAP hanya untuk daytrade
    if is_daytrade:
        df['CumVol'] = df['Volume'].cumsum()
        df['CumPV'] = (df['Close'] * df['Volume']).cumsum()
        df['VWAP'] = df['CumPV'] / df['CumVol']
        vwap_now = df['VWAP'].iloc[-1]
        vwap_bias = "Di Atas VWAP (Bullish)" if harga_terakhir > vwap_now else "Di Bawah VWAP (Bearish)"
    else:
        vwap_now = None
        vwap_bias = "N/A"

    # ------------------------------------------------------------------
    # 4. FUNDAMENTAL
    # ------------------------------------------------------------------
    try:
        ticker_info = yf.Ticker(ticker_input).info
    except:
        ticker_info = {}
    mc = ticker_info.get('marketCap')
    per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
    pbv = ticker_info.get('priceToBook')
    roe = ticker_info.get('returnOnEquity')
    de = ticker_info.get('debtToEquity')

    # ------------------------------------------------------------------
    # 4.5. BANDARMOLOGY & FOREIGN FLOW (V12)
    # ------------------------------------------------------------------
    ticker_raw = ticker_input.replace('.JK', '')
    bandar_flow_val = 0.0
    foreign_zscore_val = 0.0
    is_retail_trap = False

    # 1. Bandar Flow (Broksum)
    broksum = get_latest_broksum_for_ticker(ticker_raw)
    if broksum:
        top_buyers = broksum.get('top_buyers', [])
        top_sellers = broksum.get('top_sellers', [])
        
        # Hitung rasio akumulasi Top 3 Net Buy
        if top_buyers:
            tot_buyer_vol = sum(float(b.get("volume_lot", 0) or 0) for b in top_buyers if isinstance(b, dict))
            top3_vol = sum(float(b.get("volume_lot", 0) or 0) for b in top_buyers[:3] if isinstance(b, dict))
            
            tot_seller_vol = sum(float(s.get("volume_lot", 0) or 0) for s in top_sellers if isinstance(s, dict))
            top3_sell_vol = sum(float(s.get("volume_lot", 0) or 0) for s in top_sellers[:3] if isinstance(s, dict))
            
            tot_vol_proxy = (tot_buyer_vol + tot_seller_vol) / 2
            if tot_vol_proxy > 0:
                accum_ratio = (top3_vol - top3_sell_vol) / tot_vol_proxy
                bandar_flow_val = float(np.clip(accum_ratio * 5.0, -1.0, 1.0)) # >0.2 -> 1, <-0.2 -> -1
        
        # Retail trap: Retail mendominasi Buy, Bandar mendominasi Sell
        retail_brokers = {"YP", "PD", "XC", "KK", "NI", "CC"}
        top_3_buyer_codes = {str(b.get("broker", "")).upper() for b in top_buyers[:3] if isinstance(b, dict)}
        top_3_seller_codes = {str(s.get("broker", "")).upper() for s in top_sellers[:3] if isinstance(s, dict)}
        
        bandar_sellers = len(top_3_seller_codes - retail_brokers)
        if len(top_3_buyer_codes.intersection(retail_brokers)) >= 2 and bandar_sellers >= 2:
            is_retail_trap = True

    # 2. Foreign Flow Z-Score
    foreign_df = load_foreign_flow_history(ticker_raw, days=30)
    if foreign_df is not None and not foreign_df.empty and 'net_foreign' in foreign_df.columns:
        if len(foreign_df) >= 5:
            net_f = foreign_df['net_foreign'].values
            mean_20 = np.mean(net_f[-20:]) if len(net_f) >= 20 else np.mean(net_f)
            std_20 = np.std(net_f[-20:]) if len(net_f) >= 20 else np.std(net_f)
            if std_20 > 0:
                recent_5_mean = np.mean(net_f[-5:])
                z = (recent_5_mean - mean_20) / std_20
                foreign_zscore_val = float(np.clip(z / 2.0, -1.0, 1.0)) # z=2 -> 1.0

    # ------------------------------------------------------------------
    # 5. BERITA & SENTIMEN
    # ------------------------------------------------------------------
    news_pool = []
    translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
    translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None

    rss, _ = get_google_news_rss(f"{ticker_raw} saham")
    if rss:
        news_pool.extend(rss)
    ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
    if ysearch:
        news_pool.extend(ysearch)
    ipot, _ = get_ipot_news(f"{ticker_raw}")
    if ipot:
        news_pool.extend(ipot)

    news_pool = filter_relevant(news_pool, ticker_raw)
    seen = set()
    unique_news = []
    for n in news_pool:
        if n['title'] not in seen:
            seen.add(n['title'])
            unique_news.append(n)
        if len(unique_news) >= 5:
            break

    avg_sentiment = analyze_sentiment_weighted(unique_news, translator_en)
    headlines = [n['title'] for n in unique_news]
    sources = [n['source'] for n in unique_news]
    translated = []
    for n in unique_news:
        if TRANSLATOR_AVAILABLE and translator_id:
            try:
                translated.append(translator_id.translate(n['title']))
            except:
                translated.append("")
        else:
            translated.append("")
    sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪")

    # ------------------------------------------------------------------
    # 6. THRESHOLD & DISTRIBUSI
    # ------------------------------------------------------------------
    if is_daytrade:
        backtest_bars = 100
        if len(df) < 200:
            split_point = int(len(df) * 0.6)
            df_thresh = df.iloc[:split_point]
            backtest_window = len(df) - split_point
        else:
            df_thresh = df.iloc[-(backtest_bars * 2):-backtest_bars]
            backtest_window = backtest_bars
    else:
        split_idx = max(126, len(df) - 126)
        df_thresh = df.iloc[:split_idx]
        backtest_window = 126
    
    returns_thresh = df_thresh['Close'].pct_change().dropna()
    adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
    z_oversold_th = -1.5
    mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0

    def t_loglike(p, d):
        if p[0] <= 2 or p[2] <= 0:
            return np.inf
        return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))

    res_opt = minimize(
        t_loglike,
        [5, returns_thresh.mean(), returns_thresh.std()],
        bounds=[(2.1, 100), (-0.1, 0.1), (1e-6, None)],
        args=(returns_thresh,),
        method='L-BFGS-B'
    )
    df_est, t_loc, t_scale = res_opt.x if res_opt.success else (5, returns_thresh.mean(), returns_thresh.std())

    # ------------------------------------------------------------------
    # 7. REGIME
    # ------------------------------------------------------------------
    def get_regime_row(row):
        h, e20, e50, a, z, m = row['Close'], row['EMA20'], row['EMA50'], row['ADX'], row['ZScore'], row['Mom5D']
        if a > adx_threshold:
            if h > e20 and e20 > e50:
                return ("Strong Bullish 🚀", "RISK-ON 🔥") if (m > mom_median_th or z > z_oversold_th) else ("Bullish 📈", "RISK-ON 🔥")
            elif h < e20 and e20 < e50:
                return ("Panic Sell 🚨", "RISK-OFF 🛑") if (m < mom_median_th or z < z_oversold_th) else ("Bearish 🔻", "RISK-OFF 🛑")
            elif h > e20 and e20 < e50:
                return ("Early Recovery 🔄", "TRANSISI ⚠️")
            elif h < e20 and e20 > e50:
                return ("Distribution 📉", "TRANSISI ⚠️")
            else:
                return ("Konsolidasi Tren ↔️", "NEUTRAL ⚖️")
        else:
            if h > e20 and e20 > e50:
                return ("Bullish Accumulation 🏗️", "NEUTRAL ⚖️")
            elif h < e20 and e20 < e50:
                return ("Bearish Accumulation 🧊", "NEUTRAL ⚖️")
            elif h > e20 and e20 < e50:
                return ("Sideways Bias Naik ↗️", "NEUTRAL ⚖️")
            elif h < e20 and e20 > e50:
                return ("Sideways Bias Turun ↘️", "NEUTRAL ⚖️")
            else:
                return ("Sideways Normal ↔️", "NEUTRAL ⚖️")

    regime, ihsg_cond = get_regime_row(df.iloc[-1])
    adx = df['ADX'].iloc[-1]

    # ------------------------------------------------------------------
    # 8. BETA
    # ------------------------------------------------------------------
    beta_ihsg = 1.0
    ihsg_ret = pd.Series(dtype=float)
    try:
        if not df_ihsg.empty:
            ihsg_ret = df_ihsg['Close'].pct_change().dropna()
            common = returns.index.intersection(ihsg_ret.index)
            if len(common) > 20:
                beta_ihsg = np.cov(returns.loc[common], ihsg_ret.loc[common])[0, 1] / np.var(ihsg_ret.loc[common])
    except:
        pass

    # ------------------------------------------------------------------
    # 9. ATR & RSI
    # ------------------------------------------------------------------
    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr14_val = df['TR'].rolling(14).mean().iloc[-1]
    atr_pct = (atr14_val / harga_terakhir_asli) * 100

    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    if is_daytrade:
        bars_remaining = hitung_bars_remaining(now_jkt, actual_interval, bars_per_day_map)
    else:
        bars_remaining = None

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    if avg_loss is None or avg_loss == 0:
        rsi14 = 100.0
    else:
        rsi14 = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

    # ------------------------------------------------------------------
    # 10. PIVOT
    # ------------------------------------------------------------------
    if is_daytrade:
        today_jkt = datetime.now(pytz.timezone("Asia/Jakarta")).date()
        if not df_daily.empty:
            df_daily_filtered = df_daily[df_daily.index.date < today_jkt]
            if not df_daily_filtered.empty:
                prev_day = df_daily_filtered.iloc[-1]
                hi_daily = float(prev_day['High'])
                lo_daily = float(prev_day['Low'])
                cl_daily = float(prev_day['Close'])
            else:
                prev_day = None
                for i in range(1, min(len(df_daily), 6)):
                    row = df_daily.iloc[-i]
                    h_val = float(row['High'])
                    l_val = float(row['Low'])
                    c_val = float(row['Close'])
                    if h_val != l_val and h_val > 0 and l_val > 0:
                        prev_day = row
                        hi_daily, lo_daily, cl_daily = h_val, l_val, c_val
                        break
                if prev_day is None:
                    last = df_daily.iloc[-1]
                    hi_daily = float(last['High'])
                    lo_daily = float(last['Low'])
                    cl_daily = float(last['Close'])
        else:
            hi_daily = float(df['High'].iloc[-1])
            lo_daily = float(df['Low'].iloc[-1])
            cl_daily = float(df['Close'].iloc[-1])

        if hi_daily != lo_daily:
            pp = (hi_daily + lo_daily + cl_daily) / 3
            r1 = 2 * pp - lo_daily
            s1 = 2 * pp - hi_daily
            r2 = pp + (hi_daily - lo_daily)
            s2 = pp - (hi_daily - lo_daily)
        else:
            pp = r1 = s1 = r2 = s2 = cl_daily
    else:
        hi = lo = cl = None
        for i in range(1, min(6, len(df))):
            row = df.iloc[-i]
            h_val = float(row['High']); l_val = float(row['Low']); c_val = float(row['Close'])
            if h_val != l_val and h_val > 0 and l_val > 0:
                hi, lo, cl = h_val, l_val, c_val
                break
        if hi is None:
            hi = float(df['High'].iloc[-1]); lo = float(df['Low'].iloc[-1]); cl = float(df['Close'].iloc[-1])
        if hi == lo:
            pp = r1 = s1 = r2 = s2 = cl
        else:
            pp = (hi + lo + cl) / 3
            r1 = 2 * pp - lo; s1 = 2 * pp - hi
            r2 = pp + (hi - lo); s2 = pp - (hi - lo)

    # ------------------------------------------------------------------
    # 11. V12 ADAPTIVE SIGNAL
    # ------------------------------------------------------------------
    adaptive_w = get_adaptive_weights(ticker_raw, regime, v12_mem=v12_mem)
    coppock_val, coppock_prev = coppock_curve(df['Close'].values)
    factor_signals = {
        "Momentum": (df['Mom5D'].iloc[-1] - mom_median_th) / max(0.1, df['Mom5D'].std()),
        "AI_Senti": avg_sentiment,
        "MeanRev": -df['ZScore'].iloc[-1] / 3.0,
        "Beta_IHSG": beta_ihsg * (ihsg_ret.iloc[-1] if not ihsg_ret.empty else 0.0),
        "Coppock": coppock_val / 10.0,
        "OFI": df['OFI_Enhanced'].iloc[-1] / 5.0,  # Enhanced OFI dengan shadow-weighted volume
        "Bandar_Flow": bandar_flow_val,
        "Foreign_ZScore": foreign_zscore_val
    }
    norm_signals = {k: max(-1.0, min(1.0, v)) for k, v in factor_signals.items()}
    total_score = sum(norm_signals[k] * adaptive_w.get(k, 0.15) for k in FACTOR_KEYS)
    
    historical_scores = []
    for i in range(max(20, len(df)-60), len(df)):
        row_signals = {
            "Momentum": float(np.clip((df['Mom5D'].iloc[i] - mom_median_th) / max(0.1, df['Mom5D'].std()), -1, 1)),
            "AI_Senti": avg_sentiment,  # konstanta (fine, ini lagging)
            "MeanRev": float(np.clip(-df['ZScore'].iloc[i] / 3.0, -1, 1)),
            "Beta_IHSG": 0.0,            # skip, butuh return historis
            "Coppock": 0.0,              # skip, expensive
            "OFI": float(np.clip(df['OFI_Enhanced'].iloc[i] / 5.0, -1, 1)),
            "Bandar_Flow": bandar_flow_val,       # gunakan nilai konstan terbaru
            "Foreign_ZScore": foreign_zscore_val  # gunakan nilai konstan terbaru
        }
        s = sum(row_signals[k] * adaptive_w.get(k, 0.15) for k in FACTOR_KEYS)
        historical_scores.append(s)
    
    score_std = np.std(historical_scores) if len(historical_scores) > 5 else 0.15
    score_std = max(0.10, min(0.35, score_std))   # clamp reasonable range
    
    # Threshold = ±1.0σ untuk STRONG, ±0.4σ untuk BUY
    th_strong = score_std * 1.0
    th_buy    = score_std * 0.4
    th_hold   = -score_std * 0.4
    
    if total_score > th_buy:
        if is_retail_trap:
            signal = "⏸️ HOLD / WAIT (Retail Trap Warning)"
            total_score = th_hold + 0.01
        elif is_marking_close or is_no_demand:
            # VSA Penalty: Sinyal beli tapi structure harga palsu
            signal = "⏸️ HOLD / WAIT (Fake Breakout / Marking Close)"
            total_score = th_hold + 0.01
        elif not is_mtf_bullish:
            # Counter-Trend Risk Penalty
            if total_score > th_strong:
                signal = "⚡ BUY (TACTICAL) [Counter-Trend]"
            else:
                signal = "⏸️ HOLD / WAIT (Counter-Trend Risk)"
                total_score = th_hold + 0.01
        elif total_score > th_strong:
            signal = "🔥 STRONG BUY"
        else:
            signal = "⚡ BUY (TACTICAL)"
    elif total_score > th_hold:
        signal = "⏸️ HOLD / WAIT"
        if is_stopping_volume:
            signal += " 🟢 (Potential Reversal — Absorption Terdeteksi)"
    else:
        signal = "🚨 AVOID"
        if is_stopping_volume:
            signal += " 🟢 (Watchlist — Absorption Terdeteksi)"

    # ------------------------------------------------------------------
    # 12. ENTRY ZONE
    # ------------------------------------------------------------------
    if s1 >= harga_terakhir * 0.98:
        entry_low = s1
    else:
        entry_low = harga_terakhir * (1 - atr_pct / 100)

    if "STRONG BUY" in signal:
        entry_high = harga_terakhir
    else:
        entry_high = harga_terakhir * (1 - 0.3 * atr_pct / 100)

    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    min_entry_width = 0.5 * atr14_val
    if (entry_high - entry_low) < min_entry_width:
        entry_low = max(0, entry_high - min_entry_width)
        entry_high = entry_low + min_entry_width
        entry_high = min(entry_high, harga_terakhir)

    # Baca entry_error dari v12_mem (parameter thread-safe), fallback ke session_state
    if v12_mem is not None:
        mem_for_entry = v12_mem.get(ticker_raw, {})
    else:
        mem_for_entry = st.session_state.v12_memory.get(ticker_raw, {})
    entry_error = mem_for_entry.get('entry_error_ema', 0.0)
    if entry_error > 0:
        entry_low += entry_error * 0.2
        entry_high += entry_error * 0.2

    entry_high = min(entry_high, harga_terakhir)
    entry_low = min(entry_low, entry_high)

    # ── Guard: kalau entry_low/high NaN, fallback ke harga_terakhir ──
    if (entry_low is None) or (isinstance(entry_low, float) and math.isnan(entry_low)) or entry_low <= 0:
        entry_low = harga_terakhir * 0.98 if harga_terakhir > 0 else 1
    if (entry_high is None) or (isinstance(entry_high, float) and math.isnan(entry_high)) or entry_high <= 0:
        entry_high = harga_terakhir if harga_terakhir > 0 else 1
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    entry_low_f = fraksi_bei(entry_low)
    entry_high_f = fraksi_bei(entry_high)
    entry_zone_f = f"Rp {entry_low_f:,.0f} - Rp {entry_high_f:,.0f}"

    # ------------------------------------------------------------------
    # 13. SL & TP
    # ------------------------------------------------------------------
    sl_mult = 1.0
    if adx > 30 and 30 < rsi14 < 70:
        sl_mult = 0.75
    elif adx < 20:
        sl_mult = 1.25
    if rsi14 > 70 or rsi14 < 30:
        sl_mult = 1.5

    tp_mult_low = 1.5
    tp_mult_high = 2.5
    if adx > 30 and 30 < rsi14 < 70:
        tp_mult_low, tp_mult_high = 2.0, 3.0
    elif adx < 20:
        tp_mult_low, tp_mult_high = 1.2, 1.8

    if is_daytrade:
        base_sl_dist = harga_terakhir * 0.04 * sl_mult
        min_ticks_dist = 2 * fraksi_step(entry_low)
        sl_dist = max(min_ticks_dist, base_sl_dist)
        sl_harga = entry_low - sl_dist
    else:
        sl_harga = entry_low - sl_mult * atr14_val

    sl_harga = fraksi_bei(sl_harga)
    step = fraksi_step(entry_low)
    if sl_harga >= entry_low:
        sl_harga = fraksi_bei(entry_low - 2 * step)
    if sl_harga <= 0:
        sl_harga = fraksi_bei(harga_terakhir * 0.95)
    sl_pct = (harga_terakhir - sl_harga) / harga_terakhir * 100

    if is_daytrade:
        # --- PERHITUNGAN TP DAYTRADE BERBASIS ATR INTRADAY & FAKTOR FEE BROKER ---
        # 1. Target ATR Intraday (5m)
        tp_low_raw = entry_low + (tp_mult_low * atr14_val)
        tp_high_raw = entry_low + (tp_mult_high * atr14_val)

        # 2. Safety Floor untuk Memastikan Cover Fee Broker (Beli + Jual) + Target Net Profit Margin (+0.6% net)
        total_fee_pct = (fee_beli_pct + fee_jual_pct) / 100.0
        min_net_margin = 0.0060
        fee_floor = entry_low * (1.0 + total_fee_pct + min_net_margin)

        # Gunakan nilai terbesar antara ATR Intraday & Fee Floor
        tp_low_raw = max(tp_low_raw, fee_floor)
        tp_high_raw = max(tp_high_raw, tp_low_raw + (2 * fraksi_step(tp_low_raw)))

        # 3. Pastikan minimal 2 tick di atas entry_low dan harga_terakhir agar komisi tercover penuh
        min_tp_low_ticks = max(entry_low + (2 * step), harga_terakhir + (2 * fraksi_step(harga_terakhir)))
        if tp_low_raw < min_tp_low_ticks:
            tp_low_raw = min_tp_low_ticks

        tp_low = fraksi_bei(tp_low_raw)
        tp_high = fraksi_bei(tp_high_raw)
        if tp_low <= entry_low:
            tp_low = fraksi_bei(entry_low + 2 * step)
        if tp_high <= tp_low:
            tp_high = fraksi_bei(tp_low + 2 * fraksi_step(tp_low))
    else:
        # --- PERHITUNGAN TP SWING (RESISTANCE HARIAN / PIVOT R1 R2) ---
        if r1 > harga_terakhir:
            tp_low = r1
        else:
            tp_low = harga_terakhir + tp_mult_low * atr14_val
        if r2 > harga_terakhir:
            tp_high = r2
        else:
            tp_high = harga_terakhir + tp_mult_high * atr14_val
        if tp_low > tp_high:
            tp_low, tp_high = tp_high, tp_low

    tp_pct_low = (tp_low - harga_terakhir) / harga_terakhir * 100
    tp_pct_high = (tp_high - harga_terakhir) / harga_terakhir * 100

    risk = harga_terakhir - sl_harga
    reward = tp_low - harga_terakhir
    rrr = reward / risk if risk > 0 else 0
    if rrr >= 2.0:
        rrr_status = "Sangat Baik (≥ 2.0) 🟢"
    elif rrr >= 1.5:
        rrr_status = "Baik (1.5 - 2.0) 🟢"
    elif rrr >= 1.0:
        rrr_status = "Cukup (1.0 - 1.5) 🟡"
    else:
        rrr_status = "Buruk (< 1.0) 🔴"

    # Dynamic Dip Target untuk RRR Ideal 1:2.0
    if tp_low > sl_harga:
        entry_ideal_raw = (tp_low + 2 * sl_harga) / 3.0
        entry_ideal_f = fraksi_bei(entry_ideal_raw)
        entry_ideal_f = min(entry_ideal_f, harga_terakhir)
        entry_ideal_f = max(entry_ideal_f, fraksi_bei(sl_harga + 2 * fraksi_step(sl_harga)))
    else:
        entry_ideal_f = fraksi_bei(entry_low)


    # Breakout
    if is_daytrade:
        bars_per_day = bars_per_day_map.get(actual_interval, 54)
        if len(df) >= bars_per_day:
            res20 = float(df['High'].iloc[-bars_per_day:-1].max())
            breakout_label = f"Breakout Sesi Sebelumnya ({bars_per_day} bar)"
        else:
            res20 = float(df['High'].max())
            breakout_label = "Breakout N-Bar"
    else:
        if len(df) >= 21:
            res20 = float(df['High'].iloc[-21:-1].max())
        else:
            res20 = float(df['High'].max())
        breakout_label = "Breakout 20 Hari"
    breakout = f"YES (🔥)" if harga_terakhir > res20 else "NO"

    # ------------------------------------------------------------------
    # 14. BACKTEST — V12-Synced Engine
    # Menggunakan factor_signals yang IDENTIK dengan Section 11 (total_score)
    # agar Win Rate & Profit Factor benar-benar mencerminkan performa model V12.
    # + Biaya riil: fee beli 0.15%, jual 0.25%, slippage 1 fraksi BEI.
    # ------------------------------------------------------------------
    _fee_beli  = fee_beli_pct  / 100.0   # default 0.0015
    _fee_jual  = fee_jual_pct  / 100.0   # default 0.0025

    def _v12_score_series(dataframe, _adaptive_w, _mom_th, _avg_sent):
        """
        Hitung V12 total_score per baris menggunakan formula identik section 11.
        Dikembalikan sebagai pd.Series (float).
        Beta_IHSG di-skip untuk efisiensi (seperti historical_scores di section 11).
        """
        mom_std = max(0.1, dataframe['Mom5D'].std())
        s_mom  = ((dataframe['Mom5D'] - _mom_th) / mom_std).clip(-1, 1)
        s_mr   = (-dataframe['ZScore'] / 3.0).clip(-1, 1)
        s_ofi  = (dataframe['OFI_Enhanced'] / 5.0).clip(-1, 1) if 'OFI_Enhanced' in dataframe.columns else pd.Series(0.0, index=dataframe.index)
        s_sent = float(np.clip(_avg_sent, -1, 1))
        # Coppock: skip per-baris karena mahal; gunakan nilai terakhir (konstan)
        s_copp = float(np.clip(coppock_val / 10.0, -1, 1))

        score = (
            s_mom  * _adaptive_w.get("Momentum",   0.23) +
            s_mr   * _adaptive_w.get("MeanRev",    0.15) +
            s_ofi  * _adaptive_w.get("OFI",        0.12) +
            s_sent * _adaptive_w.get("AI_Senti",   0.17) +
            s_copp * _adaptive_w.get("Coppock",    0.18) +
            bandar_flow_val * _adaptive_w.get("Bandar_Flow", 0.15) +
            foreign_zscore_val * _adaptive_w.get("Foreign_ZScore", 0.10)
            # Beta_IHSG skip (butuh return historis per-baris IHSG)
        )
        return score

    df['V12_Score'] = _v12_score_series(df, adaptive_w, mom_median_th, avg_sentiment)

    # Threshold dinamis berbasis std historical (identik section 11)
    bt_score_std = max(0.10, min(0.35, df['V12_Score'].std()))
    bt_th_strong = bt_score_std * 1.0
    bt_th_buy    = bt_score_std * 0.4
    bt_th_hold   = -bt_score_std * 0.4

    def _v12_signal_from_score(sc):
        if sc > bt_th_strong: return "🔥 STRONG BUY"
        if sc > bt_th_buy:    return "⚡ BUY (TACTICAL)"
        if sc > bt_th_hold:   return "⏸️ HOLD / WAIT"
        return "🚨 AVOID"

    df['Signal'] = df['V12_Score'].apply(_v12_signal_from_score)
    df_back = df.iloc[-backtest_window:].copy()

    if len(df_back) == 0:
        st.warning(f"⚠️ df_back kosong untuk {ticker_raw} (mode: {'DT' if is_daytrade else 'SW'})")
        return None

    trades, daily_returns = [], []
    in_position, entry_price_net = False, 0.0
    for i in range(len(df_back)):
        curr_sig   = df_back['Signal'].iloc[i]
        curr_close = float(df_back['Close'].iloc[i])
        prev_close = float(df_back['Close'].iloc[i - 1]) if i > 0 else curr_close

        if in_position:
            daily_returns.append((curr_close - prev_close) / prev_close if prev_close else 0)
            if "AVOID" in curr_sig or i == len(df_back) - 1:
                # Harga jual dikurangi 1 fraksi slippage + fee jual
                slip_jual    = fraksi_step(curr_close)
                exit_price   = max(0, curr_close - slip_jual)
                net_exit     = exit_price * (1 - _fee_jual)
                net_return   = (net_exit - entry_price_net) / entry_price_net if entry_price_net > 0 else 0
                trades.append(net_return)
                in_position  = False
        else:
            daily_returns.append(0.0)
            if "BUY" in curr_sig:
                # Harga beli ditambah 1 fraksi slippage + fee beli
                slip_beli        = fraksi_step(curr_close)
                entry_price_net  = (curr_close + slip_beli) * (1 + _fee_beli)
                in_position      = True

    if trades:
        win_bt = sum(1 for r in trades if r > 0) / len(trades)
        loss_trades = [r for r in trades if r < 0]
        profit_trades = [r for r in trades if r > 0]
        pf_bt = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else np.inf
        avg_bt = np.mean(trades)
        equity = np.cumprod([1 + r for r in trades])
        max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100) if len(equity) else 0
        daily_ret = np.array(daily_returns)
        if is_daytrade:
            bars_per_day = bars_per_day_map.get(actual_interval, 54)
            annual_factor = np.sqrt(bars_per_day * 252)
        else:
            annual_factor = np.sqrt(252)
        sharpe_bt = (daily_ret.mean() / daily_ret.std()) * annual_factor if daily_ret.std() else 0
        trades_bt = len(trades)
    else:
        win_bt = pf_bt = avg_bt = max_dd_bt = sharpe_bt = trades_bt = 0

    # ------------------------------------------------------------------
    # 15. KELLY & DRAWDOWN
    # ------------------------------------------------------------------
    roll_max_th = df_thresh['Close'].cummax()
    drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
    max_dd = float(drawdown_th.min() * 100)
    max_dd_30 = float(drawdown_th.tail(30).min() * 100) if len(drawdown_th) >= 30 else max_dd

    if trades_bt >= 2:
        win_r = win_bt
        avg_g = np.mean(profit_trades) if profit_trades else 0.01
        avg_l = abs(np.mean(loss_trades)) if loss_trades else 0.01
    else:
        win_r = len(returns_thresh[returns_thresh > 0]) / len(returns_thresh)
        avg_g = returns_thresh[returns_thresh > 0].mean() if win_r > 0 else 0.01
        avg_l = abs(returns_thresh[returns_thresh < 0].mean()) if len(returns_thresh[returns_thresh < 0]) else 0.01

    wl = avg_g / avg_l if avg_l else 1
    kelly_raw = win_r - (1 - win_r) / wl
    ret_skew = float(skew(returns_thresh))
    ret_kurt = float(kurtosis(returns_thresh, fisher=True))
    kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
    kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1) * kurt_penalty))
    target_risk_pct = 1.5
    risk_adjusted_alloc = min(kelly_adj * 100, (target_risk_pct / sl_pct) * 100) if sl_pct > 0 else kelly_adj * 100


    # ------------------------------------------------------------------
    # 16. MONTE CARLO — Regime-Switching (GBM untuk trending, OU untuk sideways)
    #
    # Problem lama: OU murni memaksa proyeksi kembali ke mean 20 hari untuk
    # SEMUA kondisi, termasuk saham yang baru saja breakout ATH / super-trend.
    # Akibatnya prob_bull dan estimasi harga besok terlalu pesimistis.
    #
    # Solusi:
    #  • BULLISH / STRONG BUY / Breakout  → GBM drift positif (momentum-based)
    #  • BEARISH / PANIC / AVOID          → GBM drift negatif
    #  • SIDEWAYS / CONSOLIDATION / HOLD  → OU mean-reverting (seperti sebelumnya)
    # ------------------------------------------------------------------
    if is_daytrade:
        n_sim   = 2000
        n_steps = max(1, bars_remaining)
    else:
        n_sim   = 2000
        n_steps = 30

    latest_vol = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
    scale_corrected = latest_vol / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol

    # ── Deteksi regime Monte Carlo ──
    _trending_up_regimes   = {"STABLE BULLISH", "VOLATILE UPTREND"}
    _trending_down_regimes = {"HIGH-STRESS PANIC"}
    _sideways_regimes      = {"SIDEWAYS / CONSOLIDATION", "BEARISH ACCUMULATION"}

    _is_mc_bullish = (
        regime in _trending_up_regimes or
        "STRONG BUY" in signal or
        (breakout == "YES (🔥)" and "BUY" in signal)
    )
    _is_mc_bearish = (
        regime in _trending_down_regimes or
        "AVOID" in signal
    )
    # Default: sideways / mean-reverting

    paths       = np.zeros((n_steps, n_sim))
    current_log = np.ones(n_sim) * np.log(harga_terakhir)

    if _is_mc_bullish:
        # ── GBM Bullish: drift berbasis momentum 5D harian (annualized → per-step) ──
        mom5d_raw  = df['Mom5D'].iloc[-1] if 'Mom5D' in df.columns else 0.0
        # Konversi: Mom5D adalah pct change 5D, bagi 5 → per-hari, lalu clamp +0.1%~+1%
        drift_daily = float(np.clip(mom5d_raw / 500.0, 0.001, 0.010))
        mc_regime_label = "GBM Bullish 🚀"
        for step in range(n_steps):
            inov        = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            current_log = current_log + drift_daily + inov
            paths[step] = np.exp(current_log)

    elif _is_mc_bearish:
        # ── GBM Bearish: drift negatif berbasis momentum ──
        mom5d_raw   = df['Mom5D'].iloc[-1] if 'Mom5D' in df.columns else 0.0
        drift_daily = float(np.clip(mom5d_raw / 500.0, -0.010, -0.001))
        mc_regime_label = "GBM Bearish 🔻"
        for step in range(n_steps):
            inov        = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            current_log = current_log + drift_daily + inov
            paths[step] = np.exp(current_log)

    else:
        # ── OU Mean-Reverting: untuk sideways / konsolidasi ──
        theta_ou           = estimate_theta_ou(df['Close'])
        locked_log_mean20  = np.log(df['Close']).tail(20).mean()
        mc_regime_label    = "OU Mean-Reverting ↔️"
        for step in range(n_steps):
            inov        = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            current_log = current_log + theta_ou * (locked_log_mean20 - current_log) + inov
            paths[step] = np.exp(current_log)

    final_prices = paths[-1, :]
    est_besok = float(np.median(final_prices))
    if "STRONG BUY" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 75))
    elif "BUY" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 65))
    elif "HOLD" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 50))
    else:
        est_besok_sinyal = float(np.percentile(final_prices, 35))

    low_est, up_est = float(np.percentile(final_prices, 25)), float(np.percentile(final_prices, 75))
    prob_bull = (final_prices > harga_terakhir).mean() * 100
    hit_tp    = (np.any(paths >= r1, axis=0).sum() / n_sim) * 100
    hit_sl    = (np.any(paths <= s2, axis=0).sum() / n_sim) * 100

    if is_daytrade:
        estimasi_label = "Estimasi Sesi Berikutnya"
        prob_label = "Prob Naik Sesi Berikutnya"
    else:
        estimasi_label = "Estimasi Besok"
        prob_label = "Prob Naik Besok"

    # ------------------------------------------------------------------
    # 17. METRIK TAMBAHAN
    # ------------------------------------------------------------------
    if "STRONG BUY" in signal:
        signal_score = 0.7 + (prob_bull / 200)
    elif "BUY" in signal:
        signal_score = 0.4 + (prob_bull / 200)
    elif "HOLD" in signal:
        signal_score = 0.2 + (prob_bull / 300)
    else:
        signal_score = max(0, (prob_bull - 30) / 100)
    signal_score = min(1.0, max(0.0, signal_score))
    confidence = min(0.99, 0.5 + (signal_score * 0.5) + (win_bt - 0.5) * 0.1)
    if confidence is None or np.isnan(confidence):
        confidence = 0.5

    trend_consistency = np.mean([
        1 if (df['Close'].iloc[-i] > df['Close'].iloc[-i-1]) == (df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]) else 0
        for i in range(1, 11)
    ]) * 100
    if np.isnan(trend_consistency):
        trend_consistency = 50.0

    avg_vol_5 = df['Volume'].iloc[-5:].mean()
    avg_vol_20 = df['Volume'].iloc[-20:].mean()
    if avg_vol_20 > 0:
        vol_surge_pct = ((avg_vol_5 / avg_vol_20) - 1) * 100
    else:
        vol_surge_pct = 0.0

    avg_value = (df['Volume'].iloc[-5:] * df['Close'].iloc[-5:]).mean()
    if np.isnan(avg_value):
        avg_value = 0.0
    if avg_value >= 1e9:
        likuiditas_str = f"Rp {avg_value/1e9:.2f} M"
    elif avg_value >= 1e6:
        likuiditas_str = f"Rp {avg_value/1e6:.0f} Jt"
    elif avg_value >= 1e3:
        likuiditas_str = f"Rp {avg_value/1e3:.0f} rb"
    else:
        likuiditas_str = f"Rp {avg_value:,.0f}"

    if rsi14 > 70:
        rsi_status = "Overbought"
    elif rsi14 < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Normal"

    zscore_val = df['ZScore'].iloc[-1]
    if pd.isna(zscore_val):
        zscore_val = 0.0
    if zscore_val > 2:
        zs_status = "Overbought"
    elif zscore_val < -2:
        zs_status = "Oversold"
    else:
        zs_status = "Normal"

    if vol_surge_pct > 50:
        vs_status = "Tinggi"
    elif vol_surge_pct < -30:
        vs_status = "Rendah"
    else:
        vs_status = "Normal"

    coppock_rising = coppock_val > coppock_prev
    coppock_turning_up = coppock_rising and coppock_prev <= 0
    if coppock_turning_up:
        coppock_status = "Turning Up"
    elif coppock_rising:
        coppock_status = "Rising"
    else:
        coppock_status = "Falling"

    est_besok_f = fraksi_bei(est_besok)
    est_besok_sinyal_f = fraksi_bei(est_besok_sinyal)
    low_est_f = fraksi_bei(low_est)
    up_est_f = fraksi_bei(up_est)
    tp_low_f = fraksi_bei(tp_low)
    tp_high_f = fraksi_bei(tp_high)
    sl_harga_f = fraksi_bei(sl_harga)

    # ------------------------------------------------------------------
    # 18. RINGKASAN UNTUK RIWAYAT
    # ------------------------------------------------------------------
    ringkasan = {
        "Waktu": datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M"),
        "Saham": ticker_raw,
        "Harga": f"{harga_terakhir:,.0f}",
        "Sinyal": signal,
        "Estimasi": f"{est_besok:,.0f}",
        "Estimasi_Netral": f"Rp {est_besok_f:,.0f}",
        "Estimasi_Sinyal": f"Rp {est_besok_sinyal_f:,.0f}",
        "Prob Naik": f"{prob_bull:.1f}%",
        "RRR": f"{rrr:.2f}",
        "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
        "Rezim": regime,
        "TP%": f"{tp_pct_low:.1f}% - {tp_pct_high:.1f}%",
        "SL%": f"{sl_pct:.1f}%",
        "AI_Insight": "",
        "Score": f"{signal_score:.3f}",
        "Confidence": f"{confidence:.0%}",
        "Coppock": coppock_status,
        "Est_Return": f"{((est_besok - harga_terakhir) / harga_terakhir * 100):+.2f}%",
        "Est_Return_Sinyal": f"{((est_besok_sinyal - harga_terakhir) / harga_terakhir * 100):+.2f}%",
        "TP_Harga": f"{tp_low_f:,.0f} - {tp_high_f:,.0f}",
        "TP_Range": f"Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}",
        "SL_Harga": f"{sl_harga_f:,.0f}",
        "Likuiditas": likuiditas_str,
        "RSI": f"{rsi14:.1f}",
        "RSI_Status": rsi_status,
        "Vol_Surge": f"{vol_surge_pct:+.0f}%",
        "VS_Status": vs_status,
        "ZScore": f"{zscore_val:.2f}",
        "ZS_Status": zs_status,
        "Trend_Consistency": f"{trend_consistency:.0f}%",
        "Beta": f"{beta_ihsg:.2f}",
        "Momentum": f"{df['Mom5D'].iloc[-1]:.2f}%",
        "Entry_Zone": entry_zone_f,
        "Entry_Ideal_RRR2": f"Rp {entry_ideal_f:,.0f}",
        "Risk_Adjusted_Alloc": f"{risk_adjusted_alloc:.1f}%",
        "Gaya": "DT" if is_daytrade else "SW",
        "Status_Posisi": "Sudah Beli" if sudah_beli else "Belum",
        "Harga_Beli": f"{harga_beli_float:,.0f}" if harga_beli_float else "",
        "Floating_PL": f"{floating_pl_pct:+.2f}%" if floating_pl_pct is not None else ""
    }

    # ------------------------------------------------------------------
    # 19. KUMPULKAN RESULT
    # ------------------------------------------------------------------
    result = {
        "df": df,
        "df_back": df_back,
        "harga_terakhir": harga_terakhir,
        "signal": signal,
        "entry_zone_f": entry_zone_f,
        "entry_ideal_f": entry_ideal_f,
        "risk_adjusted_alloc": risk_adjusted_alloc,
        "sl_harga_f": sl_harga_f,

        "tp_low_f": tp_low_f,
        "tp_high_f": tp_high_f,
        "rrr": rrr,
        "rrr_status": rrr_status,
        "prob_bull": prob_bull,
        "signal_score": signal_score,
        "confidence": confidence,
        "est_besok_f": est_besok_f,
        "est_besok_sinyal_f": est_besok_sinyal_f,
        "low_est_f": low_est_f,
        "up_est_f": up_est_f,
        "tp_pct_low": tp_pct_low,
        "tp_pct_high": tp_pct_high,
        "sl_pct": sl_pct,
        "adx": adx,
        "rsi14": rsi14,
        "atr_pct": atr_pct,
        "avg_sentiment": avg_sentiment,
        "sentimen_status": sentimen_status,
        "headlines": headlines,
        "sources": sources,
        "translated": translated,
        "regime": regime,
        "ihsg_cond": ihsg_cond,
        "coppock_val": coppock_val,
        "coppock_prev": coppock_prev,
        "coppock_turning_up": coppock_turning_up,
        "beta_ihsg": beta_ihsg,
        "win_bt": win_bt,
        "pf_bt": pf_bt,
        "avg_bt": avg_bt,
        "max_dd_bt": max_dd_bt,
        "sharpe_bt": sharpe_bt,
        "trades_bt": trades_bt,
        "kelly_adj": kelly_adj,
        "max_dd": max_dd,
        "max_dd_30": max_dd_30,
        "breakout": breakout,
        "breakout_label": breakout_label,
        "vwap_now": vwap_now,
        "vwap_bias": vwap_bias,
        "r1": r1, "r2": r2, "s1": s1, "s2": s2, "pp": pp,
        "mc": mc, "per": per, "pbv": pbv, "roe": roe, "de": de,
        "norm_signals": norm_signals,   # untuk simpan prediksi
        "ringkasan": ringkasan,
        "is_daytrade": is_daytrade,
        "mode": "daytrade" if is_daytrade else "swing",
        "actual_interval": actual_interval,
        "harga_terakhir_asli": harga_terakhir_asli,
        "floating_pl_pct": floating_pl_pct,
        "harga_beli_float": harga_beli_float,
        "sudah_beli": sudah_beli,
        "ticker_raw": ticker_raw,
        "bandar_flow_val": bandar_flow_val,
        "foreign_zscore_val": foreign_zscore_val,
        "is_retail_trap": is_retail_trap,
        "is_mtf_bullish": is_mtf_bullish,
        "mtf_status_text": mtf_status_text,
        "is_marking_close": is_marking_close,
        "is_no_demand": is_no_demand,
        "is_stopping_volume": is_stopping_volume,
        "vsa_status_text": vsa_status_text
    }
        # Tambahan untuk UI
    result["ticker_info"] = ticker_info
    result["adx_threshold"] = adx_threshold
    result["hit_tp"] = hit_tp
    result["hit_sl"] = hit_sl
    result["estimasi_label"] = estimasi_label
    result["prob_label"] = prob_label
    result["backtest_window"] = backtest_window
    result["ofi_now"] = df['OFI_Enhanced'].iloc[-1]  # Enhanced OFI dengan shadow weighting
    result["adaptive_w"] = adaptive_w
    result["mc_regime_label"] = mc_regime_label   # label model MC: GBM Bullish/Bearish / OU
    result["returns"] = returns
    result["mom_median_th"] = mom_median_th
    result["coppock_rising"] = coppock_rising
    result["coppock_turning_up"] = coppock_turning_up
    result["coppock_status"] = coppock_status
    result["avg_sentiment"] = avg_sentiment
    result["norm_signals"] = norm_signals
    result["entry_low_f"] = entry_low_f
    result["entry_high_f"] = entry_high_f
    result["ticker_raw"] = ticker_raw
    result["harga_terakhir_asli"] = harga_terakhir_asli
    result["floating_pl_pct"] = floating_pl_pct
    result["sudah_beli"] = sudah_beli
    result["harga_beli_float"] = harga_beli_float
    result["df_est"] = df_est
    return result
def display_analysis_result(res):
    # ===== AMBIL SEMUA VARIABEL DARI RES =====
    df = res['df']
    df_back = res['df_back']
    harga_terakhir = res['harga_terakhir']
    signal = res['signal']
    entry_zone_f = res['entry_zone_f']
    entry_ideal_f = res.get('entry_ideal_f', fraksi_bei(harga_terakhir))
    risk_adjusted_alloc = res.get('risk_adjusted_alloc', res['kelly_adj'] * 100)
    sl_harga_f = res['sl_harga_f']

    tp_low_f = res['tp_low_f']
    tp_high_f = res['tp_high_f']
    rrr = res['rrr']
    rrr_status = res['rrr_status']
    prob_bull = res['prob_bull']
    est_besok_f = res['est_besok_f']
    est_besok_sinyal_f = res['est_besok_sinyal_f']
    low_est_f = res['low_est_f']
    up_est_f = res['up_est_f']
    tp_pct_low = res['tp_pct_low']
    tp_pct_high = res['tp_pct_high']
    sl_pct = res['sl_pct']
    adx = res['adx']
    rsi14 = res['rsi14']
    atr_pct = res['atr_pct']
    avg_sentiment = res['avg_sentiment']
    sentimen_status = res['sentimen_status']
    headlines = res['headlines']
    sources = res['sources']
    translated = res['translated']
    regime = res['regime']
    ihsg_cond = res['ihsg_cond']
    coppock_val = res['coppock_val']
    coppock_prev = res['coppock_prev']
    coppock_turning_up = res['coppock_turning_up']
    coppock_rising = res['coppock_rising']
    coppock_status = res['coppock_status']
    beta_ihsg = res['beta_ihsg']
    win_bt = res['win_bt']
    pf_bt = res['pf_bt']
    avg_bt = res['avg_bt']
    max_dd_bt = res['max_dd_bt']
    sharpe_bt = res['sharpe_bt']
    trades_bt = res['trades_bt']
    kelly_adj = res['kelly_adj']
    max_dd = res['max_dd']
    max_dd_30 = res['max_dd_30']
    breakout = res['breakout']
    breakout_label = res['breakout_label']
    vwap_now = res['vwap_now']
    vwap_bias = res['vwap_bias']
    r1 = res['r1']
    r2 = res['r2']
    s1 = res['s1']
    s2 = res['s2']
    pp = res['pp']
    mc = res['mc']
    per = res['per']
    pbv = res['pbv']
    roe = res['roe']
    de = res['de']
    ticker_info = res['ticker_info']
    adx_threshold = res['adx_threshold']
    hit_tp = res['hit_tp']
    hit_sl = res['hit_sl']
    estimasi_label = res['estimasi_label']
    prob_label = res['prob_label']
    backtest_window = res['backtest_window']
    ofi_now = res['ofi_now']
    is_daytrade = res['is_daytrade']
    floating_pl_pct = res['floating_pl_pct']
    sudah_beli = res['sudah_beli']
    ticker_raw = res['ticker_raw']

    # === Tambahan untuk V12 Adaptive & AI Insight ===
    adaptive_w = res['adaptive_w']
    returns = res['returns']
    mom_median_th = res['mom_median_th']
    harga_beli_float = res['harga_beli_float']

    # ===== TAMPILAN UTAMA =====
    st.title("📊 Quant & Risk Engine Pro")
    st.write("Algoritma kuantitatif + Berita + Backtest + AI + Grafik Interaktif + Fundamental")
    st.success(f"✅ Analisis Berhasil: {ticker_raw} | Closing Price: Rp {harga_terakhir:,.0f}".replace(",", "."))

    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    st.caption(f"⏱️ **Waktu Analisis:** {now_jkt.strftime('%d %B %Y, %H:%M:%S WIB')}")
    waktu_str = now_jkt.strftime('%d %B %Y, %H:%M WIB')

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Sinyal Eksekusi",
        signal,
        delta=f"per {now_jkt.strftime('%d/%m %H:%M')} WIB",
        delta_color="off"
    )
    with col2:
        st.metric(
            label=f"{estimasi_label} (Netral)",
            value=f"Rp {est_besok_f:,.0f}",
            delta=f"Range: Rp {low_est_f:,.0f} - {up_est_f:,.0f}"
        )
        st.metric(
            label=f"{estimasi_label} (Sinyal {signal.split()[0]})",
            value=f"Rp {est_besok_sinyal_f:,.0f}",
            delta=f"{((est_besok_sinyal_f - harga_terakhir) / harga_terakhir * 100):+.2f}%"
        )
    col3.metric(prob_label, f"{prob_bull:.1f}%")

    # ===== GRAFIK =====
    if PLOTLY_AVAILABLE:
        st.header("📈 Chart Harga & Sinyal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
        buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='#10b981'), name='Buy Signal'))
        for lvl, lbl, clr in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, annotation_text=lbl, annotation_position="right")
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), dragmode='pan')
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # RINGKASAN EKSEKUTIF — REDESIGN v2 (Visual & Interaktif)
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📋 Ringkasan Eksekutif & Rekomendasi")

    # ── Tentukan level sinyal ──
    if rrr < 1.0 and ("BUY" in signal):
        sig_color, sig_icon = "#ef4444", "⚠️"
        sig_label = "BUY ON WEAKNESS"
        sig_desc = "Tren valid, tapi RRR di bawah 1.0"
        kondisi_txt = f"Tren valid, RRR {rrr:.2f} ({rrr_status})"
        langkah_txt = f"Entry di zona {entry_zone_f}, SL Rp {sl_harga_f:,.0f}, TP bertahap Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}"
    elif "STRONG BUY" in signal:
        sig_color, sig_icon = "#10b981", "🟢"
        sig_label = "AGGRESSIVE BUY"
        sig_desc = "Tren kuat & akumulasi volume"
        kondisi_txt = "Tren Kuat & Akumulasi Volume"
        langkah_txt = f"Entry di zona {entry_zone_f}, SL Rp {sl_harga_f:,.0f} (-{sl_pct:.1f}%), TP bertahap Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}"
    elif "BUY" in signal:
        sig_color, sig_icon = "#f59e0b", "🟡"
        sig_label = "BUY ON WEAKNESS"
        sig_desc = "Tren valid, entry bertahap"
        kondisi_txt = f"Tren valid, RRR {rrr:.2f} ({rrr_status})"
        langkah_txt = f"Entry di zona {entry_zone_f}, SL Rp {sl_harga_f:,.0f}, TP bertahap Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}"
    elif "HOLD" in signal:
        sig_color, sig_icon = "#3b82f6", "🔵"
        sig_label = "HOLD"
        sig_desc = "Konsolidasi / transisi"
        kondisi_txt = "Konsolidasi / Transisi"
        langkah_txt = "Jangan tambah posisi, pantau SL"
    else:
        sig_color, sig_icon = "#ef4444", "🔴"
        sig_label = "AVOID / LIQUIDATE"
        sig_desc = "Risiko penurunan / distribusi"
        kondisi_txt = "Risiko Penurunan / Distribusi"
        langkah_txt = "Amankan modal, hindari entry baru"

    # ═══ CARD 1 — SIGNAL BADGE (prominent) ═══
    st.markdown(f"""<div style="background: linear-gradient(135deg, {sig_color}22 0%, {sig_color}08 100%); border-left: 6px solid {sig_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 16px;">
<div style="display: flex; align-items: center; gap: 16px;">
<div style="font-size: 42px; line-height: 1;">{sig_icon}</div>
<div style="flex: 1;">
<div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Rekomendasi</div>
<div style="color: {sig_color}; font-size: 24px; font-weight: 700; line-height: 1.1;">{sig_label}</div>
<div style="color: #cbd5e1; font-size: 13px; margin-top: 4px;">{sig_desc}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ═══ CARD 2 — 3 PILAR: KONDISI | REKOMENDASI | LANGKAH ═══
    col_k, col_r, col_l = st.columns(3)

    with col_k:
        st.markdown(f"""<div style="background:#1e293b; border-radius:10px; padding:14px; min-height:100px; border-top:3px solid #64748b;">
<div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1.2px;">Kondisi</div>
<div style="color:#e2e8f0; font-size:13px; margin-top:8px; line-height:1.5;">{kondisi_txt}</div>
</div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""<div style="background:#1e293b; border-radius:10px; padding:14px; min-height:100px; border-top:3px solid {sig_color};">
<div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1.2px;">Rekomendasi</div>
<div style="color:{sig_color}; font-size:13px; font-weight:600; margin-top:8px; line-height:1.5;">{sig_label}</div>
</div>""", unsafe_allow_html=True)

    with col_l:
        st.markdown(f"""<div style="background:#1e293b; border-radius:10px; padding:14px; min-height:100px; border-top:3px solid #00ffcc;">
<div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1.2px;">Langkah</div>
<div style="color:#e2e8f0; font-size:13px; margin-top:8px; line-height:1.5;">{langkah_txt}</div>
</div>""", unsafe_allow_html=True)

    # ═══ CARD 3 — POSITION STATUS (khusus sudah_beli) ═══
    if sudah_beli:
        if floating_pl_pct is not None:
            if floating_pl_pct > 5:
                pl_color, pl_icon, pl_status = "#10b981", "🚀", "PROFIT BESAR"
                pl_action = "Take profit sebagian / trailing stop"
            elif floating_pl_pct > 0:
                pl_color, pl_icon, pl_status = "#84cc16", "✅", "PROFIT"
                pl_action = "Pantau SL ketat, naikkan trailing stop"
            elif floating_pl_pct > -3:
                pl_color, pl_icon, pl_status = "#f59e0b", "⚠️", "RUGI KECIL"
                pl_action = "Tahan dengan SL sesuai rekomendasi"
            else:
                pl_color, pl_icon, pl_status = "#ef4444", "🔴", "RUGI BESAR"
                pl_action = "Jika menembus SL, segera keluar"
        else:
            pl_color, pl_icon, pl_status = "#64748b", "❔", "P/L Tidak Tersedia"
            pl_action = "Isi harga beli untuk kalkulasi P/L"
            floating_pl_pct = 0

        # Gauge bar position (clamp -20..+20)
        pl_val = floating_pl_pct or 0
        pl_bar_pct = max(-20, min(20, pl_val))
        pl_bar_position = (pl_bar_pct + 20) / 40 * 100  # 0..100

        st.markdown(f"""<div style="background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%); border-radius:12px; padding:18px; margin-top:16px; border:1px solid #334155;">
<div style="display:flex; align-items:center; gap:14px; margin-bottom:14px; flex-wrap:wrap;">
<div style="font-size:36px;">{pl_icon}</div>
<div style="flex:1; min-width:150px;">
<div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1.2px;">Status Posisi Kamu</div>
<div style="color:{pl_color}; font-size:20px; font-weight:700; margin-top:2px;">{pl_status} {pl_val:+.2f}%</div>
</div>
<div style="background:{pl_color}22; border:1px solid {pl_color}; border-radius:8px; padding:6px 12px; color:{pl_color}; font-size:11px; font-weight:600;">{pl_action}</div>
</div>
<div style="background:#0f1116; border-radius:6px; padding:6px; height:24px; position:relative; margin-top:8px;">
<div style="position:absolute; left:50%; top:0; bottom:0; width:2px; background:#475569;"></div>
<div style="position:absolute; left:{pl_bar_position:.1f}%; top:3px; bottom:3px; width:14px; margin-left:-7px; background:{pl_color}; border-radius:7px; box-shadow:0 0 8px {pl_color};"></div>
</div>
<div style="display:flex; justify-content:space-between; color:#64748b; font-size:10px; margin-top:6px;">
<span>-20%</span>
<span>0%</span>
<span>+20%</span>
</div>
</div>""", unsafe_allow_html=True)
    else:
        # Kalau belum punya posisi, tampilkan hint
        st.markdown(f"""<div style="background:#1e293b; border-radius:10px; padding:14px 18px; margin-top:16px; border-left:4px solid #64748b; display:flex; align-items:center; gap:12px;">
<div style="font-size:28px;">🆓</div>
<div>
<div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1.2px;">Status Posisi</div>
<div style="color:#cbd5e1; font-size:13px; margin-top:2px;">Belum punya posisi di saham ini. Siap entry di zona rekomendasi.</div>
</div>
</div>""", unsafe_allow_html=True)

    # ═══ CARD 4 — TIPS & WARNINGS ═══
    tips = []
    if "BUY" in signal:
        if rrr < 1.5:
            tips.append(("💡", "#f59e0b",
                         f"<b>Tips Dip Entry:</b> Untuk RRR ideal 1:2.0, antri beli di <b>Rp {entry_ideal_f:,.0f}</b> atau lebih rendah"))
        if sl_pct > 10.0:
            tips.append(("⚠️", "#ef4444",
                         f"<b>SL Lebar (-{sl_pct:.1f}%):</b> Sesuaikan ukuran posisi maksimal <b>{risk_adjusted_alloc:.1f}%</b> dari modal agar risiko total terjaga"))

    if tips:
        for icon, color, text in tips:
            st.markdown(f"""<div style="background:{color}12; border-left:4px solid {color}; border-radius:8px; padding:12px 16px; margin-top:10px; color:#cbd5e1; font-size:13px; line-height:1.5;">
<span style="color:{color}; font-weight:bold; font-size:15px;">{icon}</span>&nbsp;&nbsp;{text}
</div>""", unsafe_allow_html=True)

    # ═══ DISCLAIMER ═══
    st.markdown("""<div style="color:#64748b; font-size:11px; margin-top:20px; text-align:center; font-style:italic;">
⚠️ Hasil pengujian berbasis permodelan matematika probabilitas kuantitatif historis. Keputusan akhir eksekusi modal tetap merupakan tanggung jawab penuh masing-masing investor.
</div>""", unsafe_allow_html=True)

    # ===== DETAIL EXPANDER =====
    with st.expander("🔍 Lihat Detail Analisis (Berita, Fundamental, Backtest, dll)"):
        st.subheader("📰 Sentimen Berita Terbobot")
        c1, c2 = st.columns([1, 2])
        c1.metric("Sentimen Skor", f"{avg_sentiment:.2f}", sentimen_status)
        with c2:
            st.markdown("**5 Berita Utama Pasar:**")
            for i, h in enumerate(headlines):
                src = sources[i] if i < len(sources) else ""
                t = translated[i] if i < len(translated) else ""
                st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                if t and t != h:
                    st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🧬 Regime Pasar & Volatilitas")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Regime", regime)
        m2.metric("Kondisi Makro IHSG", ihsg_cond)
        m3.metric("ADX Adaptif", f"{adx:.1f} (Thresh: {adx_threshold:.1f})")
        m4.metric("OFI Ratio", f"{ofi_now:.2f}")
        if is_daytrade:
            vwap_col = st.columns(1)[0]
            vwap_col.metric("VWAP", f"{vwap_now:,.0f}", vwap_bias)
        st.markdown(f"**Insight Regime:** {generate_regime_insight(regime, adx, ofi_now, ihsg_cond)}")

        st.divider()
        st.subheader("📊 Metrik Fundamental Saham (IDX)")
        if ticker_info:
            def clean_val(v, f="{:.2f}"):
                return "N/A" if v is None else f.format(v)

            def singkat_angka(n):
                if n is None:
                    return "N/A"
                n = float(n)
                if n >= 1e12:
                    return f"{n/1e12:,.1f} T"
                elif n >= 1e9:
                    return f"{n/1e9:,.0f} M"
                else:
                    return f"{n:,.0f}"

            mc_short = singkat_angka(mc)
            table_html = (
                f"<table class='fundamental-table'>"
                f"<tr><td>Market Cap</td><td>{mc_short} IDR</td></tr>"
                f"<tr><td>PER</td><td>{clean_val(per, '{:.2f}x')}</td></tr>"
                f"<tr><td>PBV</td><td>{clean_val(pbv, '{:.2f}x')}</td></tr>"
                f"<tr><td>ROE</td><td>{clean_val(roe*100 if roe else None, '{:.1f}%')}</td></tr>"
                f"<tr><td>D/E</td><td>{clean_val(de, '{:.2f}%')}</td></tr>"
                f"</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            interpretation_items = []
            if mc:
                if mc >= 1e13:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong sangat besar (Mega Cap)."
                elif mc >= 1e12:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong besar (Blue Chip)."
                elif mc >= 1e10:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong menengah (Mid Cap)."
                else:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong kecil (Small Cap)."
            else:
                mct = "Market Cap tidak tersedia."
            interpretation_items.append(f"<li><b>Market Cap:</b> {mct}</li>")

            if per:
                if per < 10:
                    pt = f"PER {per:.2f}x tergolong rendah (potensi undervalue)."
                elif per < 20:
                    pt = f"PER {per:.2f}x moderat."
                else:
                    pt = f"PER {per:.2f}x tergolong tinggi (premium)."
            else:
                pt = "PER tidak tersedia."
            interpretation_items.append(f"<li><b>PER:</b> {pt}</li>")

            if pbv:
                if pbv < 1:
                    pbt = f"PBV {pbv:.2f}x di bawah 1 (di bawah nilai buku, bisa undervalue)."
                elif pbv < 3:
                    pbt = f"PBV {pbv:.2f}x moderat."
                else:
                    pbt = f"PBV {pbv:.2f}x tinggi (premium)."
            else:
                pbt = "PBV tidak tersedia."
            interpretation_items.append(f"<li><b>PBV:</b> {pbt}</li>")

            if roe:
                roep = roe * 100
                if roep > 20:
                    rt = f"ROE {roep:.1f}% sangat baik (profitabilitas tinggi)."
                elif roep > 10:
                    rt = f"ROE {roep:.1f}% cukup baik."
                else:
                    rt = f"ROE {roep:.1f}% rendah."
            else:
                rt = "ROE tidak tersedia."
            interpretation_items.append(f"<li><b>ROE:</b> {rt}</li>")

            if de:
                if de > 1:
                    dt = f"D/E {de:.2f} tinggi (leverage tinggi, risiko lebih besar)."
                elif de > 0.5:
                    dt = f"D/E {de:.2f} moderat."
                else:
                    dt = f"D/E {de:.2f} rendah (konservatif)."
            else:
                dt = "D/E tidak tersedia."
            interpretation_items.append(f"<li><b>D/E:</b> {dt}</li>")

            st.markdown(f'<div style="background-color:#1e293b;border-radius:12px;padding:15px;margin-top:15px;color:#cbd5e1;font-size:14px;"><b style="color:#00ffcc;">📝 Interpretasi Metrik:</b><ul style="margin-top:8px;padding-left:20px;">{"".join(interpretation_items)}</ul></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data fundamental finansial tidak tersedia.")

        st.divider()
        st.subheader("🎯 Target Pivot & Support/Resistance")
        p1, p2, p3, p4, p5 = st.columns(5)
        r2_f = fraksi_bei(r2)
        r1_f = fraksi_bei(r1)
        pp_f = fraksi_bei(pp)
        s1_f = fraksi_bei(s1)
        s2_f = fraksi_bei(s2)

        p1.metric("R2", f"Rp {r2_f:,.0f}".replace(",", "."))
        p2.metric("R1", f"Rp {r1_f:,.0f}".replace(",", "."))
        p3.metric("Pivot", f"Rp {pp_f:,.0f}".replace(",", "."))
        p4.metric("S1", f"Rp {s1_f:,.0f}".replace(",", "."))
        p5.metric("S2", f"Rp {s2_f:,.0f}".replace(",", "."))
        st.write(f"Kondisi {breakout_label}: **{breakout}**")

        st.divider()
        st.subheader("🔮 Sinyal Kuantitatif & Hasil Backtest" + (" (Intraday)" if is_daytrade else " (6 Bulan)"))
        if "AVOID" not in signal:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sinyal", signal)
            c2.metric(estimasi_label, f"Rp {est_besok_f:,.0f}".replace(",", "."))
            c3.metric("Entry Zone", entry_zone_f)
            c4.metric("TP Range", f"Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}",
                      f"+{tp_pct_low:.1f}% ~ +{tp_pct_high:.1f}%")
            c5.metric("Stop Loss", f"Rp {sl_harga_f:,.0f}", f"-{sl_pct:.1f}%")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Sinyal", signal)
            c2.metric(estimasi_label, f"Rp {est_besok_f:,.0f}".replace(",", "."))
            st.info("⛔ Tidak ada rekomendasi entry, TP, atau SL untuk sinyal AVOID.")

        st.markdown(f"**Hasil Backtest ({backtest_window} Bar):**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
        b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt and pf_bt != np.inf else "N/A")
        b3.metric("Avg Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
        b4.metric("Max DD Strat", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        b5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
        b6.metric("Total Trades", trades_bt)

        st.divider()
        st.subheader("🛡️ Manajemen Risiko Portofolio (Kelly)")
        rc1, rc2 = st.columns(2)
        rc1.metric("Alokasi Maks (Kelly)", f"{kelly_adj*100:.1f}%")
        rc2.metric("Beta IHSG", f"{beta_ihsg:.2f}x")
        st.markdown(f"**Interpretasi:** Berdasarkan Win Rate **{win_bt:.1%}**, maksimal alokasi **{kelly_adj*100:.1f}%** dari total ekuitas.")
        st.markdown(f"Max DD Historis: `{max_dd:.2f}%` | DD 30 Hari: `{max_dd_30:.2f}%`")

        st.divider()
        _mc_label = res.get("mc_regime_label", "Monte Carlo")
        st.subheader(f"🎲 Simulasi Monte Carlo — {_mc_label}")
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric(prob_label, f"{prob_bull:.1f}%")
        pr2.metric("Prob. Sentuh R1 (30H)", f"{hit_tp:.1f}%")
        pr3.metric("Prob. Sentuh S2 (30H)", f"{hit_sl:.1f}%")

    # ══════════════════════════════════════════════════════════
    # V12 ADAPTIVE ENGINE – EXPANDER & LOGIC (DENGAN INSIGHT)
    # ══════════════════════════════════════════════════════════
    with st.expander("🧬 V12 Adaptive Engine (Coppock, Self‑Learning)"):
        st.info(
            "⚙️ **Bagian ini adalah otak adaptif dari QuantRisk Pro.** "
            "Engine secara otomatis mempelajari akurasi setiap faktor teknikal berdasarkan riwayat analisis kamu. "
            "Semakin sering suatu ticker dianalisis, semakin akurat bobot yang dihasilkan."
        )

        if not is_daytrade:
            st.markdown("### 📈 Coppock Curve & Beta IHSG")
            if coppock_turning_up:
                coppock_insight = "🟢 **Turning Up** – Sinyal awal akumulasi. Momentum bullish jangka panjang mulai terbentuk, potensi tren naik."
            elif coppock_rising:
                coppock_insight = "🟢 **Rising** – Tren bullish jangka panjang masih sehat. Akumulasi masih berlangsung."
            else:
                coppock_insight = "🔴 **Falling** – Momentum bullish melemah. Waspadai potensi koreksi atau perubahan tren."
            if beta_ihsg > 1.2:
                beta_insight = f"⚠️ **Beta Tinggi ({beta_ihsg:.2f})** – Saham lebih volatile dari IHSG. Cocok untuk *trading agresif*, namun risikonya lebih besar saat pasar turun."
            elif beta_ihsg > 0.8:
                beta_insight = f"✅ **Beta Moderat ({beta_ihsg:.2f})** – Pergerakan selaras dengan IHSG. Cocok untuk *swing trading*."
            else:
                beta_insight = f"🛡️ **Beta Rendah ({beta_ihsg:.2f})** – Saham defensif, lebih stabil dari IHSG. Cocok untuk *investasi jangka panjang*."
            col_cop1, col_cop2 = st.columns(2)
            with col_cop1:
                st.metric("Coppock Curve", f"{coppock_val:.3f}",
                          "Turning Up ✅" if coppock_turning_up else ("Rising 📈" if coppock_rising else "Falling 📉"))
                st.caption(coppock_insight)
            with col_cop2:
                st.metric("Beta IHSG", f"{beta_ihsg:.2f}x", help="Beta > 1 : lebih volatile dari IHSG, Beta < 1 : lebih stabil.")
                st.caption(beta_insight)
        else:
            st.markdown("### 📈 Beta IHSG")
            if beta_ihsg > 1.2:
                beta_insight = f"⚠️ **Beta Tinggi ({beta_ihsg:.2f})** – Saham lebih volatile dari IHSG. Cocok untuk *trading agresif*, namun risikonya lebih besar saat pasar turun."
            elif beta_ihsg > 0.8:
                beta_insight = f"✅ **Beta Moderat ({beta_ihsg:.2f})** – Pergerakan selaras dengan IHSG. Cocok untuk *swing trading*."
            else:
                beta_insight = f"🛡️ **Beta Rendah ({beta_ihsg:.2f})** – Saham defensif, lebih stabil dari IHSG. Cocok untuk *investasi jangka panjang*."
            st.metric("Beta IHSG", f"{beta_ihsg:.2f}x", help="Beta > 1 : lebih volatile dari IHSG, Beta < 1 : lebih stabil.")
            st.caption(beta_insight)
            st.info("ℹ️ Coppock Curve tidak ditampilkan untuk Day Trade karena kurang relevan dengan timeframe intraday.")

        st.markdown("### ⚓ Multi-Timeframe (MTF) Alignment")
        st.info(res.get('mtf_status_text', 'N/A'))
        if not res.get('is_mtf_bullish', True):
            st.warning("⚠️ Tren timeframe atasan sedang tidak mendukung (Counter-Trend). Sinyal beli diturunkan risikonya.")

        st.markdown("### 🔬 Volume Spread Analysis (VSA)")
        vsa_text = res.get('vsa_status_text', 'N/A')
        if res.get('is_marking_close') or res.get('is_no_demand'):
            st.error(f"**{vsa_text}**")
        elif res.get('is_stopping_volume'):
            st.success(f"**{vsa_text}**")
        else:
            st.info(vsa_text)
        # Legenda interpretasi VSA
        with st.expander("ℹ️ Cara Membaca VSA", expanded=False):
            st.markdown(
                "- **⚠️ Marking Close:** Harga ditarik naik di 10 menit terakhir penutupan dengan volume sangat sepi. "
                "Kemungkinan besar saham akan *gap down* atau tertekan keesokan harinya.\n"
                "- **🔴 No Demand:** Candle bullish dengan spread sempit & volume di bawah rata-rata. "
                "Breakout palsu — tidak ada partisipasi buyer besar.\n"
                "- **🟢 Stopping Volume:** Candle bearish besar dengan volume meledak, tetapi harga menutup di atas tengah candle. "
                "Sinyal *smart money* sedang menampung barang (*absorption*). Waspadai *reversal*."
            )

        st.markdown("### ⚖️ Bobot Adaptif per Faktor")
        st.caption(
            "Bobot di bawah dihitung otomatis berdasarkan **akurasi historis** masing‑masing faktor. "
            "Faktor yang sering benar mendapat bobot lebih tinggi. Bobot ini digunakan untuk sinyal akhir."
        )

        if is_daytrade:
            display_adaptive_w = {k: v for k, v in adaptive_w.items() if k != "Coppock"}
            st.caption("ℹ️ Faktor **Coppock** tidak ditampilkan dalam bobot adaptif untuk Day Trade karena kurang relevan secara intraday. "
                       "Namun, data-nya tetap dihitung di background untuk menjaga konsistensi historis.")
        else:
            display_adaptive_w = adaptive_w

        w_df = pd.DataFrame.from_dict(display_adaptive_w, orient='index', columns=['Weight'])
        st.bar_chart(w_df)

        if display_adaptive_w:
            max_factor = max(display_adaptive_w, key=display_adaptive_w.get)
            min_factor = min(display_adaptive_w, key=display_adaptive_w.get)
            max_weight = display_adaptive_w[max_factor]
            min_weight = display_adaptive_w[min_factor]

            weight_insight = f"🔍 **Faktor paling dominan:** **{max_factor}** (bobot {max_weight:.1%}). "
            weight_insight += f"**{min_factor}** memiliki bobot terendah ({min_weight:.1%}).\n\n"

            interpretations = {
                "Momentum": "Sinyal momentum (harga 5 hari) paling berpengaruh – pasar sedang *trend-following*. Ikuti tren yang sedang berlangsung.",
                "AI_Senti": "Sentimen berita paling berpengaruh – pergerakan saham banyak dipicu oleh berita/isu terkini. Pantau terus sentimen.",
                "MeanRev": "*Reversal* ke rata-rata (Z-Score) paling berpengaruh – saham cenderung kembali ke level wajar setelah jenuh beli/jual.",
                "Beta_IHSG": "Beta IHSG paling berpengaruh – saham sangat terpengaruh oleh pergerakan pasar secara keseluruhan. Perhatikan arah IHSG.",
                "Coppock": "Coppock Curve paling berpengaruh – sinyal jangka panjang mendominasi, tren utama sedang kuat. Ikuti sinyal makro.",
                "OFI": "Order Flow Imbalance (OFI) paling berpengaruh – tekanan order book agresif sangat menentukan arah pergerakan.",
                "Bandar_Flow": "Bandarmology (Top 3 Net Buy) paling berpengaruh – akumulasi/distribusi bandar memegang kendali atas tren saat ini.",
                "Foreign_ZScore": "Foreign Flow paling berpengaruh – aksi beli/jual investor asing menjadi penggerak utama saham ini."
            }
            weight_insight += interpretations.get(max_factor, "")
            st.info(weight_insight)

        st.markdown("### 🧠 Status Memori Adaptif")
        st.caption(
            "**Accuracy** = seberapa sering sinyal faktor sesuai arah harga. **Error EMA** = rata‑rata kesalahan prediksi (makin kecil makin baik)."
        )
        mem = st.session_state.v12_memory.get(ticker_raw, {})
        if mem:
            keys_to_show = [k for k in FACTOR_KEYS if not (is_daytrade and k == "Coppock")]
            acc_data = {k: mem.get('accuracy', {}).get(k, 0.5) for k in keys_to_show}
            err_data = {k: mem.get('error_ema', {}).get(k, 1.0) for k in keys_to_show}

            col_a, col_e = st.columns(2)
            with col_a:
                st.caption("✅ Accuracy (higher = better)")
                st.bar_chart(pd.Series(acc_data))
            with col_e:
                st.caption("⚠️ Error EMA (lower = better)")
                st.bar_chart(pd.Series(err_data))

            best_factor = max(acc_data, key=acc_data.get)
            worst_factor = min(acc_data, key=acc_data.get)
            mem_insight = f"🏆 **Faktor paling akurat:** **{best_factor}** (akurasi {acc_data[best_factor]:.1%}). "
            mem_insight += f"Faktor **{worst_factor}** perlu dievaluasi (akurasi {acc_data[worst_factor]:.1%})."
            st.caption(mem_insight)

            entry_err = mem.get('entry_error_ema', 0.0)
            if entry_err > 0:
                st.caption(
                    f"🎯 **Rata‑rata error entry:** {entry_err:.2f} poin. "
                    "Entry sering tidak tersentuh, engine akan menggeser zona entry lebih dekat ke harga."
                )
            else:
                st.caption("🎯 **Error entry:** 0 — entry zone sudah cukup baik atau belum ada data Not Touched.")
        else:
            st.info("Belum ada data memori untuk ticker ini. Lakukan analisis beberapa kali agar engine mulai belajar.")

        st.markdown("### 🔁 Proses Self‑Learning")
        st.caption(
            "Setiap analisis, engine membandingkan prediksi sebelumnya dengan harga aktual. "
            "Jika benar → akurasi naik. Jika salah → error bertambah. Bobot otomatis menyesuaikan. "
            "Selain itu, engine juga mempelajari **level entry** dari kejadian Entry Tidak Tersentuh."
        )

        last_pred = load_v12_predictions(ticker_raw, mode='daytrade' if is_daytrade else 'swing')
        if last_pred:
            last_close = safe_float(last_pred.get('close_price'), 0.0)
            if last_close > 0:
                last_signals = {}
                for k in FACTOR_KEYS:
                    key = f'sig_{k}'
                    if key in last_pred:
                        last_signals[k] = safe_float(last_pred[key], 0.0)
                    else:
                        last_signals[k] = 0.0

                actual_return = (harga_terakhir - last_close) / last_close
                volatility = returns.std()
                update_v12_memory(ticker_raw, last_signals, actual_return, volatility)
                st.success(f"✅ Memory updated! Actual return sejak prediksi terakhir: {actual_return*100:.2f}%")
            else:
                st.info("ℹ️ Prediksi sebelumnya tidak memiliki close_price yang valid.")
        else:
            st.info("ℹ️ Tidak ada prediksi sebelumnya. Engine akan mulai belajar pada analisis berikutnya.")

    # ==================== AI INSIGHT OTOMATIS ====================
    st.markdown("---")
    if st.session_state.get("gemini_api_key"):
        with st.spinner("🧠 AI sedang menganalisis hasil dan riwayat..."):

            # ═══ FIX: Definisikan act_ticker_data di sini (safe) ═══
            act_ticker_data = hitung_winrate_ticker_actual(
                ticker_raw,
                st.session_state.get('riwayat_actual', {})
            )

            # Safe extraction
            _act_str = "Belum ada evaluasi"
            try:
                if isinstance(act_ticker_data, dict):
                    _tot = act_ticker_data.get('total', 0) or 0
                    _wr  = act_ticker_data.get('win_rate')
                    _w   = act_ticker_data.get('win', 0) or 0
                    _l   = act_ticker_data.get('loss', 0) or 0
                    if _tot > 0 and isinstance(_wr, (int, float)):
                        _act_str = f"{_wr:.1f}% ({_w} Win / {_l} Loss)"
            except Exception:
                _act_str = "Belum ada evaluasi"
            # ═══════════════════════════════════════════════════════

            data_ai = {
                "Saham": ticker_raw,
                "Harga": f"{harga_terakhir:,.0f}",
                "Sinyal": signal,
                "Rezim": regime,
                "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
                "RRR": f"{rrr:.2f} (Kontekstual)",
                "Prob Naik": f"{prob_bull:.1f}%",
                "TP%": f"{tp_pct_low:.1f}% - {tp_pct_high:.1f}%",
                "SL%": f"{sl_pct:.1f}",
                "Estimasi": f"{est_besok_f:,.0f}",
                "Beta": f"{beta_ihsg:.2f}x",
                "WinRate": f"{win_bt:.1%}" if trades_bt else "N/A",
                "Actual_WinRate_Ticker": _act_str,   # ← pakai variable safe
                "ProfitFactor": f"{pf_bt:.2f}" if trades_bt else "N/A",
                "MaxDD": f"{max_dd_bt:.2f}%" if trades_bt else "N/A",
                "Kelly": f"{kelly_adj*100:.1f}",
                "Fundamental_MC": f"{mc:,.0f}" if mc else "N/A",
                "Fundamental_PER": f"{per:.2f}" if per else "N/A",
                "Fundamental_PBV": f"{pbv:.2f}" if pbv else "N/A",
                "Fundamental_ROE": f"{roe*100:.1f}" if roe else "N/A",
                "Fundamental_DE": f"{de:.2f}" if de else "N/A",
                "Status_Posisi": "Sudah memiliki saham" if sudah_beli else "Belum memiliki saham",
                "Harga_Beli": f"Rp {harga_beli_float:,.0f}" if harga_beli_float else "Tidak diisi",
                "Floating_PL": f"{floating_pl_pct:+.2f}%" if floating_pl_pct is not None else "N/A"
            }
            riwayat_konteks = []
            for r in st.session_state.riwayat:
                if r['Saham'] == ticker_raw:
                    r_copy = dict(r)
                    # Ambil mode/gaya dari baris riwayat
                    mode_actual = r.get('Gaya', 'SW')   # "SW" atau "DT"

                    # Coba key 3 elemen (format baru)
                    key_actual_baru = (r.get('Waktu'), r.get('Saham'), mode_actual)
                    actual = st.session_state.riwayat_actual.get(key_actual_baru)

                    # Fallback ke key 2 elemen (format lama)
                    if actual is None:
                        key_actual_lama = (r.get('Waktu'), r.get('Saham'))
                        actual = st.session_state.riwayat_actual.get(key_actual_lama, {})

                    if actual:
                        r_copy['Actual_High']   = actual.get('Actual_High', '')
                        r_copy['Actual_Low']    = actual.get('Actual_Low', '')
                        r_copy['Actual_Close']  = actual.get('Actual_Close', '')
                        r_copy['Actual_Outcome']= actual.get('Outcome', '')
                        r_copy['Entry_Miss']    = actual.get('Entry_Miss', '')

                    riwayat_konteks.append(r_copy)
                    if len(riwayat_konteks) >= 20:
                        break

            hasil_ai, error_ai = analisis_saham_dengan_ai(data_ai, riwayat_konteks, st.session_state.gemini_api_key, ticker=ticker_clean)
            if not error_ai and hasil_ai:
                hasil_ai_bersih = bersihkan_teks_ai(hasil_ai)
                html_ai = f'<div class="ai-insight-card"><h3>🤖 Insight AI</h3><p>{hasil_ai_bersih}</p></div>'
                st.markdown(html_ai, unsafe_allow_html=True)
            elif error_ai:
                st.warning(f"AI tidak dapat memberikan insight: {error_ai}")
    else:
        st.info("💡 Isi API Key Gemini di sidebar untuk mendapatkan insight AI otomatis.")
# ==================== PROSES ANALISIS ====================
if run_btn:
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!")
        st.stop()



    with st.spinner("🤖 Menganalisis mode Swing dan Daytrade secara paralel..."):
        from concurrent.futures import ThreadPoolExecutor
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        except ImportError:
            try:
                from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx
            except ImportError:
                add_script_run_ctx = None
                get_script_run_ctx = None

        ctx = get_script_run_ctx() if get_script_run_ctx is not None else None
        v12_mem_snapshot = dict(st.session_state.v12_memory) if "v12_memory" in st.session_state else {}

        def run_analysis_task(func, *args, **kwargs):
            if add_script_run_ctx and ctx:
                add_script_run_ctx(ctx=ctx)
            return func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_swing = executor.submit(
                run_analysis_task,
                analyze_stock,
                ticker_input, harga_manual, sudah_beli, harga_beli_float,
                False, v12_mem_snapshot, fee_beli_pct, fee_jual_pct  # swing
            )
            future_day = executor.submit(
                run_analysis_task,
                analyze_stock,
                ticker_input, harga_manual, sudah_beli, harga_beli_float,
                True, v12_mem_snapshot, fee_beli_pct, fee_jual_pct   # daytrade
            )
            res_swing = future_swing.result()
            res_day = future_day.result()

    if res_swing is None or res_day is None:
        st.error("❌ Gagal mengambil data untuk salah satu mode.")
        st.stop()

    # ----- REKOMENDASI MODE HYBRID (KUANTITATIF + AI + BANDARMOLOGY) -----
    def skor_mode_math(res):
        sig_raw = res.get('signal_score', 0)
        sig = sig_raw * 100.0 if sig_raw <= 1.0 else sig_raw
        rrr_score = min(max(res.get('rrr', 0), 0.0), 5.0) * 20.0
        conf_raw = res.get('confidence', 0)
        conf = conf_raw * 100.0 if conf_raw <= 1.0 else conf_raw
        return (sig * 0.5) + (rrr_score * 0.2) + (conf * 0.3)

    skor_math_swing = skor_mode_math(res_swing)
    skor_math_day = skor_mode_math(res_day)

    ai_data = None
    ai_err = None
    gemini_key = st.session_state.get("gemini_api_key")
    if gemini_key:
        with st.spinner("🤖 AI Gemini sedang mengevaluasi rekomendasi mode..."):
            ai_data, ai_err = evaluasi_mode_dengan_ai(res_swing, res_day, ticker_raw, gemini_key)

    if ai_data and isinstance(ai_data, dict) and 'swing_score' in ai_data and 'day_score' in ai_data:
        try:
            ai_swing_score = float(ai_data.get('swing_score', 50))
            ai_day_score = float(ai_data.get('day_score', 50))
            
            final_swing_score = (skor_math_swing * 0.7) + (ai_swing_score * 0.3)
            final_day_score = (skor_math_day * 0.7) + (ai_day_score * 0.3)
            
            ai_reason = str(ai_data.get('reasoning', '')).strip()
            mode_badge = "🤖 Hybrid (Kuantitatif 70% + AI 30%)"
        except Exception:
            final_swing_score = skor_math_swing
            final_day_score = skor_math_day
            ai_reason = ""
            mode_badge = "📊 Kuantitatif Only"
    else:
        final_swing_score = skor_math_swing
        final_day_score = skor_math_day
        ai_reason = ""
        if not gemini_key:
            mode_badge = "📊 Kuantitatif Only (Gemini API Key belum diisi di Sidebar)"
        elif ai_err:
            mode_badge = f"📊 Kuantitatif Only (AI Error: {ai_err})"
        else:
            mode_badge = "📊 Kuantitatif Only"

    if final_swing_score >= final_day_score:
        mode_terbaik = "Swing Trade"
        alasan_default = "Sinyal swing lebih kuat dan RRR lebih baik."
        res_terbaik = res_swing
        best_final_score = final_swing_score
    else:
        mode_terbaik = "Day Trade"
        alasan_default = "Sinyal intraday lebih kuat dan probabilitas naik lebih tinggi."
        res_terbaik = res_day
        best_final_score = final_day_score

    alasan_final = ai_reason if ai_reason else alasan_default

    st.success(
        f"🏆 **Rekomendasi Mode: {mode_terbaik}** — {alasan_final}\n\n"
        f"`{mode_badge}` | Skor Final: **{best_final_score:.1f}/100** "
        f"(Swing: {final_swing_score:.1f} vs Day: {final_day_score:.1f})"
    )

    # ----- TAMPILKAN HASIL KETIGA MODE DALAM TAB -----
    tab_swing, tab_day, tab_bandar = st.tabs([
        "📆 Swing Trade",
        "⏱️ Day Trade",
        "🐳 Bandarmology"
    ])

    with tab_swing:
        display_analysis_result(res_swing)

    with tab_day:
        display_analysis_result(res_day)

    with tab_bandar:
        display_bandarmology_tab(ticker_clean)

    # ----- SIMPAN PREDIKSI V12 UNTUK KEDUA MODE -----
    for res in [res_swing, res_day]:
        try:
            # GUNAKAN HARGA ASLI, bukan harga manual user
            close_for_learning = res.get('harga_terakhir_asli') or res['harga_terakhir']
            save_v12_prediction(
                ticker_raw,
                close_for_learning,          # ✅ FIX
                res['norm_signals'],
                entry_low=res['entry_low_f'],
                entry_high=res['entry_high_f'],
                mode=res['mode']
            )
        except Exception as e:
            st.warning(f"Gagal menyimpan prediksi {res['mode']}: {e}")
    # ----- SIMPAN RIWAYAT UNTUK KEDUA MODE (SWING & DAYTRADE) -----
    simpan_riwayat(
        [res_swing['ringkasan'], res_day['ringkasan']],
        aksi_mode=st.session_state.get('aksi_simpan_mode', 'simpan_baru'),
        target_saham=ticker_raw
    )

    st.stop()
# ==================== SCANNER SAHAM IDX (V12 TECH SCORE) ====================
if scan_btn:
    st.title("🔍 Scanner Saham IDX (V12 Tech Score)")
    st.write(f"Mode: {mode_scan} | Likuiditas Min: Rp {likuiditas_min:,.0f}/hari")

    with st.spinner("📡 Mengambil daftar saham..."):
        daftar_saham = get_daftar_saham(mode_scan)
        st.info(f"📋 {len(daftar_saham)} saham akan dipindai.")

    # Ambil data IHSG sekali saja untuk semua saham (perlu 6 bulan agar Coppock akurat)
    ihsg_data = load_ihsg_data(period="6mo", interval="1d")
    if ihsg_data.empty:
        st.error("❌ Gagal mengambil data IHSG. Pastikan koneksi internet stabil.")
        st.stop()

    # --- Tombol batal scan (dengan session state) ---
    if "cancel_scan" not in st.session_state:
        st.session_state.cancel_scan = False

    cancel_col, _ = st.columns([1, 5])
    with cancel_col:
        if st.button("⏹️ Batalkan Scan"):
            st.session_state.cancel_scan = True

    progress_bar = st.progress(0)
    status_text = st.empty()
    hasil_scan = []

    # --- Fungsi worker per saham (menggunakan score_stock_tech) ---
    def process_ticker(ticker):
        try:
            ticker_jk = f"{ticker}.JK"
            df = load_stock_data(ticker_jk, period="6mo", interval="1d")
            if df.empty or len(df) < 65:
                return None
            # Filter likuiditas: volume rata2 20 hari * harga terakhir
            if 'Volume' in df.columns and len(df) >= 20:
                avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
                last_price = float(df['Close'].iloc[-1])
                if avg_vol * last_price < likuiditas_min:
                    return None
            # Panggil scoring teknikal (adaptasi Kotlin)
            return score_stock_tech(df, ticker, ihsg_data)
        except:
            return None

    total = len(daftar_saham)
    max_workers = 4  # batasi koneksi paralel agar tidak di-banned Yahoo

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in daftar_saham}
        completed = 0
        for future in as_completed(future_to_ticker):
            if st.session_state.cancel_scan:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            res = future.result()
            if res is not None:
                hasil_scan.append(res)
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Memindai {future_to_ticker[future]} ({completed}/{total})...")

    if st.session_state.cancel_scan:
        st.warning("Scan dibatalkan oleh pengguna.")
        st.stop()

    progress_bar.empty()
    status_text.empty()

    if not hasil_scan:
        st.warning("Tidak ada saham yang lolos filter likuiditas atau data tidak lengkap.")
        st.stop()

    # Urutkan berdasarkan techScore
    hasil_scan.sort(key=lambda x: x['techScore'], reverse=True)

        # --- Pisahkan Beli dan Jual ---
    buy_signals = [r for r in hasil_scan if r['techScore'] > 0.05]
    sell_signals = [r for r in hasil_scan if r['techScore'] < -0.05]
    # TOP BUY: ambil 10 terkuat (techScore tertinggi)
    top_buys = buy_signals[:10]
    # TOP SELL: ambil 10 terlemah (techScore paling negatif)
    # Urutkan sell_signals dari paling negatif ke kurang negatif
    top_sells = sorted(sell_signals, key=lambda x: x['techScore'])[:10]
    # Simpan hasil scan ke session_state agar tidak hilang saat re-run
    st.session_state.scan_results = {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'top_buys': top_buys,
        'top_sells': top_sells,
        'total': total,
        'hasil_scan_count': len(hasil_scan),
        'daftar_saham_count': len(daftar_saham)
    }
    st.rerun()
# ==================== TAMPILAN HASIL SCAN (DARI SESSION STATE) ====================
if st.session_state.get('scan_results'):
    sr = st.session_state.scan_results
    buy_signals  = sr['buy_signals']
    sell_signals = sr['sell_signals']
    top_buys     = sr['top_buys']
    top_sells    = sr['top_sells']
    total        = sr['total']
    hasil_scan_count = sr['hasil_scan_count']
    daftar_saham_count = sr['daftar_saham_count']

    dict_active_swings_scan = dapatkan_dict_swing_aktif()
    if st.session_state.get('hide_active_swings', False):
        buy_signals = [r for r in buy_signals if r['ticker'].replace(".JK", "").strip().upper() not in dict_active_swings_scan]
        top_buys    = [r for r in top_buys if r['ticker'].replace(".JK", "").strip().upper() not in dict_active_swings_scan]

    st.title("🔍 Scanner Saham IDX (V12 Tech Score)")
    st.write(f"Mode: {mode_scan} | Likuiditas Min: Rp {likuiditas_min:,.0f}/hari")

    st.markdown(f"✅ **Berhasil scan:** {hasil_scan_count}/{daftar_saham_count} saham")
    st.markdown(f"📈 Kandidat Beli: {len(buy_signals)} | 📉 Kandidat Jual: {len(sell_signals)}")

    # ==================== TAMPILAN UTAMA: TOP BUY & TOP SELL ====================
    col_buy, col_sell = st.columns([3, 1])
    with col_buy:
        # Jumlah top_buys bisa berubah (maksimal 10)
        st.subheader(f"🏆 TOP {len(top_buys)} RELATIF TERKUAT - Beli")
    with col_sell:
        # Jumlah top_sells juga adaptif (maksimal 10)
        st.subheader(f"🔻 TOP {len(top_sells)} RELATIF TERLEMAH - Jual")

    # ==================== AI RE-RANK (ditingkatkan) ===================
    if ai_rerank and st.session_state.get("gemini_api_key"):
        with st.spinner("🤖 AI memverifikasi 15 kandidat (1 panggilan batch)..."):
            candidates = buy_signals[:15]
            if not candidates:
                st.info("Tidak ada kandidat Beli untuk diverifikasi AI.")
            else:
                # Kumpulkan berita terbaru untuk setiap kandidat
                headlines_map = {}
                for r in candidates:
                    headlines_map[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                # Prompt dengan berita aktual
                prompt = (
                    "Berikut hasil scan teknikal 15 saham. Verifikasi sinyal BUY dengan sentimen berita TERBARU yang saya berikan untuk setiap saham. "
                    "KELUARKAN HANYA JSON array, TANPA teks pembuka, analisis, atau catatan apapun. "
                    "Format: [{\"ticker\": \"BBRI\", \"confirm\": true, \"confidence_boost\": 0.0-0.15, \"reason\": \"singkat berdasarkan berita\"}]\n\n"
                )
                for r in candidates:
                    tick = r['ticker']
                    headlines = headlines_map.get(tick, ["(tidak ada berita)"])
                    prompt += (
                        f"{tick} | Signal: {r['signal']} | Tech Score: {r['techScore']:.3f} | "
                        f"Coppock: {r['coppockLabel']} | Est Return: {r['muEst']*100:.2f}% | "
                        f"Vol Surge: {r['volSurge']*100:.0f}% | RSI: {r['rsi']:.1f} | "
                        f"Z-Score: {r['zScore']:.2f} | Regime: {r['regime']} | "
                        f"Berita: {'; '.join(headlines)}\n"
                    )

                model, err = dapatkan_model_gemini(st.session_state.gemini_api_key)
                if model and not err:
                    try:
                        response = model.generate_content(prompt)
                        raw = response.text.strip()

                        start_idx = raw.rfind('[')
                        ai_data = []
                        if start_idx != -1:
                            json_str = raw[start_idx:].strip()
                            if json_str.startswith("```json"):
                                json_str = json_str[7:]
                            if json_str.endswith("```"):
                                json_str = json_str[:-3]
                            try:
                                ai_data = json.loads(json_str)
                            except json.JSONDecodeError:
                                st.error("Gagal parse JSON dari akhir respons. Menampilkan debug...")
                                with st.expander("🔎 Debug: Raw Response"):
                                    st.code(raw)
                        else:
                            st.error("Tidak ditemukan array JSON dalam respons AI.")
                            with st.expander("🔎 Debug: Raw Response"):
                                st.code(raw)
                            ai_data = []

                        ai_confirmed = 0
                        ai_upgraded = 0
                        for item in ai_data:
                            ticker = item.get("ticker", "").upper()
                            for r in candidates:
                                if r['ticker'] == ticker:
                                    r['ai_confirm'] = item.get('confirm', False)
                                    r['ai_reason'] = item.get('reason', '')
                                    boost = item.get('confidence_boost', 0.0)
                                    r['ai_boost'] = boost
                                    r['hybrid_score'] = r['techScore'] + boost
                                    if r['ai_confirm']:
                                        ai_confirmed += 1
                                        if boost > 0.01:
                                            ai_upgraded += 1
                                    break

                        msg = f"🤖 AI Re‑Rank selesai: **{ai_confirmed}** saham dikonfirmasi"
                        if ai_upgraded > 0:
                            msg += f", **{ai_upgraded}** naik peringkat karena AI"
                        st.success(msg)
                        # Urutkan ulang buy_signals berdasarkan hybrid_score
                        if candidates:
                            for r in candidates:
                                if 'hybrid_score' not in r:
                                    r['hybrid_score'] = r['techScore']
                            buy_signals.sort(key=lambda x: x.get('hybrid_score', x['techScore']), reverse=True)
                            top_buys = buy_signals[:10]
                            # Update session_state agar render menggunakan urutan baru
                            st.session_state.scan_results['buy_signals'] = buy_signals
                            st.session_state.scan_results['top_buys'] = top_buys
                        with st.expander("📋 Lihat Detail AI Re‑Rank"):
                            ai_table = []
                            for r in candidates:
                                if r.get('ai_confirm') is not None:
                                    ai_table.append({
                                        "Ticker": r['ticker'],
                                        "Tech Score": f"{r['techScore']:.3f}",
                                        "Hybrid Score": f"{r.get('hybrid_score', r['techScore']):.3f}",
                                        "AI Boost": f"{r.get('ai_boost', 0):.3f}",
                                        "AI Confirm": "✅" if r['ai_confirm'] else "❌",
                                        "Reason": r.get('ai_reason', '')
                                    })
                            if ai_table:
                                df_ai = pd.DataFrame(ai_table)
                                df_ai.index = range(1, len(df_ai) + 1)
                                st.dataframe(df_ai, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal memproses respons AI: {e}")
                else:
                    st.error("Gagal mengakses Gemini untuk AI Re‑Rank.")
    elif ai_rerank:
        st.info("Isi API Key Gemini di sidebar untuk mengaktifkan AI Re‑Rank.")

    # ---------- Kartu BUY (mirip UI Kotlin) ----------
    for idx, r in enumerate(top_buys):
        rank = idx + 1
        tick_clean = r['ticker'].replace(".JK", "").strip().upper()
        active_info = dict_active_swings_scan.get(tick_clean)

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                badge = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"#{rank}"
                title_text = f"### {badge} {r['ticker']}  —  **{r['signal']}**"
                if active_info:
                    title_text += f" &nbsp; ⏳ `[SWING AKTIF - Hari ke-{active_info['b_days']}]`"
                st.markdown(title_text)
            with col2:
                st.metric("Harga", f"Rp {r['lastPrice']:,.0f}")
                
                # Tampilkan AI Cross-check jika sudah dijalankan
                ai_results = st.session_state.get('ai_crosscheck_buy', [])
                ai_match = next((item for item in ai_results if item.get("ticker", "").upper() == tick_clean), None)
                if ai_match:
                    sent_score = ai_match.get("sentiment_score", 0.0)
                    sent_label = f"+{sent_score:.2f}" if sent_score >= 0 else f"{sent_score:.2f}"
                    status = "☑️ Sejalan" if sent_score > 0 else "⛔ Berlawanan"
                    note = ai_match.get("note", "")
                    st.markdown(f"**AI Sentimen: {sent_label}**<br/>{status}<br/>📰 <small>_{note}_</small>", unsafe_allow_html=True)

            if active_info:
                st.caption(f"⏳ **Swing Aktif dari {active_info['waktu']}** (Hari bursa ke-{active_info['b_days']}) — Target belum tersentuh / outcome belum diisi.")

            # Bar skor + badge konfirmasi AI (jika ada)
            bar_len = int(abs(r['techScore']) * 10)
            bar_str = "█" * bar_len + "░" * (10 - bar_len)
            score_text = f"Tech Score: **{r['techScore']:.3f}**  {bar_str}"
            if r.get('ai_confirm'):
                score_text += f"  |  Hybrid: **{r.get('hybrid_score', r['techScore']):.3f}**"
                st.markdown("☑️ **Dikonfirmasi Tech + Scanner AI Re‑Rank**")
            st.caption(score_text)

            with st.expander("🔎 Detail Indikator"):
                # Baris 1
                ca, cb, cc = st.columns(3)
                ca.metric("Coppock", r['coppockLabel'])
                cb.metric("Est. Return", f"{r['muEst']*100:.2f}%")
                cc.metric("Regime", r['regime'])
                # Baris 2
                ca.metric("Confidence", f"{r['confidence']*100:.0f}%")
                cb.metric("Risk‑Adj (RRR)", f"{r['rrr']:.2f}")
                cc.metric("Likuiditas", r['likuiditas'])
                # Baris 3
                ca.metric("Est. TP Besok", f"Rp {r['tpEst']:,.0f}")
                cb.metric("Est. SL Besok", f"Rp {r['slEst']:,.0f}")
                cc.metric("Zona Entry", f"Rp {r['entryLow']:,.0f}-{r['entryHigh']:,.0f}")
                # Baris 4
                ca.metric("RSI-14", f"{r['rsi']:.1f}")
                cb.metric("Volume Surge", f"{r['volSurge']*100:.0f}%")
                cc.metric("Z‑Score", f"{r['zScore']:.2f}σ")
                # Baris 5
                ca.metric("Bollinger %B", f"{r['bbPct']:.2f}")
                cb.metric("Trend Consistency", f"{r['trendConsistency']:.0f}%")
                cc.metric("Beta | Mom", f"{r['beta']:.2f}β | {r['momScore']*100:.2f}%")
                if r['isCoppockTurningUp']:
                    st.info("⚡ **Coppock Turning Up** — Sinyal akumulasi terkuat!")
                if r.get('ai_reason'):
                    st.caption(f"🧠 AI Reason: {r['ai_reason']}")
            st.divider()

    # ---------- Kartu SELL (ringkas) ----------
    if top_sells:
        with st.container():
            for idx, r in enumerate(top_sells):
                rank = idx + 1
                tick_clean = r['ticker'].replace(".JK", "").strip().upper()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"#{rank} **{r['ticker']}** — {r['signal']}")
                    st.caption(f"Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Regime: {r['regime']}")
                with col2:
                    st.metric("Harga", f"Rp {r['lastPrice']:,.0f}")
                    
                    # Tampilkan AI Cross-check jika sudah dijalankan
                    ai_results = st.session_state.get('ai_crosscheck_sell', [])
                    ai_match = next((item for item in ai_results if item.get("ticker", "").upper() == tick_clean), None)
                    if ai_match:
                        sent_score = ai_match.get("sentiment_score", 0.0)
                        sent_label = f"+{sent_score:.2f}" if sent_score >= 0 else f"{sent_score:.2f}"
                        status = "☑️ Sejalan" if sent_score < 0 else "⛔ Berlawanan"
                        note = ai_match.get("note", "")
                        st.markdown(f"**AI Sentimen: {sent_label}**<br/>{status}<br/>📰 <small>_{note}_</small>", unsafe_allow_html=True)
                st.divider()
    else:
        st.caption("(Tidak ada kandidat Jual yang memenuhi threshold)")

    # ==================== PERKUAT CROSS‑CHECK DENGAN AI (OPSIONAL) ====================
    if buy_signals:   # hanya tampil jika ada kandidat Beli
        st.markdown("---")
        reinforce_col, _ = st.columns([1, 3])
        with reinforce_col:
            if st.button("🛡️ Perkuat Cross‑Check dgn Sentimen AI (tambahan)", key="reinforce_ai"):
                if not st.session_state.get("gemini_api_key"):
                    st.error("API Key Gemini diperlukan.")
                else:
                    with st.spinner("🧠 Mengambil berita terbaru & menganalisis sentimen..."):
                        # Gunakan top_buys yang sudah tampil (maksimal 10)
                        candidates = sr.get('top_buys', [])
                        top_sell_candidates = sr.get('top_sells', [])
                        if not candidates and not top_sell_candidates:
                            st.warning("Tidak ada kandidat untuk dianalisis.")
                            st.stop()
                        # Ambil headline
                        headlines_map = {}
                        for r in candidates:
                            headlines_map[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                        prompt = (
                            "Berikut hasil scan teknikal 15 saham. Verifikasi sinyal BUY dengan sentimen berita TERBARU yang saya berikan. "
                            "KELUARKAN HANYA JSON array, TANPA teks lain. "
                            "Format: [{\"ticker\": \"BBRI\", \"sentiment_score\": 0.0 (skala -1..1), \"note\": \"singkat berdasarkan berita\"}]\n\n"
                        )
                        for r in candidates:
                            tick = r['ticker']
                            headlines = headlines_map.get(tick, ["(tidak ada berita)"])
                            prompt += f"{tick} | Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Berita: {'; '.join(headlines)}\n"

                        model, err = dapatkan_model_gemini(st.session_state.gemini_api_key)
                        if model and not err:
                            try:
                                response = model.generate_content(prompt)
                                raw = response.text.strip()
                                start_idx = raw.rfind('[')
                                sentiments = []
                                if start_idx != -1:
                                    json_str = raw[start_idx:].strip()
                                    if json_str.startswith("```json"): json_str = json_str[7:]
                                    if json_str.endswith("```"): json_str = json_str[:-3]
                                    try:
                                        sentiments = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        st.error("Gagal parse JSON dari akhir respons.")
                                        with st.expander("🔎 Debug: Raw Response"):
                                            st.code(raw)
                                else:
                                    st.error("Tidak ditemukan array JSON dalam respons AI.")
                                    with st.expander("🔎 Debug: Raw Response"):
                                        st.code(raw)

                                # --- TOP JUAL (dengan berita juga) ---
                                sell_ai = []
                                if top_sell_candidates:
                                    headlines_sell = {}
                                    for r in top_sell_candidates:
                                        headlines_sell[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                                    sell_prompt = (
                                        "Berikut hasil scan teknikal saham dengan sinyal JUAL. "
                                        "Verifikasi sentimen berita TERBARU yang saya berikan. "
                                        "KELUARKAN HANYA JSON array: [{\"ticker\": \"BBRI\", \"sentiment_score\": -0.5..0.5, \"note\": \"singkat\"}]\n\n"
                                    )
                                    for r in top_sell_candidates:
                                        tick = r['ticker']
                                        headlines = headlines_sell.get(tick, ["(tidak ada berita)"])
                                        sell_prompt += f"{tick} | Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Berita: {'; '.join(headlines)}\n"

                                    try:
                                        resp_s = model.generate_content(sell_prompt)
                                        raw_s = resp_s.text.strip()
                                        start_s = raw_s.rfind('[')
                                        if start_s != -1:
                                            json_s = raw_s[start_s:].strip()
                                            if json_s.startswith("```json"): json_s = json_s[7:]
                                            if json_s.endswith("```"): json_s = json_s[:-3]
                                            try:
                                                sell_ai = json.loads(json_s)
                                            except json.JSONDecodeError:
                                                pass
                                    except Exception:
                                        pass

                                # --- SIMPAN KE SESSION STATE DAN RE-RUN ---
                                st.session_state['ai_crosscheck_buy'] = sentiments
                                st.session_state['ai_crosscheck_sell'] = sell_ai
                                st.session_state['ai_crosscheck_done'] = True
                                st.rerun()

                            except Exception as e:
                                st.error(f"Gagal memproses respons AI: {e}")
                        else:
                            st.error("Gagal mengakses Gemini.")

            if st.session_state.get('ai_crosscheck_done'):
                st.success("✅ Cross‑Check Sentimen AI berhasil! Hasil analisis (Skor -1 s.d +1) telah disematkan di sebelah kanan pada masing-masing kartu saham di atas.")
                st.caption(
                    "Quick Technical Cross-Check (7 faktor) + sentimen berita AI "
                    "(berita terbaru dari Google News & Ipotnews) "
                    "= 8 dari 9 faktor Single Quant. "
                    "MASIH bukan Single Quant penuh. Broker Summary tetap perlu input manual per-saham."
                )
# ==================== TAMPILAN AWAL (SEBELUM ANALISIS) ====================
else:
    st.title("📊 Quant & Risk Engine Pro")
    st.markdown("""
    ## Selamat Datang di Dashboard Analisis Saham IHSG
    
    **Fitur Utama:**
    - 🔍 Analisis teknikal lengkap (EMA, ADX, RSI, Z-Score, Momentum, dll)
    - 📈 Sinyal trading adaptif (BUY/HOLD/AVOID) berdasarkan kondisi pasar
    - 🧠 V12 Adaptive Engine dengan self-learning untuk bobot indikator
    - 📰 Analisis sentimen berita dari berbagai sumber
    - 📊 Metrik fundamental (Market Cap, PER, PBV, ROE, D/E)
    - 🎲 Simulasi Monte Carlo untuk probabilitas naik & sentuh level
    - 🤖 AI Insight otomatis menggunakan Google Gemini (perlu API key)
    - 💾 Riwayat analisis tersimpan di Google Sheets (persisten)
    
    **Cara Memulai:**
    1. Pilih **Gaya Trading** di sidebar (Swing Trade mingguan / Day Trade harian)
    2. Masukkan **kode saham** IHSG (contoh: BBRI, TLKM, BMRI) – akhiran `.JK` otomatis ditambahkan
    3. Klik tombol **🚀 ANALISIS** dan tunggu beberapa detik
    
    > **Disclaimer:** Dashboard ini merupakan alat bantu analisis kuantitatif. Keputusan investasi tetap tanggung jawab masing-masing. Data historis tidak menjamin performa masa depan.
    """)

    st.markdown("---")
    st.subheader("🏆 Track Record System (Win Rate Live Engine)")
    st.caption("Statistik akurasi sinyal berdasarkan evaluasi outcome riil yang tersimpan di `riwayat_actual`.")

    stats_actual = hitung_statistik_riwayat_actual(st.session_state.get('riwayat_actual', {}))
    
    if stats_actual and stats_actual['total_eval'] > 0:
        c_wr1, c_wr2, c_wr3, c_wr4 = st.columns(4)
        
        wr_val = stats_actual['win_rate']
        c_wr1.metric(
            "System Win Rate",
            f"{wr_val:.1f}%",
            delta=f"{stats_actual['total_eval']} Sinyal Ter-evaluasi",
            delta_color="normal" if wr_val >= 50 else "inverse"
        )
        
        c_wr2.metric(
            "Hasil Evaluation",
            f"🟢 {stats_actual['total_win']} Win / 🔴 {stats_actual['total_loss']} Loss",
            delta=f"⚪ {stats_actual['total_not_touched']} Not Touched" if stats_actual['total_not_touched'] > 0 else None,
            delta_color="off"
        )
        
        sw_text = f"{stats_actual['wr_sw']:.1f}% ({stats_actual['eval_sw']} trade)" if stats_actual['wr_sw'] is not None else "Belum ada data"
        c_wr3.metric(
            "Swing Trade Win Rate",
            sw_text
        )
        
        dt_text = f"{stats_actual['wr_dt']:.1f}% ({stats_actual['eval_dt']} trade)" if stats_actual['wr_dt'] is not None else "Belum ada data"
        c_wr4.metric(
            "Day Trade Win Rate",
            dt_text
        )
    else:
        st.info("ℹ️ Belum ada data outcome riil di `riwayat_actual`. Isi **Form Evaluasi Sinyal (Outcome Journal)** setelah sinyal berjalan untuk merekam Win Rate live engine.")

    st.markdown("---")
    st.subheader("📈 Informasi Pasar Terkini (IHSG)")

    periode_pilihan = st.selectbox(
        "Periode data IHSG:",
        options=["1d", "5d", "1mo"],
        format_func=lambda x: {"1d": "1 Hari", "5d": "5 Hari", "1mo": "1 Bulan"}[x],
        index=0,
        key="ihsg_period"
    )

    if periode_pilihan == "1d":
        interval_candidates = ["1m", "5m", "15m", "30m", "60m", "1d"]
    elif periode_pilihan == "5d":
        interval_candidates = ["5m", "15m", "30m", "60m", "1d"]
    else:
        interval_candidates = ["1d"]

    df_ihsg_preview = pd.DataFrame()
    interval_terpakai = None

    for interval in interval_candidates:
        temp_df = load_ihsg_data(period=periode_pilihan, interval=interval)
        if not temp_df.empty and len(temp_df) >= 2:
            df_ihsg_preview = temp_df
            interval_terpakai = interval
            break
        elif not temp_df.empty and len(temp_df) == 1 and interval == interval_candidates[-1]:
            df_ihsg_preview = temp_df
            interval_terpakai = interval
            break

    try:
        try:
            ihsg_info = yf.Ticker("^JKSE").info
            prev_close = ihsg_info.get('previousClose', None)
            open_price = ihsg_info.get('regularMarketOpen', None)
        except:
            prev_close = None
            open_price = None

        if not df_ihsg_preview.empty and len(df_ihsg_preview) >= 2:
            ihsg_close = float(df_ihsg_preview['Close'].iloc[-1])
            open_period = float(df_ihsg_preview['Open'].iloc[0])

            if periode_pilihan == "1d":
                if prev_close is not None and prev_close > 0:
                    ihsg_change = (ihsg_close - prev_close) / prev_close * 100
                else:
                    ihsg_prev = float(df_ihsg_preview['Close'].iloc[-2])
                    ihsg_change = (ihsg_close - ihsg_prev) / ihsg_prev * 100
            else:
                if open_period > 0:
                    ihsg_change = (ihsg_close - open_period) / open_period * 100
                else:
                    ihsg_change = 0.0

            ihsg_high = float(df_ihsg_preview['High'].max())
            ihsg_low = float(df_ihsg_preview['Low'].min())
            if open_price is None or open_price == 0:
                open_price = open_period

            if interval_terpakai in ("1m", "5m", "15m", "30m", "60m"):
                vol_val = df_ihsg_preview['Volume'].sum()
            else:
                vol_val = float(df_ihsg_preview['Volume'].iloc[-1])
            volume_str = f"{vol_val:,.0f}" if vol_val > 0 else "N/A"

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("IHSG", f"{ihsg_close:,.0f}", f"{ihsg_change:+.2f}%")
            col2.metric("Open", f"{open_price:,.0f}" if open_price else "N/A")
            col3.metric("High", f"{ihsg_high:,.0f}")
            col4.metric("Low", f"{ihsg_low:,.0f}")
            col5.metric("Volume", volume_str)

            if PLOTLY_AVAILABLE:
                line_color = '#26a69a' if ihsg_change >= 0 else '#ef5350'
                area_color = f"rgba({38 if ihsg_change >= 0 else 239}, {166 if ihsg_change >= 0 else 83}, {154 if ihsg_change >= 0 else 80}, 0.25)"

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ihsg_preview.index,
                    y=df_ihsg_preview['Close'],
                    mode='lines',
                    line=dict(color=line_color, width=1.5),
                    fill='tozeroy',
                    fillcolor=area_color,
                    name='IHSG',
                    hovertemplate='<b>%{x|%d %b %H:%M WIB}</b><br>Close: %{y:,.0f}<extra></extra>'
                ))
                fig.add_hline(y=ihsg_high, line_dash='dot', line_color='rgba(255,255,255,0.4)')
                fig.add_annotation(x=0.5, y=ihsg_high, xref='paper', yref='y',
                                   text=f'H {ihsg_high:,.0f}', showarrow=False,
                                   font=dict(size=9, color='rgba(255,255,255,0.6)'),
                                   bgcolor='rgba(15, 17, 22, 0.7)',
                                   bordercolor='rgba(255,255,255,0.3)',
                                   borderwidth=1, borderpad=4, xanchor='center', yanchor='bottom')
                fig.add_hline(y=ihsg_low, line_dash='dot', line_color='rgba(255,255,255,0.4)')
                fig.add_annotation(x=0.5, y=ihsg_low, xref='paper', yref='y',
                                   text=f'L {ihsg_low:,.0f}', showarrow=False,
                                   font=dict(size=9, color='rgba(255,255,255,0.6)'),
                                   bgcolor='rgba(15, 17, 22, 0.7)',
                                   bordercolor='rgba(255,255,255,0.3)',
                                   borderwidth=1, borderpad=4, xanchor='center', yanchor='bottom')
                y_min = float(df_ihsg_preview['Low'].min()) * 0.998
                y_max = float(df_ihsg_preview['High'].max()) * 1.002
                fig.update_yaxes(range=[y_min, y_max])

                chart_title = {
                    "1d": "IHSG Hari Ini",
                    "5d": "IHSG 5 Hari Terakhir",
                    "1mo": "IHSG 1 Bulan Terakhir"
                }.get(periode_pilihan, "IHSG")

                fig.update_layout(
                    title=dict(text=chart_title, x=0.01, xanchor='left', font=dict(size=14, color='#e0e0e0')),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=20, t=40, b=10),
                    dragmode='pan',
                    xaxis=dict(title=None, showgrid=False, zeroline=False, showline=True, linecolor='rgba(128,128,128,0.2)'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='rgba(128,128,128,0.1)', zeroline=False, side='right'),
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor='#1e293b', font_size=11, font_family="monospace"),
                    paper_bgcolor='#0f1116',
                    plot_bgcolor='#0f1116',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={
                    'modeBarButtonSize': 4,
                    'displaylogo': False
                })
                if interval_terpakai != "1m" and periode_pilihan == "1d":
                    st.info("ℹ️ Data 1 menit tidak tersedia, menggunakan interval yang lebih besar.")
            else:
                st.line_chart(df_ihsg_preview['Close'])
        elif not df_ihsg_preview.empty and len(df_ihsg_preview) == 1:
            ihsg_close = float(df_ihsg_preview['Close'].iloc[-1])
            if prev_close:
                ihsg_change = (ihsg_close - prev_close) / prev_close * 100
                st.metric("IHSG", f"{ihsg_close:,.0f}", f"{ihsg_change:+.2f}%")
            else:
                st.metric("IHSG", f"{ihsg_close:,.0f}")
            st.warning("Data IHSG hanya tersedia 1 titik (kemungkinan di luar jam bursa).")
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ihsg_preview.index,
                    y=df_ihsg_preview['Close'],
                    mode='lines+markers',
                    marker=dict(color='#f59e0b', size=8),
                    line=dict(color='#f59e0b', width=2),
                    name='IHSG'
                ))
                fig.update_layout(title="IHSG (Data Terbatas)", template="plotly_dark", height=350, dragmode='pan')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_ihsg_preview['Close'])
        else:
            st.warning("Data IHSG tidak tersedia untuk periode yang dipilih.")
    except Exception as e:
        st.error(f"Gagal memuat data IHSG: {e}")
        
# --- ANALISIS RIWAYAT DENGAN AI (TOMBOL SIDEBAR) ---
if ai_riwayat_btn:
    if not st.session_state.gemini_api_key: st.error("Masukkan API Key terlebih dahulu!")
    elif not st.session_state.riwayat: st.warning("Belum ada riwayat.")
    else:
        with st.spinner("🧠 AI menganalisis riwayat..."):
            hasil, error = analisis_riwayat_global(
                st.session_state.riwayat,
                st.session_state.riwayat_actual,
                st.session_state.gemini_api_key
            )
            if error:
                st.error(error)
            elif hasil:
                hasil_bersih = bersihkan_teks_ai(hasil)
                st.markdown(
                    f'<div class="ai-insight-card" style="border-left-color:#06b6d4;">'
                    f'<h3 style="color:#67e8f9;">📊 Insight AI dari Riwayat</h3>'
                    f'<p>{hasil_bersih}</p></div>',
                    unsafe_allow_html=True
                )
