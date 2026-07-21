import json

# ---------- Table 3 + Table 4 (MMLU, overall IR) for Evaluation Space scatter ----------
models = [
    ("GPT-4.1", 90.1, 8.8), ("GPT-4o", 88.7, 8.9), ("Claude 4.5 Opus", 88.1, 4.2),
    ("Claude 4.5 Sonnet", 85.4, 5.9), ("Grok-3", 87.5, 8.1), ("o4-mini", 89.4, 8.0),
    ("o3-mini", 88.2, 5.4), ("DeepSeek R1", 86.1, 10.9), ("Gemini 2.0 Flash", 82.3, 9.5),
    ("GPT-4o mini", 82.0, 14.2), ("Llama 4 Maverick", 83.1, 12.2), ("Llama 3.3 70B", 83.7, 11.4),
    ("DeepSeek-V3", 84.0, 9.1), ("Qwen3-235B", 85.2, 9.3), ("Mistral Large 2411", 82.6, 10.2),
    ("Phi-4", 78.9, 10.5),
]
reasoning_models = {"o4-mini", "o3-mini", "DeepSeek R1"}

# chart area
W, H = 720, 480
PAD_L, PAD_R, PAD_T, PAD_B = 56, 20, 20, 46
plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
x_min, x_max = 76, 92
y_min, y_max = 82, 98  # consistency score = 100 - IR

def xmap(v):
    return PAD_L + (v - x_min) / (x_max - x_min) * plot_w

def ymap(v):
    return PAD_T + (1 - (v - y_min) / (y_max - y_min)) * plot_h

circles = []
labels = []
for name, mmlu, ir in models:
    cy = 100 - ir
    x, y = xmap(mmlu), ymap(cy)
    shape = "triangle" if name in reasoning_models else "circle"
    circles.append((name, x, y, shape))

svg_points = []
for name, x, y, shape in circles:
    cls = "pt-reason" if shape == "triangle" else "pt-std"
    if shape == "triangle":
        r = 6.5
        pts = f"{x:.1f},{y-r:.1f} {x-r:.1f},{y+r*0.8:.1f} {x+r:.1f},{y+r*0.8:.1f}"
        svg_points.append(f'<polygon class="{cls}" points="{pts}" data-name="{name}"/>')
    else:
        svg_points.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5.8" data-name="{name}"/>')

svg_labels = []
# manual small offsets to reduce overlap, keyed by name
offsets = {
    "GPT-4.1": (7, -6), "GPT-4o": (7, 10), "Claude 4.5 Opus": (-58, -8), "Claude 4.5 Sonnet": (-70, 4),
    "Grok-3": (7, 4), "o4-mini": (7, -6), "o3-mini": (-52, -8), "DeepSeek R1": (7, 12),
    "Gemini 2.0 Flash": (7, 4), "GPT-4o mini": (-30, 16), "Llama 4 Maverick": (-72, 4),
    "Llama 3.3 70B": (7, 4), "DeepSeek-V3": (7, -8), "Qwen3-235B": (-64, -8),
    "Mistral Large 2411": (-40, 16), "Phi-4": (7, 4),
}
for name, x, y, shape in circles:
    dx, dy = offsets.get(name, (7, 4))
    svg_labels.append(f'<text class="pt-label" x="{x+dx:.1f}" y="{y+dy:.1f}">{name}</text>')

# axes / gridlines
xticks = [76, 80, 84, 88, 92]
yticks = [82, 86, 90, 94, 98]
grid = []
for xt in xticks:
    gx = xmap(xt)
    grid.append(f'<line class="grid" x1="{gx:.1f}" y1="{PAD_T}" x2="{gx:.1f}" y2="{H-PAD_B}"/>')
    grid.append(f'<text class="axis-lbl" x="{gx:.1f}" y="{H-PAD_B+18}" text-anchor="middle">{xt}</text>')
for yt in yticks:
    gy = ymap(yt)
    grid.append(f'<line class="grid" x1="{PAD_L}" y1="{gy:.1f}" x2="{W-PAD_R}" y2="{gy:.1f}"/>')
    grid.append(f'<text class="axis-lbl" x="{PAD_L-10}" y="{gy+4:.1f}" text-anchor="end">{yt}</text>')

evalspace_svg = f'''<svg viewBox="0 0 {W} {H}" class="chart-svg" role="img" aria-label="MMLU accuracy versus consistency score across 16 models">
{''.join(grid)}
<rect x="{PAD_L}" y="{PAD_T}" width="{plot_w}" height="{plot_h}" fill="none" stroke="var(--rule)" stroke-width="1"/>
{''.join(svg_points)}
{''.join(svg_labels)}
<text class="axis-title" x="{PAD_L + plot_w/2}" y="{H-6}" text-anchor="middle">MMLU accuracy (%)</text>
<text class="axis-title" transform="rotate(-90)" x="{-(PAD_T + plot_h/2)}" y="16" text-anchor="middle">Consistency score  (100 &#8722; IR%)</text>
</svg>'''

# ---------- Table 4 average row: family difficulty ranking ----------
fam_avg = [("Equivalence", 12.8), ("Reversal", 11.8), ("Complement", 7.0), ("Ordering", 2.3), ("Composition", 1.5)]
FW, FH = 640, 260
fpad_l, fpad_r, fpad_t, fpad_b = 130, 50, 16, 30
fplot_w = FW - fpad_l - fpad_r
fplot_h = FH - fpad_t - fpad_b
fmax = 14
bar_h = fplot_h / len(fam_avg) * 0.55
gap = fplot_h / len(fam_avg)
fam_bars = []
for i, (name, val) in enumerate(fam_avg):
    y = fpad_t + i * gap + (gap - bar_h) / 2
    w = val / fmax * fplot_w
    fam_bars.append(f'<text class="fam-lbl" x="{fpad_l-12}" y="{y+bar_h*0.72:.1f}" text-anchor="end">{name}</text>')
    fam_bars.append(f'<rect class="fam-bar fam-{name.lower()}" x="{fpad_l}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="3"/>')
    fam_bars.append(f'<text class="fam-val" x="{fpad_l+w+8:.1f}" y="{y+bar_h*0.72:.1f}">{val}%</text>')
family_bar_svg = f'''<svg viewBox="0 0 {FW} {FH}" class="chart-svg" role="img" aria-label="Mean inconsistency rate by transformation family">
{''.join(fam_bars)}
<line x1="{fpad_l}" y1="{fpad_t}" x2="{fpad_l}" y2="{FH-fpad_b}" stroke="var(--rule)"/>
</svg>'''

# ---------- Table 7: IR vs CCS scatter ----------
irccs = [
    ("GPT-4.1", 8.8, 0.2156), ("GPT-4o", 8.9, 0.2173), ("GLM-5.2", 5.7, 0.2230), ("o4-mini", 8.0, 0.2197),
    ("o3-mini", 5.4, 0.2178), ("DeepSeek R1", 10.9, 0.2171), ("Gemini 2.0 Flash", 9.5, 0.2185),
    ("GPT-4o mini", 14.2, 0.2148), ("Llama 4 Maverick", 12.2, 0.2187), ("Llama 3.3 70B", 11.4, 0.2131),
    ("DeepSeek-V3", 9.1, 0.2165), ("Qwen3-235B", 9.3, 0.2178), ("Mistral Large 2411", 10.2, 0.2155),
    ("Phi-4", 10.5, 0.2179), ("Grok-3", 8.1, 0.2160), ("Claude 4.5 Sonnet", 5.9, 0.2110),
    ("Claude 4.5 Opus", 4.2, 0.2050),
]
CW, CH = 640, 420
cpad_l, cpad_r, cpad_t, cpad_b = 60, 20, 16, 40
cplot_w, cplot_h = CW - cpad_l - cpad_r, CH - cpad_t - cpad_b
cx_min, cx_max = 3, 15
cy_min, cy_max = 0.204, 0.224
def cxmap(v): return cpad_l + (v - cx_min) / (cx_max - cx_min) * cplot_w
def cymap(v): return cpad_t + (1 - (v - cy_min) / (cy_max - cy_min)) * cplot_h
cc_pts = []
for name, ir, ccs in irccs:
    x, y = cxmap(ir), cymap(ccs)
    cc_pts.append(f'<circle class="cc-pt" cx="{x:.1f}" cy="{y:.1f}" r="5.5" data-name="{name}" data-ir="{ir}" data-ccs="{ccs}"/>')
cc_grid = []
for xt in [4,6,8,10,12,14]:
    gx = cxmap(xt)
    cc_grid.append(f'<line class="grid" x1="{gx:.1f}" y1="{cpad_t}" x2="{gx:.1f}" y2="{CH-cpad_b}"/>')
    cc_grid.append(f'<text class="axis-lbl" x="{gx:.1f}" y="{CH-cpad_b+18}" text-anchor="middle">{xt}</text>')
for yt in [0.205,0.210,0.215,0.220]:
    gy = cymap(yt)
    cc_grid.append(f'<line class="grid" x1="{cpad_l}" y1="{gy:.1f}" x2="{CW-cpad_r}" y2="{gy:.1f}"/>')
    cc_grid.append(f'<text class="axis-lbl" x="{cpad_l-8}" y="{gy+4:.1f}" text-anchor="end">{yt:.3f}</text>')
irccs_svg = f'''<svg viewBox="0 0 {CW} {CH}" class="chart-svg" role="img" aria-label="Inconsistency rate versus consistency calibration score across 17 models">
{''.join(cc_grid)}
<rect x="{cpad_l}" y="{cpad_t}" width="{cplot_w}" height="{cplot_h}" fill="none" stroke="var(--rule)"/>
{''.join(cc_pts)}
<text class="axis-title" x="{cpad_l+cplot_w/2}" y="{CH-4}" text-anchor="middle">Inconsistency rate — IR (%)</text>
<text class="axis-title" transform="rotate(-90)" x="{-(cpad_t+cplot_h/2)}" y="14" text-anchor="middle">CCS (lower = better calibrated)</text>
</svg>'''

# ---------- Finding 4: reasoning training diverging bars ----------
fams4 = ["Composition", "Reversal", "Complement", "Ordering"]
r1_v3 = [-0.8, 0.9, 2.5, 3.4]
o4_g4o = [3.1, -7.6, -3.9, -13.3]
DW, DH = 640, 320
dpad_l, dpad_r, dpad_t, dpad_b = 40, 20, 16, 60
dplot_w, dplot_h = DW - dpad_l - dpad_r, DH - dpad_t - dpad_b
dmax = 14
def dymap(v):
    return dpad_t + dplot_h/2 - (v/dmax)*(dplot_h/2)
zero_y = dpad_t + dplot_h/2
group_w = dplot_w / len(fams4)
bar_w = group_w * 0.28
bars4 = []
for i, fam in enumerate(fams4):
    gx = dpad_l + i*group_w + group_w/2
    v1, v2 = r1_v3[i], o4_g4o[i]
    x1, x2 = gx - bar_w - 3, gx + 3
    y1, h1 = (dymap(v1), zero_y-dymap(v1)) if v1>=0 else (zero_y, dymap(v1)-zero_y)
    y2, h2 = (dymap(v2), zero_y-dymap(v2)) if v2>=0 else (zero_y, dymap(v2)-zero_y)
    bars4.append(f'<rect class="div-bar div-r1" x="{x1:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{abs(h1):.1f}"/>')
    bars4.append(f'<rect class="div-bar div-o4" x="{x2:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{abs(h2):.1f}"/>')
    bars4.append(f'<text class="axis-lbl" x="{gx:.1f}" y="{DH-dpad_b+18}" text-anchor="middle">{fam}</text>')
    for v,x in [(v1,x1+bar_w/2),(v2,x2+bar_w/2)]:
        ly = dymap(v) - 6 if v>=0 else dymap(v)+14
        bars4.append(f'<text class="div-val" x="{x:.1f}" y="{ly:.1f}" text-anchor="middle">{v:+.1f}</text>')
finding4_svg = f'''<svg viewBox="0 0 {DW} {DH}" class="chart-svg" role="img" aria-label="Change in inconsistency rate for reasoning-trained models versus base models by family">
<line x1="{dpad_l}" y1="{zero_y:.1f}" x2="{DW-dpad_r}" y2="{zero_y:.1f}" class="grid" stroke-width="1.4"/>
{''.join(bars4)}
</svg>'''

# ---------- Table 6: Intervention grouped bars ----------
interv = [
    ("DeepSeek R1", [6.9, 5.9, 1.5, 4.5]),
    ("GPT-4o", [5.6, 5.8, 5.7, 5.9]),
    ("Llama 4 Maverick", [13.8, 10.6, 5.7, 9.5]),
]
conds = ["Baseline", "+CR", "+SC", "+FTSC"]
IW, IH = 660, 340
ipad_l, ipad_r, ipad_t, ipad_b = 40, 20, 16, 56
iplot_w, iplot_h = IW-ipad_l-ipad_r, IH-ipad_t-ipad_b
imax = 14
group_w2 = iplot_w/len(interv)
bar_w2 = group_w2/len(conds)*0.72
ibars = []
for gi, (name, vals) in enumerate(interv):
    gx0 = ipad_l + gi*group_w2 + (group_w2 - bar_w2*len(conds))/2
    for ci, v in enumerate(vals):
        h = v/imax*iplot_h
        x = gx0 + ci*bar_w2
        y = ipad_t + iplot_h - h
        ibars.append(f'<rect class="interv-bar interv-{ci}" x="{x:.1f}" y="{y:.1f}" width="{bar_w2*0.86:.1f}" height="{h:.1f}" rx="2"/>')
        ibars.append(f'<text class="interv-val" x="{x+bar_w2*0.43:.1f}" y="{y-5:.1f}" text-anchor="middle">{v}</text>')
    lx = ipad_l + gi*group_w2 + group_w2/2
    ibars.append(f'<text class="axis-lbl" x="{lx:.1f}" y="{IH-ipad_b+22:.1f}" text-anchor="middle">{name}</text>')
intervention_svg = f'''<svg viewBox="0 0 {IW} {IH}" class="chart-svg" role="img" aria-label="Inconsistency rate under four prompting conditions for three models">
<line x1="{ipad_l}" y1="{ipad_t+iplot_h}" x2="{IW-ipad_r}" y2="{ipad_t+iplot_h}" class="grid" stroke-width="1.4"/>
{''.join(ibars)}
</svg>'''

out = {
    "evalspace_svg": evalspace_svg,
    "family_bar_svg": family_bar_svg,
    "irccs_svg": irccs_svg,
    "finding4_svg": finding4_svg,
    "intervention_svg": intervention_svg,
}
with open("/home/claude/site_build/charts.json", "w") as f:
    json.dump(out, f)
print("done", {k: len(v) for k,v in out.items()})
