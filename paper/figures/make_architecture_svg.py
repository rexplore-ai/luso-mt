"""luso-mt architecture figure (SVG), laid out with explicit spacing so no line touches an arrowhead or box.

Rules: residual branches leave the main path in their own gap (marked with a dot) at least 20 px from any
arrowhead; arrowheads stop 3 px short of boxes and circles; vertical residual runs never overlap.
"""
import sys

OUT = sys.argv[1]
O = 112          # vertical offset (room for the decoder output at the top)
W, H = 1040, 1240
y = lambda v: v + O
parts = []
add = parts.append

def rect(cls, x, y0, w, h, rx=6):
    add(f'<rect class="{cls}" x="{x}" y="{y(y0)}" width="{w}" height="{h}" rx="{rx}"/>')

def text(cls, x, y0, s, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    add(f'<text class="{cls}" x="{x}" y="{y(y0)}"{a}>{s}</text>')

def box(cls, cx, y0, w, h, title, sub=None):
    rect(cls, cx - w / 2, y0, w, h)
    if sub:
        text("t", cx, y0 + h / 2 - 2, title); text("s", cx, y0 + h / 2 + 15, sub)
    else:
        text("t", cx, y0 + h / 2 + 5, title)

def arrow(cx, y_from, y_to, cls="line", m="a"):
    add(f'<path class="{cls}" d="M{cx},{y(y_from)} V{y(y_to)}" marker-end="url(#{m})"/>')

def plus(cx, cy):
    add(f'<circle class="add" cx="{cx}" cy="{y(cy)}" r="12"/>')
    add(f'<path class="plus" d="M{cx-7},{y(cy)} H{cx+7} M{cx},{y(cy)-7} V{y(cy)+7}"/>')

def residual(cx, y_branch, x_col, y_plus):
    add(f'<circle class="dot" cx="{cx}" cy="{y(y_branch)}" r="3.4"/>')
    add(f'<path class="res" d="M{cx+4},{y(y_branch)} H{x_col} V{y(y_plus)} H{cx+15}" marker-end="url(#ar)"/>')

def callout(cx, cy, n):
    add(f'<circle class="c" cx="{cx}" cy="{y(cy)}" r="13"/><text class="cn" x="{cx}" y="{y(cy)+4}">{n}</text>')

# ---------------- encoder ----------------
EX, ER = 280, 444            # main-path x, residual column x
text("lab", 112, 436, "Encoder")
for dx, cls in ((12, "stack2"), (6, "stack2"), (0, "stack")):
    rect(cls, 106 + dx, 450 + dx, 362, 430, 14)
text("rep", 56, 650, "N×", "middle"); text("sub", 56, 674, "tiny 8", "middle"); text("sub", 56, 690, "small 12", "middle")
text("sub", 56, 710, "(deep)", "middle")
box("emb", EX, 900, 260, 40, "Token embedding E")
residual(EX, 860, ER, 696)
box("norm", EX, 800, 260, 32, "RMSNorm")
box("attn", EX, 724, 260, 54, "Multi-head self-attention", "relative position buckets · bidirectional")
plus(EX, 696)
residual(EX, 660, ER, 490)
box("norm", EX, 598, 260, 32, "RMSNorm")
box("ffn", EX, 516, 260, 54, "Gated-GELU feed-forward", "GELU(xW₀) ⊙ (xW₁) · W₂")
plus(EX, 490)
arrow(EX, 900, 835); arrow(EX, 800, 781); arrow(EX, 724, 711)
arrow(EX, 684, 633); arrow(EX, 598, 573); arrow(EX, 516, 505)
arrow(EX, 478, 415)
box("norm", EX, 380, 180, 32, "Final RMSNorm")
text("sub", EX + 14, 368, "encoder states h₁…hₙ")
# input side
box("tok", EX, 966, 310, 44, "SentencePiece unigram", "one vocabulary for en · ny · tum (16k tiny, 32k small)")
arrow(EX, 966, 943)
rect("io", EX - 180, 1040, 360, 40, 8)
add(f'<text class="m" x="{EX}" y="{y(1065)}">&lt;2ny&gt; Where is the nearest clinic?</text>')
arrow(EX, 1040, 1013)
text("sub", EX, 1104, "source sentence with target-language tag", "middle")

# ---------------- decoder ----------------
DX, DR = 720, 884
text("lab", 552, 230, "Decoder")
for dx, cls in ((6, "stack2"), (0, "stack")):
    rect(cls, 546 + dx, 244 + dx, 362, 636, 14)
text("rep", 962, 560, "N×", "middle"); text("sub", 962, 584, "tiny 2", "middle"); text("sub", 962, 600, "small 4", "middle")
text("sub", 962, 620, "(shallow)", "middle")
box("emb", DX, 900, 260, 40, "Token embedding E")
residual(DX, 860, DR, 696)
box("norm", DX, 800, 260, 32, "RMSNorm")
box("attn", DX, 724, 260, 54, "Causal self-attention", "sees only earlier output tokens")
plus(DX, 696)
residual(DX, 660, DR, 490)
box("norm", DX, 598, 260, 32, "RMSNorm")
box("xattn", DX, 516, 260, 54, "Cross-attention", "queries: decoder · keys, values: encoder")
plus(DX, 490)
residual(DX, 454, DR, 284)
box("norm", DX, 392, 260, 32, "RMSNorm")
box("ffn", DX, 310, 260, 54, "Gated-GELU feed-forward", "same form as the encoder")
plus(DX, 284)
arrow(DX, 900, 835); arrow(DX, 800, 781); arrow(DX, 724, 711)
arrow(DX, 684, 633); arrow(DX, 598, 573); arrow(DX, 516, 505)
arrow(DX, 478, 427); arrow(DX, 392, 367); arrow(DX, 310, 299)
arrow(DX, 272, 211)
box("norm", DX, 176, 180, 32, "Final RMSNorm")
arrow(DX, 176, 147)
box("emb", DX, 104, 240, 40, "Linear · Eᵀ (tied)")
arrow(DX, 104, 75)
box("out", DX, 40, 190, 32, "Softmax · beam search")
arrow(DX, 40, 11)
rect("io", DX - 175, -32, 350, 40, 8)
add(f'<text class="m" x="{DX}" y="{y(-7)}">Chipatala chapafupi chili kuti?</text>')
text("sub", DX, -48, "output, one token at a time", "middle")
box("tok", DX, 966, 310, 44, "Same vocabulary", "tokens generated so far (shifted right)")
arrow(DX, 966, 943)
rect("io", DX - 175, 1040, 350, 40, 8)
add(f'<text class="m" x="{DX}" y="{y(1065)}">&lt;s&gt; Chipatala chapafupi …</text>')
arrow(DX, 1040, 1013)

# encoder states -> cross-attention (K, V)
add(f'<path class="kv" d="M{EX},{y(380)} V{y(334)} H508 V{y(543)} H{DX-133}" marker-end="url(#ak)"/>')
text("kvt", 516, 440, "K, V")
# tied embedding matrix
add(f'<path class="tie" d="M{EX+130},{y(920)} H{DX-130}"/>')
text("tiet", 500, 912, "one matrix E", "middle")
add(f'<path class="tie" d="M{DX+120},{y(124)} H1010 V{y(920)} H{DX+130}"/>')
text("tiet", 925, 116, "reuses E", "middle")

for args in ((84, 1060, 1), (114, 988, 2), (128, 920, 3), (128, 816, 4), (128, 751, 5), (128, 543, 6),
             (56, 618, 7), (962, 528, 8), (598, 516, 9), (582, 124, 10)):
    callout(*args)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d" font-family="Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif">
  <title id="t">luso-mt architecture</title>
  <desc id="d">Encoder-decoder Transformer with a deep encoder and a shallow decoder, RMSNorm pre-normalisation, gated-GELU feed-forward layers, relative position attention, a shared SentencePiece vocabulary, target-language tags and tied embeddings.</desc>
  <style>
    .bg {{ fill: #ffffff; }}
    .stack {{ fill: #f3f6fa; stroke: #8fa0b5; stroke-width: 1.5; }}
    .stack2 {{ fill: #e9eef5; stroke: #b6c2d1; stroke-width: 1.2; }}
    .norm {{ fill: #fff3c4; stroke: #c9a227; stroke-width: 1.4; }}
    .attn {{ fill: #fde3cf; stroke: #e07b39; stroke-width: 1.4; }}
    .xattn {{ fill: #fbd0b0; stroke: #c9581b; stroke-width: 1.6; }}
    .ffn {{ fill: #d6ecfa; stroke: #008fdb; stroke-width: 1.4; }}
    .emb {{ fill: #ece4f7; stroke: #8a63c7; stroke-width: 1.4; }}
    .tok {{ fill: #eef1f4; stroke: #8a96a6; stroke-width: 1.2; }}
    .out {{ fill: #e3f3ea; stroke: #2f9e6a; stroke-width: 1.4; }}
    .io {{ fill: #ffffff; stroke: #041023; stroke-width: 1.2; }}
    .line {{ fill: none; stroke: #041023; stroke-width: 1.6; }}
    .res {{ fill: none; stroke: #5b6b86; stroke-width: 1.4; }}
    .dot {{ fill: #5b6b86; }}
    .kv {{ fill: none; stroke: #c9581b; stroke-width: 2; }}
    .tie {{ fill: none; stroke: #8a63c7; stroke-width: 1.4; stroke-dasharray: 6 5; }}
    .add {{ fill: #ffffff; stroke: #041023; stroke-width: 1.5; }}
    .t {{ fill: #041023; font-size: 15px; font-weight: 600; text-anchor: middle; }}
    .s {{ fill: #3d4a5c; font-size: 11.5px; text-anchor: middle; }}
    .m {{ fill: #041023; font-size: 13px; text-anchor: middle; font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace; }}
    .lab {{ fill: #041023; font-size: 17px; font-weight: 700; }}
    .rep {{ fill: #041023; font-size: 22px; font-weight: 700; }}
    .sub {{ fill: #5b6b86; font-size: 12px; }}
    .kvt {{ fill: #c9581b; font-size: 12.5px; font-weight: 700; }}
    .tiet {{ fill: #8a63c7; font-size: 12px; font-weight: 600; }}
    .c {{ fill: #008fdb; }}
    .cn {{ fill: #ffffff; font-size: 12px; font-weight: 700; text-anchor: middle; }}
    .plus {{ stroke: #041023; stroke-width: 1.6; }}
  </style>
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#041023"/></marker>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5b6b86"/></marker>
    <marker id="ak" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c9581b"/></marker>
  </defs>
  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>
  {chr(10).join("  " + p for p in parts)}
</svg>
'''
open(OUT, "w", encoding="utf-8", newline="\n").write(svg)
print("wrote", OUT)
