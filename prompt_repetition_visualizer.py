"""
Prompt Repetition Attention Visualizer
Based on: "Prompt Repetition Improves Non-Reasoning LLMs" (Leviathan et al., 2025)

Extracts and visualizes real attention weights from GPT-2 for a single prompt
vs. the repeated prompt [prompt | prompt], showing how repetition gives
the model effective bidirectional context within a causal architecture.
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

# Chart colormaps.
CMAP_SINGLE = LinearSegmentedColormap.from_list("single", ["#0d1117", "#1a2744", "#1e4d8c", "#4a90d9", "#7fc8f8", "#ffffff"])
CMAP_COPY   = LinearSegmentedColormap.from_list("copy",   ["#0d1117", "#25163d", "#49237e", "#8d52c5", "#c793fb", "#ffffff"])
CMAP_DIFF   = LinearSegmentedColormap.from_list("diff",   ["#8b0000", "#cc3300", "#0d1117", "#1a6b2e", "#3aad50"])

# Curated prompt scenarios highlighting different model weaknesses addressed by prompt repetition.
PRESETS = {
    "🔘 Options First (weak spot)": {
        "prompt": "(A) water (B) air (C) salt (D) ammonia Which is a mix?",
        "description": (
            "Options appear <b>before</b> the question. In the baseline, option tokens are "
            "processed before the question exists - a known weakness of causal LLMs. "
            "With repetition, copy-2 option tokens can attend back to the question in copy-1."
        ),
    },
    "❔ Question First": {
        "prompt": "Which is a mix? (A) water (B) air (C) salt (D) ammonia",
        "description": (
            "Question appears <b>before</b> options. The baseline already lets options attend to "
            "the question. Repetition allows question tokens in copy-2 to re-examine options "
            "through copy-1, potentially reinforcing relevant connections."
        ),
    },
    "🪡 Needle-in-Haystack": {
        "prompt": "Emily lives in Paris. Gabriel is tall. Where does Emily live?",
        "description": (
            "Relevant fact buried in context, question at the end. With repetition, the question "
            "tokens in copy-2 directly attend to 'Emily' and 'Paris' in copy-1 without relying "
            "on information cascading through intervening tokens."
        ),
    },
    "🗂 List Index Lookup": {
        "prompt": "Peter Raul Tiffany Chris Joao Ema Henry 5th name?",
        "description": (
            "The NameIndex benchmark: find the Nth item in a list."
            "Easier with repetition since '5th' in copy-2 can directly attend to list items in copy-1"
            "while the baseline must rely on less direct pathways through the sequence."
        ),
    },
}

DEFAULT_PROMPT = list(PRESETS.values())[0]["prompt"]

# Module-level singletons - avoids reloading 500 MB weights on every inference call.
_model: GPT2Model | None = None
_tokenizer: GPT2Tokenizer | None = None


def load_model() -> tuple[GPT2Tokenizer, GPT2Model]:
    global _model, _tokenizer
    if _model is None:
        print("Loading GPT-2 small (~500 MB, one-time download)…")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
        _model.eval()
        print("GPT-2 ready.")
    return _tokenizer, _model


def clean_token(tok: str) -> str:
    return tok.replace("Ġ", " ").replace("Ċ", "\\n").strip() or tok


def get_attentions(prompt: str, repeated: bool = False):
    tokenizer, model = load_model()
    enc = tokenizer(prompt, return_tensors="pt")
    # Define a hard cap: beyond 20 tokens axis labels overlap
    n = min(enc["input_ids"].shape[1], 20)

    input_ids = enc["input_ids"][:, :n]
    if repeated:
        # Build [prompt | prompt] by doubling the token sequence
        input_ids = torch.cat([input_ids, input_ids], dim=1)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)

    raw = tokenizer.convert_ids_to_tokens(enc["input_ids"][0, :n])
    tokens = [clean_token(t) for t in raw]
    return tokens, out.attentions, n


def aggregate_attentions(single_attns, rep_attns, n: int):
    """Average attention across all layers and heads. Returns three (n, n) arrays:
    agg_s (single prompt), agg_r1 (copy-2 → copy-1), agg_r2 (copy-2 → copy-2)."""
    
    agg_s = np.zeros((n, n))
    agg_r1 = np.zeros((n, n))
    agg_r2 = np.zeros((n, n))
    
    for ls, lr in zip(single_attns, rep_attns):
        
        agg_s += ls[0].mean(0).cpu().numpy()[:n, :n]
        rm = lr[0].mean(0).cpu().numpy()
        # copy-2 queries copy-1 keys: the new cross-copy attention channel
        agg_r1 += rm[n:2*n, :n]
        # copy-2 queries copy-2 keys: self-attention within the second copy
        agg_r2 += rm[n:2*n, n:2*n]
    
    nl = len(single_attns)
    return agg_s / nl, agg_r1 / nl, agg_r2 / nl


def make_heatmap_fig(prompt: str) -> Image.Image:
    tokens, single_attns, n = get_attentions(prompt, repeated=False)
    _, rep_attns, _ = get_attentions(prompt, repeated=True)

    attn_single, attn_rep_cross, attn_rep_self = aggregate_attentions(single_attns, rep_attns, n)

    # Row-normalization before taking the difference so both matrices sum to 1.0 per row.
    def row_norm(m):
        s = m.sum(axis=-1, keepdims=True)
        return m / np.where(s > 1e-10, s, 1.0)

    diff = row_norm(attn_rep_cross) - row_norm(attn_single)

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    
    # Add 2×3 grid: top row = (baseline | cross-copy | difference), bottom = (reference | self-copy | magnitude)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4, top=0.88, bottom=0.06, left=0.06, right=0.96)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    panel_data = [
        (attn_single, CMAP_SINGLE, "BASELINE: Single Prompt\nCausal (lower-triangular) attention", "#4a90d9"),
        (attn_rep_cross, CMAP_COPY, "REPEATED: Copy-2 → Copy-1\nNew cross-copy attention  (the key gain)","#9b59b6"),
        (diff, CMAP_DIFF, "DIFFERENCE: Row-normalized Repeated − Baseline\nGreen = gained | Red = shifted away","#e8a030"),
        (attn_single, CMAP_SINGLE, "Baseline attention  (reference)\nSame as top-left, for entropy comparison", "#4a90d9"),
        (attn_rep_self, CMAP_COPY, "REPEATED: Copy-2 → Copy-2\nSelf-attention within second copy","#9b59b6"),
        (np.abs(diff), CMAP_COPY, "MAGNITUDE of Change  (row-normalized)\nHighlights which token pairs shift most","#9b59b6"),
    ]

    for ax, (data, cmap, title, accent) in zip(axes, panel_data):
        ax.set_facecolor("#0d1117")
        vmax = max(float(np.abs(data).max()), 1e-6)
        vmin = -vmax if "DIFFERENCE" in title else 0
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.outline.set_edgecolor("#333344")

        tick_pos = list(range(n))
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8, color="#aac8f0", fontfamily="monospace")
        ax.set_yticklabels(tokens, fontsize=8, color="#aac8f0", fontfamily="monospace")
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.5)
        ax.set_title(title, color="white", fontsize=9, pad=8, fontfamily="monospace", fontweight="bold")
        ax.set_xlabel("Attended-to token (key)", color="#667788", fontsize=7)
        ax.set_ylabel("Attending token (query)", color="#667788", fontsize=7)
        ax.tick_params(axis="both", colors="#aac8f0", length=3)

    fig.text(0.5, 0.94, "Prompt Repetition: Real GPT-2 Attention Comparison",
             ha="center", fontsize=16, color="white", fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.915, "Averaged across all 12 layers × 12 heads  |  GPT-2 small",
             ha="center", fontsize=10, color="#667788", fontfamily="monospace")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0d1117")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def make_entropy_chart(prompt: str) -> Image.Image:
    tokens, single_attns, n = get_attentions(prompt, repeated=False)
    _, rep_attns, _ = get_attentions(prompt, repeated=True)
    agg_s, agg_r1, _ = aggregate_attentions(single_attns, rep_attns, n)

    # Row-normalization both to sum=1 so entropy/concentration are comparable.
    agg_s = agg_s / (agg_s.sum(axis=-1, keepdims=True) + 1e-10)
    agg_r1 = agg_r1 / (agg_r1.sum(axis=-1, keepdims=True) + 1e-10)

    def entropy(mat):
        p = np.clip(mat, 1e-10, 1)
        # Normalize to [0,1] using log(n) as max entropy.
        return -np.sum(p * np.log(p), axis=-1) / np.log(max(n, 2))

    def top_k_concentration(mat, k=3):
        return np.sort(mat, axis=-1)[:, ::-1][:, :k].sum(axis=-1)

    ent_s = entropy(agg_s)
    ent_r = entropy(agg_r1)
    conc_s = top_k_concentration(agg_s)
    conc_r = top_k_concentration(agg_r1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    x, w = np.arange(n), 0.38

    for ax, (ys, yr, ylabel, title) in zip(axes, [
        (ent_s,  ent_r,  "Normalized Entropy (0=focused, 1=diffuse)",
         "Attention Entropy per Token\nLower = more focused / confident attention"),
        (conc_s, conc_r, "Top-3 Token Attention Concentration",
         "Top-3 Attention Concentration\nHigher = attention focuses on fewer, more relevant tokens"),
    ]):
        ax.set_facecolor("#111122")
        ax.bar(x - w/2, ys, w, label="Baseline (single)", color="#4a90d9", alpha=0.85)
        ax.bar(x + w/2, yr, w, label="Repeated (copy-2 → copy-1)", color="#9b59b6", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", color="#aac8f0", fontfamily="monospace", fontsize=9)
        ax.tick_params(axis="y", colors="#aac8f0")
        ax.set_ylabel(ylabel, color="#667788", fontsize=9)
        ax.set_title(title, color="white", fontfamily="monospace", fontsize=10, pad=10)
        ax.legend(facecolor="#111122", edgecolor="#334455", labelcolor="white", fontsize=9)
        ax.spines["bottom"].set_color("#334455")
        ax.spines["left"].set_color("#334455")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_tick_params(labelcolor="#aac8f0")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#334455", linestyle="--", linewidth=1, alpha=0.6)

    fig.suptitle("Attention Quality Metrics (row-normalized): Baseline vs. Repeated Prompt  |  GPT-2 small",
                 color="white", fontsize=13, fontfamily="monospace", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


CSS = """
body, .gradio-container { background: #0d1117 !important; color: #c9d1d9 !important; font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
.gr-button-primary { background: #1e4d8c !important; border: 1px solid #4a90d9 !important; color: white !important; font-family: monospace !important; }
.gr-button-primary:hover { background: #4a90d9 !important; }
.gr-block, .gr-form, .gr-box { background: #111122 !important; border: 1px solid #1e3050 !important; border-radius: 8px !important; }
label, .gr-input-label { color: #7fc8f8 !important; font-family: monospace !important; font-size: 0.9em !important; }
.gr-input, .gr-dropdown select, textarea, input[type=text] { background: #0d1117 !important; color: #c9d1d9 !important; border: 1px solid #1e3050 !important; font-family: monospace !important; }
h1, h2, h3 { color: #7fc8f8 !important; font-family: monospace !important; }
.description-box { background: #111a2e; border: 1px solid #1e4d8c; border-radius: 6px; padding: 12px 16px; color: #8ab8e8; font-family: monospace; font-size: 0.88em; line-height: 1.6; }
footer { display: none !important; }
/* Fullscreen lightbox — ensure the dialog overlay is always visible above everything else */
dialog[open] { display: flex !important; z-index: 999999 !important; align-items: center !important; justify-content: center !important; border: none !important; background: transparent !important; max-width: 100vw !important; max-height: 100vh !important; padding: 0 !important; }
dialog::backdrop { background: rgba(0,0,0,0.92) !important; }
dialog img { max-width: 95vw !important; max-height: 95vh !important; object-fit: contain !important; border-radius: 4px; }
"""

INTRO_MD = """
# 🔄 Prompt Repetition: Real GPT-2 Attention Visualizer

**Based on:** *[Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/pdf/2512.14982)* — Leviathan, Kalman & Matias (Google Research, Dec 2025)

Sending a prompt twice — `[prompt | prompt]` — wins on **47/70** benchmark-model combinations for non-reasoning tasks, with no fine-tuning or additional inference calls.
The visualizations below use real attention weights from **GPT-2 small** (124M params), averaged across all 12 layers and heads.
The "Difference" panel is row-normalized so both conditions are on equal footing before subtraction.

<span style="color:#4a90d9;font-weight:bold">Blue</span> = Baseline (Single) · <span style="color:#9b59b6;font-weight:bold">Purple</span> = Repeated (Copy-2 → Copy-1) · <span style="color:#3aad50;font-weight:bold">Green</span>/<span style="color:#cc3300;font-weight:bold">Red</span> = Gain/Loss (Difference)
"""

with gr.Blocks(title="Prompt Repetition Visualizer") as demo:
    gr.Markdown(INTRO_MD)

    with gr.Row():
        with gr.Column(scale=1):
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value=list(PRESETS.keys())[0],
                label="📋 Example Use Case",
                interactive=True,
            )
            desc_box = gr.HTML(
                value=f'<div class="description-box">{list(PRESETS.values())[0]["description"]}</div>'
            )
            prompt_tb = gr.Textbox(
                value=DEFAULT_PROMPT,
                label="Prompt (edit freely — tokens shown on axes after running)",
                lines=3,
                placeholder="Type any prompt here…",
            )
            run_btn = gr.Button("▶ Run GPT-2 & Visualize Attention", variant="primary")

        with gr.Column(scale=3):
            heatmap_out = gr.Image(show_label=False, buttons=["download"])
            entropy_out = gr.Image(show_label=False, buttons=["download"])

    def on_preset(name):
        p = PRESETS[name]
        return f'<div class="description-box">{p["description"]}</div>', p["prompt"]

    def run_viz(prompt):
        attn_img = make_heatmap_fig(prompt.strip())
        entropy_img = make_entropy_chart(prompt.strip())
        return attn_img, entropy_img


    preset_dd.change(on_preset, inputs=[preset_dd], outputs=[desc_box, prompt_tb])
    run_btn.click(run_viz, inputs=[prompt_tb], outputs=[heatmap_out, entropy_out])
    demo.load(run_viz, inputs=[prompt_tb], outputs=[heatmap_out, entropy_out])


if __name__ == "__main__":
    load_model()
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False, css=CSS)
