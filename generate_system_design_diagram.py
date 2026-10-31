import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import numpy as np

def generate_system_design_diagram(out_path="report/figures/architecture_system_design.png"):
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    fig.patch.set_facecolor('#ffffff')

    # Color Palette: Modern Enterprise System Architecture Palette
    c_bg = "#f8fafc"
    c_border = "#cbd5e1"
    
    # Subsystem theme colors
    c_ingest_bg = "#eff6ff"     # Blue 50
    c_ingest_edge = "#2563eb"   # Blue 600
    
    c_enc_bg = "#f0fdf4"        # Green 50
    c_enc_edge = "#16a34a"      # Green 600
    
    c_bot_bg = "#fffbeb"        # Amber 50
    c_bot_edge = "#d97706"      # Amber 600
    
    c_dec_bg = "#f5f3ff"        # Purple 50
    c_dec_edge = "#7c3aed"      # Purple 600
    
    c_out_bg = "#fff1f2"        # Rose 50
    c_out_edge = "#e11d48"      # Rose 600

    c_gov_bg = "#f1f5f9"        # Slate 100
    c_gov_edge = "#475569"      # Slate 600

    # -------------------------------------------------------------
    # Title & Subtitle Banner
    # -------------------------------------------------------------
    ax.text(8.0, 9.15, "FastSaliencyUNet: System Design & Pipeline Architecture", 
            ha='center', va='center', fontsize=17, fontweight='800', color='#0f172a')
    ax.text(8.0, 8.80, "High-Level Architectural Decomposition: Data Ingestion, Multi-Scale Feature Engine, Skip Fusion & Output Serving", 
            ha='center', va='center', fontsize=10, color='#475569')

    def draw_component(x, y, w, h, header, title, details, bg, border, icon=""):
        # Outer box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                             facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        
        # Header bar
        header_h = 0.32
        hbar = FancyBboxPatch((x, y + h - header_h), w, header_h, 
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              facecolor=border, edgecolor=border, linewidth=1.0, zorder=3)
        ax.add_patch(hbar)
        
        ax.text(x + w/2, y + h - header_h/2, header.upper(), 
                ha='center', va='center', fontsize=7.5, fontweight='bold', color='#ffffff', zorder=4)
        
        # Title
        ax.text(x + w/2, y + h - header_h - 0.22, title, 
                ha='center', va='center', fontsize=9.0, fontweight='bold', color='#0f172a', zorder=4)
        
        # Detail items
        start_y = y + h - header_h - 0.48
        for i, line in enumerate(details):
            ax.text(x + 0.15, start_y - i * 0.24, line, 
                    ha='left', va='center', fontsize=7.2, color='#334155', zorder=4)

    def draw_subsystem_box(x, y, w, h, label, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.22",
                             facecolor="#ffffff", edgecolor=color, linewidth=1.8, linestyle="--", zorder=1)
        ax.add_patch(box)
        
        tag = FancyBboxPatch((x + 0.25, y + h - 0.28), len(label)*0.16 + 0.4, 0.40,
                             boxstyle="round,pad=0.06,rounding_size=0.1",
                             facecolor=color, edgecolor=color, zorder=2)
        ax.add_patch(tag)
        ax.text(x + 0.45 + len(label)*0.08, y + h - 0.08, label, 
                ha='center', va='center', fontsize=8.0, fontweight='bold', color='#ffffff', zorder=3)

    # -------------------------------------------------------------
    # 1. SUBSYSTEM: Ingestion & Preprocessing
    # -------------------------------------------------------------
    draw_subsystem_box(0.5, 3.8, 2.7, 4.6, "TIER 1: INGESTION", c_ingest_edge)
    
    draw_component(0.7, 6.2, 2.3, 1.8, "Client Source", "Photo Buffer", 
                   ["• Format: 256×256×3 BGR", "• Memory: Contiguous uint8", "• Zero-copy RAM stream"],
                   c_ingest_bg, c_ingest_edge)

    draw_component(0.7, 4.0, 2.3, 1.8, "Transform Unit", "Standardizer", 
                   ["• BGR → RGB conversion", "• ImageNet Mean & Std", "• Tensor: [1, 3, 256, 256]"],
                   c_ingest_bg, c_ingest_edge)

    # -------------------------------------------------------------
    # 2. SUBSYSTEM: Encoder Backbone
    # -------------------------------------------------------------
    draw_subsystem_box(3.5, 3.8, 3.4, 4.6, "TIER 2: ENCODER CORE", c_enc_edge)

    draw_component(3.7, 6.6, 3.0, 1.5, "Stem Layer", "Conv Stem (s=2)", 
                   ["• 3×3 Conv, Stride 2, BN, ReLU", "• Out: 128×128×16 (448 Params)", "• Eliminates 75% front-end FLOPs"],
                   c_enc_bg, c_enc_edge)

    draw_component(3.7, 4.9, 3.0, 1.45, "Feature Stage 1-2", "IRB Blocks (s=1, s=2)", 
                   ["• Stage 1: 128×128×24 (t=2, 1.5k P)", "• Stage 2: 64×64×32 (t=2, 5.6k P)", "• Mobile Inverted Residuals"],
                   c_enc_bg, c_enc_edge)

    draw_component(3.7, 3.9, 3.0, 0.85, "Feature Stage 3", "Deep IRB (s=2)", 
                   ["• Out: 32×32×64 (23.0k Params)"],
                   c_enc_bg, c_enc_edge)

    # -------------------------------------------------------------
    # 3. SUBSYSTEM: Global Receptive Bottleneck Core
    # -------------------------------------------------------------
    draw_subsystem_box(7.2, 3.8, 2.5, 4.6, "TIER 3: BOTTLENECK", c_bot_edge)

    draw_component(7.4, 4.8, 2.1, 3.2, "Semantic Engine", "Stage 4 Dilated Core", 
                   ["• Shape: 16×16×96", 
                    "• Dilation Rate: d = 2", 
                    "• Receptive Field: 2× wider", 
                    "• Captures global scene context", 
                    "• 39,936 Params (33.8% total)", 
                    "• 10.2M Arithmetic MACs", 
                    "• Zero-cost FOV expansion"],
                   c_bot_bg, c_bot_edge)

    # -------------------------------------------------------------
    # 4. SUBSYSTEM: Lateral Synthesis & Decoder Engine
    # -------------------------------------------------------------
    draw_subsystem_box(10.0, 3.8, 3.2, 4.6, "TIER 4: DECODER SYNTHESIS", c_dec_edge)

    draw_component(10.2, 6.7, 2.8, 1.4, "High-Level Synthesis", "Decoder 3 Module", 
                   ["• Bilinear 2×: 16→32 + Skip 3", "• DW-Sep Conv (32×32×48)", "• 25.2k Params | 25.8M MACs"],
                   c_dec_bg, c_dec_edge)

    draw_component(10.2, 5.1, 2.8, 1.4, "Mid-Level Synthesis", "Decoder 2 Module", 
                   ["• Bilinear 2×: 32→64 + Skip 2", "• DW-Sep Conv (64×64×32)", "• 13.8k Params | 56.4M MACs"],
                   c_dec_bg, c_dec_edge)

    draw_component(10.2, 3.9, 2.8, 1.0, "Low-Level Synthesis", "Decoder 1 Module", 
                   ["• Bilinear 2×: 64→128 + Skip 1", "• DW-Sep Conv (128×128×24)"],
                   c_dec_bg, c_dec_edge)

    # -------------------------------------------------------------
    # 5. SUBSYSTEM: Output Head & Mask Serving
    # -------------------------------------------------------------
    draw_subsystem_box(13.4, 3.8, 2.2, 4.6, "TIER 5: SERVING", c_out_edge)

    draw_component(13.55, 6.3, 1.9, 1.8, "Prediction Head", "Linear Output Head", 
                   ["• Bilinear 2×: 128→256", "• 3×3 Conv + 1×1 Conv", "• Logits: 256×256×1"],
                   c_out_bg, c_out_edge)

    draw_component(13.55, 4.0, 1.9, 1.9, "Serving Filter", "Mask Formatter", 
                   ["• Sigmoid Probabilities", "• Decision: p ≥ 0.50", "• Format: uint8 {0, 255}", "• Output: PNG on disk"],
                   c_out_bg, c_out_edge)

    # -------------------------------------------------------------
    # Connecting Arrows & Data Flow
    # -------------------------------------------------------------
    arr_fwd = dict(arrowstyle="-|>", color="#0f172a", lw=2.0, mutation_scale=14, zorder=5)
    arr_dash = dict(arrowstyle="-|>", color="#7c3aed", lw=1.8, ls="--", mutation_scale=14, zorder=6)

    # Within Ingestion
    ax.annotate("", xy=(1.85, 5.8), xytext=(1.85, 6.2), arrowprops=arr_fwd)

    # Ingestion -> Encoder
    ax.annotate("", xy=(3.7, 7.35), xytext=(3.0, 4.9), 
                arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, connectionstyle="arc3,rad=-0.15", mutation_scale=14, zorder=5))
    ax.text(3.3, 6.5, "RGB Tensor", fontsize=7.2, color="#0f172a", fontweight='bold', rotation=55)

    # Within Encoder
    ax.annotate("", xy=(5.2, 6.35), xytext=(5.2, 6.6), arrowprops=arr_fwd)
    ax.annotate("", xy=(5.2, 4.75), xytext=(5.2, 4.9), arrowprops=arr_fwd)

    # Encoder -> Bottleneck
    ax.annotate("", xy=(7.4, 6.4), xytext=(6.7, 4.3), 
                arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, connectionstyle="arc3,rad=-0.15", mutation_scale=14, zorder=5))
    ax.text(6.8, 5.7, "32×32×64", fontsize=7.2, color="#0f172a", fontweight='bold', rotation=35)

    # Bottleneck -> Decoder 3
    ax.annotate("", xy=(10.2, 7.4), xytext=(9.5, 6.4), 
                arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, connectionstyle="arc3,rad=-0.15", mutation_scale=14, zorder=5))

    # Within Decoder
    ax.annotate("", xy=(11.6, 6.5), xytext=(11.6, 6.7), arrowprops=arr_fwd)
    ax.annotate("", xy=(11.6, 4.9), xytext=(11.6, 5.1), arrowprops=arr_fwd)

    # Decoder -> Output Head
    ax.annotate("", xy=(13.55, 7.2), xytext=(13.0, 4.4), 
                arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, connectionstyle="arc3,rad=-0.15", mutation_scale=14, zorder=5))

    # Output Head -> Formatter
    ax.annotate("", xy=(14.5, 5.9), xytext=(14.5, 6.3), arrowprops=arr_fwd)

    # LATERAL SKIP CONNECTIONS (System Bus Routing)
    # Skip 3: Stage 3 -> Decoder 3
    ax.annotate("", xy=(10.2, 7.0), xytext=(6.7, 4.2), 
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.0, ls="--",
                                connectionstyle="arc3,rad=-0.22", mutation_scale=13, zorder=6))
    ax.text(8.4, 5.2, "Lateral Skip 3 (32×32, 64ch)", fontsize=7.2, color="#7c3aed", fontweight='bold', rotation=-22)

    # Skip 2: Stage 2 -> Decoder 2
    ax.annotate("", xy=(10.2, 5.5), xytext=(6.7, 5.3), 
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.0, ls="--",
                                connectionstyle="arc3,rad=-0.08", mutation_scale=13, zorder=6))
    ax.text(8.4, 5.65, "Lateral Skip 2 (64×64, 32ch)", fontsize=7.2, color="#7c3aed", fontweight='bold', rotation=2)

    # Skip 1: Stage 1 -> Decoder 1
    ax.annotate("", xy=(10.2, 4.4), xytext=(6.7, 5.9), 
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.0, ls="--",
                                connectionstyle="arc3,rad=0.15", mutation_scale=13, zorder=6))
    ax.text(8.4, 4.5, "Lateral Skip 1 (128×128, 24ch)", fontsize=7.2, color="#7c3aed", fontweight='bold', rotation=18)

    # -------------------------------------------------------------
    # 6. LOWER PANEL: System Governance, Hardware Watchdogs & Bus Metrics
    # -------------------------------------------------------------
    gov_box = FancyBboxPatch((0.5, 0.4), 15.0, 2.9, boxstyle="round,pad=0.15,rounding_size=0.22",
                             facecolor="#f8fafc", edgecolor="#475569", linewidth=1.5, zorder=1)
    ax.add_patch(gov_box)
    
    gtag = FancyBboxPatch((0.75, 3.05), 4.2, 0.45, boxstyle="round,pad=0.06,rounding_size=0.1",
                          facecolor="#475569", edgecolor="#475569", zorder=2)
    ax.add_patch(gtag)
    ax.text(2.85, 3.28, "SYSTEM GOVERNANCE & HARDWARE ENFORCEMENT BUS", 
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#ffffff', zorder=3)

    # 4 Governance Metric Cards
    def draw_card(cx, cy, cw, ch, c_title, items, border_c, fill_c):
        card = FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.08,rounding_size=0.12",
                              facecolor=fill_c, edgecolor=border_c, linewidth=1.2, zorder=3)
        ax.add_patch(card)
        ax.text(cx + 0.15, cy + ch - 0.25, c_title, fontsize=8.5, fontweight='bold', color='#0f172a', zorder=4)
        for idx, text in enumerate(items):
            ax.text(cx + 0.15, cy + ch - 0.55 - idx * 0.26, text, fontsize=7.5, color='#334155', zorder=4)

    draw_card(0.8, 0.7, 3.4, 2.2, "Single-Core Thread Governor", 
              ["• Policy: Strict OMP/Torch Core Locking", 
               "• OMP_NUM_THREADS = 1", 
               "• torch.set_num_threads(1)", 
               "• cv2.setNumThreads(1)", 
               "• Verified single-thread compliance", 
               "• Zero background multi-threading"], 
              "#3b82f6", "#eff6ff")

    draw_card(4.4, 0.7, 3.4, 2.2, "Latency Watchdog Telemetry", 
              ["• SLA Limit: T_CPU ≤ 50.0 ms", 
               "• Ingest & Prep: ~1.2 ms", 
               "• Neural Forward Pass: ~21.5 ms", 
               "• Output Mask Assembly: ~0.4 ms", 
               "• Total Latency: 23.09 ms", 
               "• 53.8% Headroom beneath budget"], 
              "#10b981", "#ecfdf5")

    draw_card(8.0, 0.7, 3.4, 2.2, "Storage & Packaging Bus", 
              ["• Budget Ceiling: S_disk ≤ 4.0 MiB", 
               "• Parameter Serialization: FP16 Quantized", 
               "• Raw Parameters: 118,297 (FP32: 538 KB)", 
               "• Compressed Weight File: 300,819 Bytes", 
               "• Footprint: 0.287 MiB (7.17% of limit)", 
               "• Storage Penalty: Exactly 0.00%"], 
              "#f59e0b", "#fffbeb")

    draw_card(11.6, 0.7, 3.6, 2.2, "Quality & Objective Optimization", 
              ["• Loss: BCE + Soft Dice + Sobel Edge", 
               "• DUTS Validation Quality: Q = 0.7506", 
               "• DUTS-TE Test Score: Q = 0.5707", 
               "• Real Campus Generalization: Q = 0.9696", 
               "• Total Budget Penalties: 0.00%", 
               "• Cost-Efficient Score: CES = 0.5707"], 
              "#8b5cf6", "#f5f3ff")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated enterprise system design architecture at {out_path}")

if __name__ == "__main__":
    generate_system_design_diagram()
