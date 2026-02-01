import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ==========================================
# 1. Configuration (Fonts & Style)
# ==========================================
def configure_plot_style():
    # Use standard sans-serif fonts for English
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math formulas


# ==========================================
# 2. Main Drawing Logic
# ==========================================
def draw_english_expansion_map():
    configure_plot_style()

    # Initialize Canvas
    fig, ax = plt.subplots(figsize=(16, 12))  # Increased size for better spacing
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.axis('off')

    # Color Palette (Softer, professional tones)
    C_INPUT = '#f0f0f0'  # Light Gray
    C_COMPET = '#fff2cc'  # Light Yellow
    C_FINANCE = '#dae8fc'  # Light Blue
    C_CALC = '#ffe6cc'  # Light Orange
    C_OUTPUT = '#d5e8d4'  # Light Green
    C_BORDER = '#666666'  # Dark Gray border

    # ==========================================
    # 3. Node Definitions
    # Format: Key: (x, y, width, height, color, text)
    # Coordinates (x,y) are the bottom-left corner of the box
    # ==========================================
    nodes = {
        # --- Level 1: Inputs (Top) ---
        'EXPANSION': (40, 92, 20, 8, C_INPUT, "League Expansion\nEvent"),
        'RULES': (15, 80, 22, 8, C_INPUT, "Expansion Rules\n(Draft / Fees)"),
        'CITY': (65, 80, 22, 8, C_INPUT, "Candidate City\nCharacteristics ($\\ell$)"),

        # --- Level 2: Competitive Path (Left) ---
        'PROTECT': (10, 62, 22, 8, C_COMPET, "Protection Optimization\n(List $z_i$)"),
        'DRAFT': (5, 46, 20, 8, C_COMPET, "Talent Loss\n($\\Delta V$)"),
        'DILUTION': (28, 46, 20, 8, C_COMPET, "Schedule Dilution\nEffect"),

        # --- Level 3: Financial Path (Right) ---
        'FEE': (55, 62, 20, 8, C_FINANCE, "Fee Income Share\n($\\lambda F_{exp}/N$)"),
        'DIST': (80, 62, 18, 8, C_FINANCE, "Geo-Distance\n($d_{\\ell j}$)"),
        'MARKET': (78, 46, 22, 8, C_FINANCE, "Market Cannibalization\n($-R \\cdot e^{-d/\\sigma}$)"),

        # --- Level 4: Aggregation (Middle) ---
        'NET_WP': (15, 28, 25, 10, C_CALC, "Net Competitive Impact\n$\\Delta WP_{exp}$"),
        'NET_FIN': (60, 28, 25, 10, C_CALC, "Net Financial Impact\n$\\Delta \\Pi_{exp}$"),

        # --- Level 5: Final Goal (Bottom) ---
        'TOTAL': (38, 10, 24, 12, C_OUTPUT, "Total Impact Score\n$Impact_{\\ell}$")
    }

    # ==========================================
    # 4. Draw Edges (Arrows)
    # ==========================================
    edges = [
        # From Expansion Event
        ('EXPANSION', 'RULES'),
        ('EXPANSION', 'CITY'),
        ('EXPANSION', 'DILUTION'),  # Direct link to dilution

        # Competitive Branch
        ('RULES', 'PROTECT'),
        ('PROTECT', 'DRAFT'),

        # Financial Branch
        ('CITY', 'FEE'),
        ('CITY', 'DIST'),
        ('DIST', 'MARKET'),

        # Aggregation
        ('DRAFT', 'NET_WP'),
        ('DILUTION', 'NET_WP'),
        ('FEE', 'NET_FIN'),
        ('MARKET', 'NET_FIN'),

        # Final
        ('NET_WP', 'TOTAL'),
        ('NET_FIN', 'TOTAL')
    ]

    def draw_arrow(start_key, end_key):
        # Get coordinates
        x1, y1, w1, h1, _, _ = nodes[start_key]
        x2, y2, w2, h2, _, _ = nodes[end_key]

        # Calculate centers
        start_pt = (x1 + w1 / 2, y1)  # Bottom center
        end_pt = (x2 + w2 / 2, y2 + h2)  # Top center

        # Logic for specific connections to make them look cleaner
        connection_style = "arc3,rad=0.1"  # Default gentle curve

        # Adjust start point for side-branching to avoid text overlap
        if start_key == 'EXPANSION':
            start_pt = (x1 + w1 / 2, y1)  # Center bottom

        # Special case: Vertical straight lines
        if abs((x1 + w1 / 2) - (x2 + w2 / 2)) < 5:
            connection_style = "arc3,rad=0.0"

        ax.annotate("",
                    xy=end_pt, xycoords='data',
                    xytext=start_pt, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    color="#555555",
                                    lw=1.5,
                                    shrinkA=5, shrinkB=5,  # Gap between box and arrow
                                    connectionstyle=connection_style))

    for u, v in edges:
        draw_arrow(u, v)

    # ==========================================
    # 5. Draw Nodes (Boxes)
    # ==========================================
    for key, (x, y, w, h, color, text) in nodes.items():
        # Draw Box
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.5,rounding_size=0.5",
                             linewidth=1.5,
                             edgecolor=C_BORDER,
                             facecolor=color,
                             zorder=10)  # Ensure boxes are on top of arrows
        ax.add_patch(box)

        # Draw Text
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=11,
                fontweight='normal',
                color='#000000',
                linespacing=1.5,
                zorder=11)

    # ==========================================
    # 6. Legends and Titles
    # ==========================================
    # Title
    ax.text(50, 102, "H2D2 Expansion Mechanism: Influence Diagram",
            ha="center", fontsize=18, fontweight='bold', color='#333333')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_INPUT, edgecolor=C_BORDER, label='Input / Environment'),
        mpatches.Patch(facecolor=C_COMPET, edgecolor=C_BORDER, label='Competitive Mechanism'),
        mpatches.Patch(facecolor=C_FINANCE, edgecolor=C_BORDER, label='Financial Mechanism'),
        mpatches.Patch(facecolor=C_CALC, edgecolor=C_BORDER, label='Intermediate Calculation'),
        mpatches.Patch(facecolor=C_OUTPUT, edgecolor=C_BORDER, label='Final Decision Metric')
    ]

    ax.legend(handles=legend_elements,
              loc='lower right',
              bbox_to_anchor=(0.98, 0.02),
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_english_expansion_map()
