from pathlib import Path
import matplotlib.pyplot as plt
import string

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "webapp" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

def save_latex_image(latex_code, filename, fontsize=200):
    fig = plt.figure(figsize=(4, 4), dpi=128)
    fig.text(0.5, 0.5, f"${latex_code}$", fontsize=fontsize, ha='center', va='center', color='black')
    plt.axis('off')
    plt.savefig(TEMPLATES_DIR / filename, bbox_inches='tight', pad_inches=0.2, facecolor='white', transparent=False)
    plt.close(fig)

chars = (
    list("0123456789") +
    list(string.ascii_lowercase) +
    list("αβγδεζηθικλμνξοπρστυφχψω")
)

for char in chars:
    filename = f"u{ord(char):04x}.png"
    save_latex_image(char, filename, fontsize=220)

EXPRESSIONS = {
    # MEDIUM
    "exp_1": r"x + 1", "exp_2": r"2a", "exp_3": r"y = 5", "exp_4": r"a - b",
    "exp_9": r"3x", "exp_10": r"b = 2", "exp_11": r"x + y", "exp_12": r"m - n",
    "exp_13": r"5 - 2", "exp_14": r"4c", "exp_15": r"p = q", "exp_16": r"7 + 8",

    # HARD
    "exp_5": r"\frac{a}{b}", "exp_6": r"x^2", "exp_7": r"\sqrt{x}", "exp_8": r"2x + 3",
    "exp_17": r"y^3", "exp_18": r"\frac{1}{x}", "exp_19": r"\sqrt{y}", "exp_20": r"a^2 + b^2",
    "exp_21": r"\frac{x}{2}", "exp_22": r"3x - 1", "exp_23": r"\sqrt{x+1}", "exp_24": r"x^2 = 4",

    # IMPOSSIBLE
    "imp_1": r"\int_0^1 x^2 \, dx",
    "imp_2": r"\lim_{x \to 0} \frac{\sin x}{x}",
    "imp_3": r"\sum_{i=1}^n i^2",
    "imp_4": r"e^{i\pi} + 1 = 0"
}

for exp_id, latex_code in EXPRESSIONS.items():
    filename = f"{exp_id}.png"
    save_latex_image(latex_code, filename, fontsize=100)

print("done.")