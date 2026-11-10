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


# 1. Генерация БУКВ И ЦИФР (Easy)
# Маппинг греческих букв в LaTeX-команды
greek_map = {
    'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'ε': r'\epsilon',
    'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta', 'ι': r'\iota', 'κ': r'\kappa',
    'λ': r'\lambda', 'μ': r'\mu', 'ν': r'\nu', 'ξ': r'\xi', 'ο': 'o', 'π': r'\pi',
    'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon', 'φ': r'\phi',
    'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega'
}

# Объединяем цифры, латиницу и греческий
chars = list("0123456789") + list(string.ascii_lowercase) + list(greek_map.keys())

print("Генерирую буквы...")
for char in chars:
    filename = f"u{ord(char):04x}.png"

    # Если это греческая буква, берем её LaTeX-код, иначе оставляем как есть
    latex_symbol = greek_map.get(char, char)

    # Для букв размер чуть больше
    save_latex_image(latex_symbol, filename, fontsize=220)

# 2. Генерация ВЫРАЖЕНИЙ (Medium, Hard, Imp)
EXPRESSIONS = {
    # MEDIUM
    "exp_1": r"x+1", "exp_2": r"2a", "exp_3": r"y=5", "exp_4": r"a-b",
    "exp_5": r"3x", "exp_6": r"b=2", "exp_7": r"x+y", "exp_8": r"m-n",
    "exp_9": r"5-2", "exp_10": r"4c", "exp_11": r"p=q", "exp_12": r"7+8",
    "exp_13": r"a+b", "exp_14": r"x=3", "exp_15": r"2y", "exp_16": r"4+5",
    "exp_17": r"n-1", "exp_18": r"3+a", "exp_19": r"y=7", "exp_20": r"c-d",
    "exp_21": r"8-4", "exp_22": r"x-2", "exp_23": r"5y", "exp_24": r"b+3",
    "exp_25": r"z=1", "exp_26": r"6+2", "exp_27": r"a=4", "exp_28": r"m+2",
    "exp_29": r"9-3", "exp_30": r"2x+1",

    # HARD
    "exp_31": r"\frac{a}{b}", "exp_32": r"x^2", "exp_33": r"\sqrt{x}", "exp_34": r"2x+3",
    "exp_35": r"y^3", "exp_36": r"\frac{1}{x}", "exp_37": r"\sqrt{y}", "exp_38": r"a^2+b^2",
    "exp_39": r"\frac{x}{2}", "exp_40": r"3x-1", "exp_41": r"\sqrt{x+1}", "exp_42": r"x^2=4",
    "exp_43": r"\frac{2}{3}", "exp_44": r"z^2", "exp_45": r"\sqrt{z}", "exp_46": r"4x-2",
    "exp_47": r"\frac{a+c}{b}", "exp_48": r"x^3", "exp_49": r"\sqrt{x-1}", "exp_50": r"x^2+2x",
    "exp_51": r"\frac{3}{x}", "exp_52": r"2x^2", "exp_53": r"y^2-1", "exp_54": r"\sqrt{2x}",
    "exp_55": r"5x+2", "exp_56": r"\frac{x+1}{2}", "exp_57": r"a^3", "exp_58": r"\sqrt{x^2}",
    "exp_59": r"4x^2", "exp_60": r"\frac{1}{x^2}", "exp_61": r"x=2^3", "exp_62": r"\sqrt{4}",
    "exp_63": r"7y-3", "exp_64": r"\frac{y}{4}", "exp_65": r"3x^2-5", "exp_66": r"\sqrt{x+2}",
    "exp_67": r"x^2-9", "exp_68": r"\frac{x}{y}", "exp_69": r"2x+2y", "exp_70": r"x^3-1",

    # IMPOSSIBLE
    "exp_71": r"\int_0^1 x^2 \, dx", "exp_72": r"\lim_{x \to 0} \frac{\sin x}{x}", "exp_73": r"\sum_{i=1}^n i^2",
    "exp_74": r"e^{i\pi} + 1 = 0",
    "exp_75": r"\int x \, dx", "exp_76": r"\lim_{x \to \infty} \frac{1}{x}", "exp_77": r"\sum_{k=1}^n k",
    "exp_78": r"\frac{d}{dx} x^2",
    "exp_79": r"\int_0^\pi \sin x \, dx", "exp_80": r"\lim_{x \to 1} \frac{x^2-1}{x-1}",
    "exp_81": r"\sum_{n=1}^\infty \frac{1}{n^2}", "exp_82": r"\frac{d}{dx} e^x",
    "exp_83": r"\int \frac{1}{x} \, dx", "exp_84": r"\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}",
    "exp_85": r"\prod_{i=1}^n i", "exp_86": r"i^2 = -1",
    "exp_87": r"\int_0^1 e^x \, dx", "exp_88": r"\lim_{x \to 0} \frac{1-\cos x}{x^2}",
    "exp_89": r"\sum_{i=0}^n \binom{n}{i}", "exp_90": r"\frac{d}{dx} \ln x",
    "exp_91": r"\int x e^x \, dx", "exp_92": r"\lim_{n \to \infty} (1+\frac{1}{n})^n",
    "exp_93": r"\sum_{i=1}^n \frac{1}{i}", "exp_94": r"\frac{\partial f}{\partial x}",
    "exp_95": r"\int_0^\infty e^{-x} \, dx", "exp_96": r"\lim_{x \to 0} \frac{\tan x}{x}",
    "exp_97": r"\sum_{i=1}^n i^3", "exp_98": r"\nabla \cdot \vec{E}",
    "exp_99": r"\int \sin x \, dx", "exp_100": r"\lim_{x \to \infty} e^{-x}"
}

print("Генерирую выражения...")
for exp_id, latex_code in EXPRESSIONS.items():
    save_latex_image(latex_code, f"{exp_id}.png", fontsize=100)

print("✅ Все эталоны готовы! Греческие буквы восстановлены.")