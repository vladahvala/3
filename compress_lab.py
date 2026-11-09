"""
Лабораторна: експериментальне стиснення зображень на основі DCT та DWT (Хаар).

Усі перетворення реалізовані вручну (NumPy лише для масивів і множення матриць):
  * DCT 8x8 через матричне представлення:  D = C . B . C^T,  B = C^T . D . C
  * DWT Хаара (ортонормований), кілька рівнів, 2D = рядки, потім стовпці
  * Стиснення: збереження заданої частки НАЙБІЛЬШИХ за модулем коефіцієнтів
    (глобальний thresholding), решта обнуляється
  * Якість: PSNR (+ коефіцієнт стиснення = усього коефіцієнтів / ненульових)

Запуск:
  python compress_lab.py                       # інтерактивне вікно, синтетичне зображення
  python compress_lab.py photo.jpg             # інтерактивне вікно зі своїм зображенням
  python compress_lab.py photo.jpg --report    # таблиця + картинки в папку results/, без вікна
  python compress_lab.py --selftest            # перевірка коректності перетворень
"""
import argparse
import os
import sys

import numpy as np

BLOCK = 8
SQRT2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------
# Завантаження зображення
# ----------------------------------------------------------------------------
def load_gray(path, size=512):
    """Читає зображення, переводить у відтінки сірого і масштабує до size x size.
    Без шляху створює синтетичне тестове зображення."""
    if path is None:
        return synthetic_image(size)
    from PIL import Image
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def synthetic_image(size=512):
    """Тестове зображення: плавні градієнти, різкі краї, текстура, дрібний шум."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    img = 90 + 70 * np.sin(x / 61.0) * np.cos(y / 83.0)          # плавні переходи
    img[(x - size * 0.30) ** 2 + (y - size * 0.35) ** 2 < (size * 0.16) ** 2] = 230  # коло
    img[int(size * 0.55):int(size * 0.85), int(size * 0.50):int(size * 0.90)] = 40    # прямокутник
    img[(x + y) % 24 < 3] += 35                                  # діагональні смуги (висока частота)
    rng = np.random.default_rng(0)
    img += rng.normal(0, 3, img.shape)                           # шум
    return np.round(np.clip(img, 0, 255))                        # цілі значення, як у 8-бітному зображенні


# ----------------------------------------------------------------------------
# DCT 8x8 (матричне представлення)
# ----------------------------------------------------------------------------
def dct_matrix(n=BLOCK):
    """Ортонормована матриця DCT-II:  C[k, j] = a(k) * cos((2j+1) k pi / 2n)."""
    c = np.zeros((n, n))
    for k in range(n):
        a = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for j in range(n):
            c[k, j] = a * np.cos((2 * j + 1) * k * np.pi / (2 * n))
    return c


C = dct_matrix()


def to_blocks(img, n=BLOCK):
    """(H, W) -> (H/n, W/n, n, n)"""
    h, w = img.shape
    return img.reshape(h // n, n, w // n, n).transpose(0, 2, 1, 3)


def from_blocks(blocks):
    """(H/n, W/n, n, n) -> (H, W)"""
    bh, bw, n, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(bh * n, bw * n)


def dct2_blocks(img):
    """Пряме DCT усіх блоків 8x8:  D = C . B . C^T  (зі зсувом -128)."""
    b = to_blocks(img - 128.0)
    return C @ b @ C.T


def idct2_blocks(coefs):
    """Обернене DCT:  B = C^T . D . C  (і повернення зсуву +128)."""
    b = C.T @ coefs @ C
    return from_blocks(b) + 128.0


# ----------------------------------------------------------------------------
# DWT Хаара (ортонормований), кілька рівнів
# ----------------------------------------------------------------------------
def _fwd_rows(a):
    even, odd = a[:, 0::2], a[:, 1::2]
    return np.hstack(((even + odd) / SQRT2, (even - odd) / SQRT2))


def _fwd_cols(a):
    even, odd = a[0::2, :], a[1::2, :]
    return np.vstack(((even + odd) / SQRT2, (even - odd) / SQRT2))


def _inv_rows(a):
    h = a.shape[1] // 2
    low, high = a[:, :h], a[:, h:]
    out = np.empty_like(a)
    out[:, 0::2] = (low + high) / SQRT2
    out[:, 1::2] = (low - high) / SQRT2
    return out


def _inv_cols(a):
    h = a.shape[0] // 2
    low, high = a[:h, :], a[h:, :]
    out = np.empty_like(a)
    out[0::2, :] = (low + high) / SQRT2
    out[1::2, :] = (low - high) / SQRT2
    return out


def haar2(img, levels):
    """Пряме 2D перетворення Хаара. Розкладка Маллата: LL у лівому верхньому куті."""
    out = (img - 128.0).copy()
    n = out.shape[0]
    for _ in range(levels):
        out[:n, :n] = _fwd_cols(_fwd_rows(out[:n, :n]))
        n //= 2
    return out


def ihaar2(coefs, levels):
    """Обернене 2D перетворення Хаара."""
    out = coefs.copy()
    n = out.shape[0] // (2 ** (levels - 1))
    for _ in range(levels):
        out[:n, :n] = _inv_rows(_inv_cols(out[:n, :n]))
        n *= 2
    return out + 128.0


# ----------------------------------------------------------------------------
# Стиснення: збереження частки найбільших коефіцієнтів
# ----------------------------------------------------------------------------
def keep_largest(coefs, fraction):
    """Лишає `fraction` (0..1] найбільших за модулем коефіцієнтів, решту обнуляє.
    Повертає (оброблені коефіцієнти, кількість збережених)."""
    flat = np.abs(coefs).ravel()
    total = flat.size
    k = int(round(fraction * total))
    k = max(1, min(k, total))
    out = np.zeros_like(coefs)
    idx = np.argpartition(flat, total - k)[total - k:]   # індекси k найбільших
    out.ravel()[idx] = coefs.ravel()[idx]
    return out, k


def to_image(arr):
    """Округлення і обрізання до діапазону 0..255."""
    return np.clip(np.round(arr), 0, 255)


def compress_dct(img, fraction):
    coefs = dct2_blocks(img)
    kept, k = keep_largest(coefs, fraction)
    return to_image(idct2_blocks(kept)), k, coefs.size


def compress_dwt(img, fraction, levels=3):
    coefs = haar2(img, levels)
    kept, k = keep_largest(coefs, fraction)
    return to_image(ihaar2(kept, levels)), k, coefs.size


# ----------------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------------
def psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def evaluate(img, fraction, levels=3):
    """Усі показники для одного рівня стиснення обох методів."""
    res = {}
    for name, fn in (("DCT", lambda f: compress_dct(img, f)),
                     ("DWT", lambda f: compress_dwt(img, f, levels))):
        rec, k, total = fn(fraction)
        res[name] = dict(image=rec, kept=k, total=total, share=k / total,
                         ratio=total / k, psnr=psnr(img, rec))
    return res


# ----------------------------------------------------------------------------
# Звіт: таблиця + зображення
# ----------------------------------------------------------------------------
LEVELS_DEFAULT = (0.50, 0.20, 0.10, 0.05)


def run_report(img, outdir, levels, fractions=LEVELS_DEFAULT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    print(f"\nЗображення {img.shape[0]}x{img.shape[1]}, DWT: {levels} рівні(в) Хаара, "
          f"блок DCT {BLOCK}x{BLOCK}")
    print("Коефіцієнт стиснення = усього коефіцієнтів / збережених (це НЕ розмір файлу).\n")
    header = f"{'Збережено':>10} | {'Метод':^5} | {'Коеф.':>8} | {'Частка':>7} | {'К-сть стиснення':>15} | {'PSNR, дБ':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    fig, axes = plt.subplots(len(fractions), 3, figsize=(10.5, 3.4 * len(fractions)))
    for r, frac in enumerate(fractions):
        res = evaluate(img, frac, levels)
        for name in ("DCT", "DWT"):
            m = res[name]
            print(f"{frac * 100:>9.0f}% | {name:^5} | {m['kept']:>8d} | {m['share'] * 100:>6.1f}% | "
                  f"{m['ratio']:>14.1f}x | {m['psnr']:>9.2f}")
            rows.append((frac, name, m["kept"], m["share"], m["ratio"], m["psnr"]))
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title("Original")
        for c, name in ((1, "DCT"), (2, "DWT")):
            m = res[name]
            axes[r, c].imshow(m["image"], cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(f"{name}  {m['share'] * 100:.0f}% коеф.  PSNR={m['psnr']:.2f} дБ")
            from PIL import Image
            Image.fromarray(m["image"].astype(np.uint8)).save(
                os.path.join(outdir, f"{name.lower()}_{int(frac * 100):02d}pct.png"))
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), dpi=110)

    # CSV для таблиці у звіті
    with open(os.path.join(outdir, "results.csv"), "w", encoding="utf-8") as f:
        f.write("fraction,method,kept,share,ratio,psnr_db\n")
        for fr, nm, k, sh, ra, ps in rows:
            f.write(f"{fr},{nm},{k},{sh:.5f},{ra:.3f},{ps:.3f}\n")

    # PSNR залежно від частки збережених коефіцієнтів
    xs = np.linspace(0.01, 0.6, 30)
    fig2, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("DCT", "tab:blue"), ("DWT", "tab:orange")):
        ax.plot(xs * 100, [evaluate(img, f, levels)[name]["psnr"] for f in xs],
                label=name, color=color)
    ax.set_xlabel("Збережено коефіцієнтів, %")
    ax.set_ylabel("PSNR, дБ")
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "psnr_curve.png"), dpi=110)
    print(f"\nФайли збережено в папку: {os.path.abspath(outdir)}")


# ----------------------------------------------------------------------------
# Інтерактивне вікно: Original | DCT | DWT + повзунок рівня стиснення
# ----------------------------------------------------------------------------
def run_gui(img, levels):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6))
    plt.subplots_adjust(bottom=0.2, top=0.88)
    shown = []
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    for a, name in zip(axes[1:], ("DCT", "DWT")):
        shown.append(a.imshow(img, cmap="gray", vmin=0, vmax=255))
    for a in axes:
        a.axis("off")

    ax_s = plt.axes([0.25, 0.07, 0.5, 0.04])
    slider = Slider(ax_s, "Збережено, %", 1, 100, valinit=20, valstep=1)

    def update(_=None):
        frac = slider.val / 100.0
        res = evaluate(img, frac, levels)
        for im, a, name in zip(shown, axes[1:], ("DCT", "DWT")):
            m = res[name]
            im.set_data(m["image"])
            a.set_title(f"{name}\nPSNR = {m['psnr']:.2f} дБ\n"
                        f"збережено {m['kept']} з {m['total']} ({m['share'] * 100:.0f}%), "
                        f"стиснення {m['ratio']:.1f}x", fontsize=10)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update()
    plt.show()


# ----------------------------------------------------------------------------
# Самоперевірка коректності
# ----------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(f"[{'OK' if cond else 'FAIL'}] {name} {info}")
        ok &= bool(cond)

    # 1. Матриця DCT ортогональна: C . C^T = I
    err = np.abs(C @ C.T - np.eye(BLOCK)).max()
    check("матриця DCT ортогональна", err < 1e-12, f"(max|C.C^T - I| = {err:.1e})")

    # 2. Матричне DCT збігається з прямою формулою (подвійна сума) для одного блока
    rng = np.random.default_rng(1)
    blk = rng.uniform(0, 255, (BLOCK, BLOCK)) - 128
    naive = np.zeros((BLOCK, BLOCK))
    for u in range(BLOCK):
        for v in range(BLOCK):
            au = np.sqrt(1 / BLOCK) if u == 0 else np.sqrt(2 / BLOCK)
            av = np.sqrt(1 / BLOCK) if v == 0 else np.sqrt(2 / BLOCK)
            s = 0.0
            for x in range(BLOCK):
                for y in range(BLOCK):
                    s += blk[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * BLOCK)) \
                         * np.cos((2 * y + 1) * v * np.pi / (2 * BLOCK))
            naive[u, v] = au * av * s
    err = np.abs(C @ blk @ C.T - naive).max()
    check("DCT через матрицю = формула DCT-II", err < 1e-9, f"(max різниця = {err:.1e})")

    # 3. DCT -> IDCT без відкидання відновлює зображення точно
    img = synthetic_image(64)
    err = np.abs(idct2_blocks(dct2_blocks(img)) - img).max()
    check("DCT -> IDCT без втрат", err < 1e-9, f"(max похибка = {err:.1e})")

    # 4. Haar -> IHaar без відкидання відновлює зображення точно (1, 2, 3 рівні)
    for lv in (1, 2, 3):
        err = np.abs(ihaar2(haar2(img, lv), lv) - img).max()
        check(f"Хаар -> обернений Хаар, {lv} рів.", err < 1e-9, f"(max похибка = {err:.1e})")

    # 5. Збереження енергії (ортонормованість): сума квадратів коефіцієнтів не змінюється
    e0 = np.sum((img - 128.0) ** 2)
    check("DCT зберігає енергію", abs(np.sum(dct2_blocks(img) ** 2) - e0) / e0 < 1e-12)
    check("Хаар зберігає енергію", abs(np.sum(haar2(img, 3) ** 2) - e0) / e0 < 1e-12)

    # 6. На константному зображенні всі коефіцієнти, крім низькочастотних, нульові
    flat = np.full((16, 16), 200.0)
    d = dct2_blocks(flat)
    check("константа: у кожному блоці ненульовий лише DC",
          np.count_nonzero(np.abs(d) > 1e-9) == 4)
    h = haar2(flat, 2)
    check("константа: Хаар (2 рів.) має ненульовими лише 4x4 коефіцієнти LL",
          np.count_nonzero(np.abs(h) > 1e-9) == 16 and np.all(np.abs(h[:4, :4]) > 1e-9))

    # 7. keep_largest лишає рівно k коефіцієнтів і саме найбільші
    a = rng.normal(size=(16, 16))
    out, k = keep_largest(a, 0.25)
    check("keep_largest: рівно 25% ненульових", np.count_nonzero(out) == k == 64)
    check("keep_largest: лишає найбільші за модулем",
          np.abs(out[out != 0]).min() >= np.abs(a[out == 0]).max())

    # 8. PSNR: 100% коефіцієнтів = нескінченність; ручний розрахунок для відомого MSE
    check("100% коефіцієнтів: відновлення ідеальне, PSNR = inf (DCT і DWT)",
          psnr(img, compress_dct(img, 1.0)[0]) == float("inf")
          and psnr(img, compress_dwt(img, 1.0, 3)[0]) == float("inf"))
    z = np.zeros((4, 4)); w = np.full((4, 4), 10.0)       # MSE = 100
    check("PSNR: MSE=100 -> 10*log10(255^2/100) = 28.13 дБ", abs(psnr(z, w) - 28.1308) < 1e-3)

    # 9. Монотонність: менше коефіцієнтів -> не краща якість
    big = synthetic_image(128)
    for name in ("DCT", "DWT"):
        vals = [evaluate(big, f)[name]["psnr"] for f in (0.5, 0.2, 0.1, 0.05)]
        check(f"{name}: PSNR зменшується зі зменшенням частки коефіцієнтів",
              all(x > y for x, y in zip(vals, vals[1:])), f"({', '.join(f'{v:.1f}' for v in vals)} дБ)")

    print("\nУСІ ПЕРЕВІРКИ ПРОЙДЕНО" if ok else "\nє ПОМИЛКИ")
    return ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Стиснення зображень: DCT vs DWT (Хаар)")
    ap.add_argument("image", nargs="?", help="шлях до зображення (за замовчуванням синтетичне)")
    ap.add_argument("--size", type=int, default=512, help="розмір після масштабування (кратний 8 і 2^levels)")
    ap.add_argument("--levels", type=int, default=3, help="кількість рівнів DWT")
    ap.add_argument("--report", action="store_true", help="таблиця і файли в папку results/, без вікна")
    ap.add_argument("--out", default="results", help="папка для --report")
    ap.add_argument("--selftest", action="store_true", help="перевірка коректності перетворень")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.size % BLOCK or args.size % (2 ** args.levels):
        ap.error("size має ділитися на 8 і на 2^levels")

    img = load_gray(args.image, args.size)
    if args.report:
        run_report(img, args.out, args.levels)
    else:
        run_gui(img, args.levels)


if __name__ == "__main__":
    main()