# %% [markdown]
# # Experiment 10: BPSK, QPSK and DPSK (first-principles, NumPy + Matplotlib only)
#
# No communications library is used. Mapping, detection, differential encoding, AWGN,
# phase offset and BER counting are all implemented by hand.
#
# **Conventions**
# * Es = 1 for every scheme. BPSK/DPSK: Eb = 1. QPSK (2 bits/symbol): Eb = Es/2 = 0.5.
# * N0 = Eb / (Eb/N0); complex noise n = sqrt(N0/2)(N(0,1) + jN(0,1)), so E|n|^2 = N0.
# * BPSK: bit 0 -> -1, bit 1 -> +1. QPSK Gray: 00 -> +1+j, 01 -> -1+j, 11 -> -1-j, 10 -> +1-j, all / sqrt(2).
# * DPSK: b[k] = 2*bit-1, d[0] = +1 (reference), d[k] = d[k-1]*b[k]; detect Re(r[k] conj(r[k-1])) >= 0 -> bit 1.

# %% Setup and parameters
import math
import os
import numpy as np
import matplotlib

try:
    get_ipython()            # defined inside Jupyter / Colab
    IN_NB = True
except NameError:
    IN_NB = False
    matplotlib.use("Agg")    # headless when run as a plain script
import matplotlib.pyplot as plt

SHOW = IN_NB                 # set True to pop up windows when running as a script
SAVE_FIGS = True
OUTDIR = "exp10_figures"

rng = np.random.default_rng(1)          # reproducible results (seed = 1)

# Data sizes
N_DEMO = 20000           # bits for constellation / statistics demos (even)
N_BER = 1_000_000        # bits per point, BER vs Eb/N0 (even)
N_PH = 200_000           # bits per point, phase-offset sweeps (even)
N_TRAJ = 40              # symbols in trajectory plots
N_PLOT = 3000            # points drawn per scatter plot

# Eb/N0 settings (dB)
EBN0_DEMO = 8
EBN0_ROT = 14
EBN0_VEC = np.arange(0, 13, 2)
EBN0_FIX = 10

# Phase offsets (degrees)
PHASE_DEG = [10, 30, 60, 90]
ROT_DEG = [0, 10, 30, 60, 90]
SWEEP_DEG = np.arange(0, 181, 10)

# Normalisation (Es = 1 for all)
ES = 1.0
EB_BPSK, EB_QPSK, EB_DPSK = ES / 1, ES / 2, ES / 1

# Passband cross-check parameters
RS = 1.0; TS = 1 / RS; FC = 4 * RS; SPS = 40

COLS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]
db2lin = lambda x: 10 ** (np.asarray(x, dtype=float) / 10)
Qf = lambda x: 0.5 * np.vectorize(math.erfc)(np.asarray(x, dtype=float) / math.sqrt(2))

if SAVE_FIGS:
    os.makedirs(OUTDIR, exist_ok=True)
checks = []


def finish(fig, name):
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=130)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)

# %% Core functions (mapper, demapper, differential codec, AWGN, phase offset, BER)


def bpsk_mod(bits):
    """bit 0 -> -1, bit 1 -> +1 (Es = Eb = 1)."""
    return 2.0 * np.asarray(bits) - 1.0


def bpsk_demod(r):
    """Coherent decision: Re(r) >= 0 -> 1 (boundary at 0 for equiprobable antipodal symbols)."""
    return (np.real(r) >= 0).astype(int)


def qpsk_gray_mod(bits):
    """bits = [b1 b2 b1 b2 ...]. b2 sets the sign of I, b1 the sign of Q. Symbol = (I + jQ)/sqrt(2)."""
    bits = np.asarray(bits)
    b1, b2 = bits[0::2], bits[1::2]
    I = 1 - 2 * b2           # b2 = 0 -> +1, 1 -> -1
    Q = 1 - 2 * b1           # b1 = 0 -> +1, 1 -> -1
    return (I + 1j * Q) / np.sqrt(2)


def qpsk_gray_demod(r):
    """Independent sign decisions on I and Q, then the inverse of the mapper."""
    r = np.atleast_1d(r)
    I_hat = 2 * (np.real(r) >= 0) - 1
    Q_hat = 2 * (np.imag(r) >= 0) - 1
    bits = np.zeros(2 * r.size, dtype=int)
    bits[0::2] = (Q_hat < 0)     # b1
    bits[1::2] = (I_hat < 0)     # b2
    return bits, I_hat, Q_hat


def differential_encode(bits):
    """d[0] = +1 (reference), d[k] = d[k-1] * b[k]. Output length N+1. cumprod == the recursion."""
    b = 2 * np.asarray(bits) - 1
    return np.concatenate(([1.0], np.cumprod(b)))


def dpsk_detect(r):
    """z[k] = r[k] conj(r[k-1]); Re(z) >= 0 -> bit 1 (no phase change)."""
    z = r[1:] * np.conj(r[:-1])
    return (np.real(z) >= 0).astype(int), z


def cnoise(shape, N0):
    """Circular complex Gaussian noise with E|n|^2 = N0 (variance N0/2 per dimension)."""
    return np.sqrt(N0 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def add_awgn(s, N0):
    return s + cnoise(np.shape(s), N0)


def apply_phase_offset(r, theta_deg):
    """Rotate by theta (degrees converted to radians)."""
    return r * np.exp(1j * np.deg2rad(theta_deg))


def calculate_ber(tx, rx):
    tx, rx = np.asarray(tx), np.asarray(rx)
    assert tx.size == rx.size, "BER inputs must have equal length"
    nerr = int(np.sum(tx != rx))
    return nerr, nerr / tx.size


def rand_bits(n):
    return (rng.random(n) > 0.5).astype(int)


def sim_bpsk(N, ebn0_db, theta_deg=0.0):
    N0 = EB_BPSK / db2lin(ebn0_db)
    b = rand_bits(N)
    r = apply_phase_offset(add_awgn(bpsk_mod(b), N0), theta_deg)
    return calculate_ber(b, bpsk_demod(r))[1]


def sim_qpsk(N, ebn0_db, theta_deg=0.0):
    N0 = EB_QPSK / db2lin(ebn0_db)
    b = rand_bits(N)
    r = apply_phase_offset(add_awgn(qpsk_gray_mod(b), N0), theta_deg)
    bh = qpsk_gray_demod(r)[0]
    ber = calculate_ber(b, bh)[1]
    ser = np.mean((b[0::2] != bh[0::2]) | (b[1::2] != bh[1::2]))
    return ber, ser


def sim_dpsk(N, ebn0_db, theta_deg=0.0, drift_deg=0.0):
    """theta_deg: constant offset; drift_deg: phase increment per symbol (frequency offset)."""
    N0 = EB_DPSK / db2lin(ebn0_db)
    b = rand_bits(N)
    r = add_awgn(differential_encode(b), N0)
    r = r * np.exp(1j * np.deg2rad(theta_deg + drift_deg * np.arange(N + 1)))
    return calculate_ber(b, dpsk_detect(r)[0])[1]


def qpsk_theory(theta_deg, ebn0_db):
    A = np.sqrt(2 * db2lin(ebn0_db)); t = np.deg2rad(theta_deg)
    return 0.5 * (Qf(A * (np.cos(t) - np.sin(t))) + Qf(A * (np.cos(t) + np.sin(t))))


def bpsk_theory(theta_deg, ebn0_db):
    return Qf(np.sqrt(2 * db2lin(ebn0_db)) * np.cos(np.deg2rad(theta_deg)))


def plot_constellation(ax, r, cls, title, lim, ideal, labels):
    for i, c in enumerate(np.unique(cls)):
        m = cls == c
        ax.scatter(r[m].real, r[m].imag, s=6, color=COLS[int(c) % 4], label=labels[i])
    if ideal is not None:
        ax.plot(np.real(ideal), np.imag(ideal), "kx", ms=11, mew=2, label="ideal points")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(True, alpha=0.4)
    ax.set_xlabel("In-phase (I) amplitude (normalized)"); ax.set_ylabel("Quadrature (Q) amplitude (normalized)")
    ax.set_title(title, fontsize=10); ax.legend(loc="upper right", fontsize=7)


def plot_stat_hist(ax, stat, m1, m0, title, xlabel, l1, l0):
    edges = np.linspace(stat.min(), stat.max(), 61)
    ax.hist(stat[m1], bins=edges, density=True, alpha=0.55, label=l1)
    ax.hist(stat[m0], bins=edges, density=True, alpha=0.55, label=l0)
    ax.axvline(0, color="k", ls="--", label="Decision boundary (0)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Probability density (1/unit)"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

# %% Random data and BPSK (Part A)
print("=== EXPERIMENT 10: BPSK, QPSK, DPSK ===")
print(f"Es=1: Eb(BPSK)={EB_BPSK}, Eb(QPSK)={EB_QPSK}, Eb(DPSK)={EB_DPSK}\n")

bits = rand_bits(N_DEMO)
b1, b2 = bits[0::2], bits[1::2]
cls_q = 2 * b1 + b2                      # 0..3 -> 00,01,10,11
lab_q = ["00", "01", "10", "11"]

N0_b = EB_BPSK / db2lin(EBN0_DEMO)
s_bpsk = bpsk_mod(bits)
r_bpsk = add_awgn(s_bpsk, N0_b)
bits_hat_bpsk = bpsk_demod(r_bpsk)
nerr, ber = calculate_ber(bits, bits_hat_bpsk)
print(f"[BPSK] Eb/N0={EBN0_DEMO} dB: {nerr} errors in {bits.size} bits, BER={ber:.4e} "
      f"(theory {float(Qf(np.sqrt(2*db2lin(EBN0_DEMO)))):.4e})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
plot_constellation(ax[0], s_bpsk[:N_PLOT].astype(complex), bits[:N_PLOT], "Ideal BPSK constellation", 2,
                   bpsk_mod([0, 1]).astype(complex), ["bit 0 (-1)", "bit 1 (+1)"])
plot_constellation(ax[1], r_bpsk[:N_PLOT], bits[:N_PLOT], f"Noisy BPSK, Eb/N0 = {EBN0_DEMO} dB", 2,
                   bpsk_mod([0, 1]).astype(complex), ["bit 0 sent", "bit 1 sent"])
finish(fig, "fig01_bpsk_constellation")

# %% QPSK Gray mapping and MANDATORY validation (Part B)
N0_q = EB_QPSK / db2lin(EBN0_DEMO)
s_qpsk = qpsk_gray_mod(bits)
r_qpsk = add_awgn(s_qpsk, N0_q)

pairs = [(0, 0), (0, 1), (1, 1), (1, 0)]
sym_manual = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)   # typed by hand

print("\n--- MANDATORY VALIDATION: manual map of 00, 01, 11, 10 ---")
print("Bits | Tx symbol              | angle   | I_hat Q_hat | Detected | Mapper==manual | Correct?")
all_ok = True
for p, s_tx in zip(pairs, sym_manual):
    s_fn = qpsk_gray_mod(np.array(p))[0]
    map_ok = abs(s_tx - s_fn) < 1e-12
    b_hat, I_hat, Q_hat = qpsk_gray_demod(s_tx)
    ok = bool(map_ok and tuple(b_hat) == p)
    all_ok &= ok
    print(f" {p[0]}{p[1]}  | {s_tx.real:+.4f} {s_tx.imag:+.4f}j     | {np.degrees(np.angle(s_tx)):+7.1f} |"
          f"  {I_hat[0]:+d}    {Q_hat[0]:+d}   |   {b_hat[0]}{b_hat[1]}     |      {int(map_ok)}         |  {'YES' if ok else 'NO'}")
checks.append(("QPSK mandatory noiseless validation (00,01,11,10)", all_ok))

NTR, EB_VAL = 5000, 12
print(f"\nNoisy repeat: Eb/N0 = {EB_VAL} dB, {NTR} trials per input pair")
for p, s_tx in zip(pairs, sym_manual):
    rr = add_awgn(s_tx * np.ones(NTR), EB_QPSK / db2lin(EB_VAL))
    bh = qpsk_gray_demod(rr)[0].reshape(-1, 2)
    nbad = int(np.sum(np.any(bh != np.array(p), axis=1)))
    print(f"  input {p[0]}{p[1]}: {nbad} symbol errors / {NTR} (SER = {nbad/NTR:.4f})")

checks.append(("BPSK mapper/demapper are inverses", np.array_equal(bpsk_demod(bpsk_mod(bits)), bits)))
checks.append(("QPSK mapper/demapper are inverses (random data)", np.array_equal(qpsk_gray_demod(qpsk_gray_mod(bits))[0], bits)))
checks.append(("QPSK mean symbol energy = 1 (Es=1, Eb=0.5)", abs(np.mean(np.abs(s_qpsk) ** 2) - 1) < 1e-12))

# %% QPSK detection, mapping diagram, constellations (Parts B, C)
bits_hat_qpsk = qpsk_gray_demod(r_qpsk)[0]
nerr, ber = calculate_ber(bits, bits_hat_qpsk)
ser = np.mean((b1 != bits_hat_qpsk[0::2]) | (b2 != bits_hat_qpsk[1::2]))
print(f"\n[QPSK] Eb/N0={EBN0_DEMO} dB: {nerr} bit errors, BER={ber:.4e} "
      f"(theory {float(Qf(np.sqrt(2*db2lin(EBN0_DEMO)))):.4e}); SER={ser:.4e}")

fig, ax = plt.subplots(figsize=(5.8, 5.4))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color="gray"))
for i, (p, s) in enumerate(zip(pairs, sym_manual)):
    ax.plot([0, s.real], [0, s.imag], color=COLS[i], lw=1.2)
    ax.plot(s.real, s.imag, "o", ms=12, mfc=COLS[i], mec="k")
    ax.text(s.real + 0.12 * np.sign(s.real), s.imag + 0.12 * np.sign(s.imag),
            f"{p[0]}{p[1]}\n{np.degrees(np.angle(s)):+.0f}°", ha="center", va="center", fontweight="bold")
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.grid(True, alpha=0.4)
ax.set_xlabel("In-phase (I) amplitude (normalized, Es = 1)"); ax.set_ylabel("Quadrature (Q) amplitude (normalized)")
ax.set_title("Gray-coded QPSK mapping: bit pair (b1 b2) and phase angle")
finish(fig, "fig02_qpsk_mapping")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8))
plot_constellation(ax[0], s_qpsk[:N_PLOT], cls_q[:N_PLOT], "Ideal QPSK constellation", 1.6, sym_manual, lab_q)
plot_constellation(ax[1], r_qpsk[:N_PLOT], cls_q[:N_PLOT],
                   f"Noisy QPSK, Eb/N0 = {EBN0_DEMO} dB (axes = decision regions)", 1.6, sym_manual, lab_q)
finish(fig, "fig03_qpsk_constellation")

# %% Differential encoding and DPSK detection (Parts D, E)
ex_bits = np.array([1, 0, 1, 1, 0, 0, 1])
ex_b = 2 * ex_bits - 1
ex_d = np.zeros(ex_bits.size + 1); ex_d[0] = 1.0
for k in range(ex_bits.size):
    ex_d[k + 1] = ex_d[k] * ex_b[k]                 # explicit recursion
ex_bhat = ((ex_d[1:] * ex_d[:-1]) >= 0).astype(int)
print("\n--- Differential encoding example (reference d0 = +1) ---")
print("bits    :", ex_bits); print("b[k]    :", ex_b); print("d[0..7] :", ex_d.astype(int)); print("detected:", ex_bhat)
checks.append(("Differential encoder loop == cumprod version", np.array_equal(ex_d, differential_encode(ex_bits))))
checks.append(("Differential hand example decoded correctly", np.array_equal(ex_bhat, ex_bits)))

N0_d = EB_DPSK / db2lin(EBN0_DEMO)
d_tx = differential_encode(bits)
r_d0 = add_awgn(d_tx, N0_d)
bits_hat_dpsk, z_d0 = dpsk_detect(r_d0)
nerr, ber = calculate_ber(bits, bits_hat_dpsk)
print(f"\n[DPSK] Eb/N0={EBN0_DEMO} dB: {nerr} errors, BER={ber:.4e} (theory 0.5*exp(-Eb/N0) = {0.5*math.exp(-float(db2lin(EBN0_DEMO))):.4e})")
checks.append(("DPSK noiseless round trip", np.array_equal(dpsk_detect(d_tx)[0], bits)))

# %% Phase-offset experiment (Part F, Tasks 38-39)
print(f"\n--- PHASE OFFSET at Eb/N0 = {EBN0_DEMO} dB (same data and noise for every offset) ---")
print("theta | BPSK sim  theory | QPSK sim  theory | DPSK sim (const) | DPSK decisions changed")
dpsk_invariant = True
for th in ROT_DEG:
    bB = calculate_ber(bits, bpsk_demod(apply_phase_offset(r_bpsk, th)))[1]
    bQ = calculate_ber(bits, qpsk_gray_demod(apply_phase_offset(r_qpsk, th))[0])[1]
    bh, _ = dpsk_detect(apply_phase_offset(r_d0, th))
    bD = calculate_ber(bits, bh)[1]
    nchg = int(np.sum(bh != bits_hat_dpsk)); dpsk_invariant &= (nchg == 0)
    print(f"{th:4d}  | {bB:.4f}  {float(bpsk_theory(th, EBN0_DEMO)):.4f}   | {bQ:.4f}  {float(qpsk_theory(th, EBN0_DEMO)):.4f}   |"
          f"  {bD:.4f}          | {nchg}")
checks.append(("DPSK decisions unchanged by common constant offset", dpsk_invariant))

bh180 = bpsk_demod(apply_phase_offset(bpsk_mod(bits), 180))
bh180d = dpsk_detect(apply_phase_offset(differential_encode(bits), 180))[0]
print(f"\n180 deg ambiguity, noiseless: coherent BPSK BER = {np.mean(bh180 != bits):.3f}, DPSK BER = {np.mean(bh180d != bits):.3f}")

checks.append(("Degrees -> radians: 10 deg = 0.17453 rad", abs(np.angle(np.exp(1j * np.deg2rad(10))) - 0.174532925) < 1e-6))
s90 = apply_phase_offset(s_qpsk, 90)
checks.append(("90 deg rotation == multiplication by j", np.max(np.abs(s90 - 1j * s_qpsk)) < 1e-12))
bh90 = qpsk_gray_demod(s90)[0]
print(f"QPSK 90 deg noiseless: BER = {np.mean(bh90 != bits):.4f}; detected (b1,b2) = (b2, NOT b1): "
      f"{np.array_equal(bh90[0::2], b2) and np.array_equal(bh90[1::2], 1 - b1)}")

print("\nDiagnostic at Eb/N0 = 60 dB (effectively noiseless; avoids rounding ties at 90 deg):")
print("theta | BPSK BER | QPSK BER")
for th in [0, 10, 30, 45, 60, 90]:
    print(f"{th:4d}  | {sim_bpsk(N_DEMO, 60, th):.4f}   | {sim_qpsk(N_DEMO, 60, th)[0]:.4f}")

# Passband correlator cross-check
t = np.arange(SPS) / (SPS * RS)
ref = np.sqrt(2 / TS) * np.cos(2 * np.pi * FC * t)
print("\nPassband correlator check (b=+1, Eb=1): correlator output vs sqrt(Eb)cos(theta)")
for th in [0, 10, 30, 60, 90]:
    tx = np.sqrt(2 * EB_BPSK / TS) * np.cos(2 * np.pi * FC * t + np.deg2rad(th))
    print(f"  theta={th:3d}: {np.sum(ref*tx)*(TS/SPS):+.4f} vs {np.sqrt(EB_BPSK)*np.cos(np.deg2rad(th)):+.4f}")

# %% Constellation rotation (Part G)
r_rot0 = add_awgn(s_qpsk, EB_QPSK / db2lin(EBN0_ROT))
fig, axs = plt.subplots(2, 3, figsize=(15, 9.4))
for ax, th in zip(axs.flat, ROT_DEG):
    rr = apply_phase_offset(r_rot0, th)
    bq = calculate_ber(bits, qpsk_gray_demod(rr)[0])[1]
    plot_constellation(ax, rr[:N_PLOT], cls_q[:N_PLOT], f"theta = {th}°, measured BER = {bq:.3f}", 1.6,
                       apply_phase_offset(sym_manual, th), lab_q)
ax = axs.flat[5]
x = np.arange(len(ROT_DEG))
th_vals = [float(qpsk_theory(th, EBN0_ROT)) for th in ROT_DEG]
sim_vals = [sim_qpsk(N_PH, EBN0_ROT, th)[0] for th in ROT_DEG]
ax.bar(x - 0.2, th_vals, 0.4, label="Theory"); ax.bar(x + 0.2, sim_vals, 0.4, label="Simulation")
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in ROT_DEG])
ax.set_xlabel("Phase offset theta (degrees)"); ax.set_ylabel("Bit error rate (BER)")
ax.set_title(f"QPSK BER vs offset, Eb/N0 = {EBN0_ROT} dB"); ax.legend(); ax.grid(True, alpha=0.4)
finish(fig, "fig04_qpsk_rotation")

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, th in zip(axs.flat, PHASE_DEG):
    rr = apply_phase_offset(r_bpsk, th)
    bq = calculate_ber(bits, bpsk_demod(rr))[1]
    plot_constellation(ax, rr[:N_PLOT], bits[:N_PLOT], f"BPSK, theta = {th}°, BER = {bq:.3f}", 2,
                       apply_phase_offset(bpsk_mod([0, 1]).astype(complex), th), ["bit 0 sent", "bit 1 sent"])
finish(fig, "fig05_bpsk_offsets")

# %% Phase trajectory and differential phase (Part H)
kk = np.arange(N_TRAJ)
r_n = r_rot0[:N_TRAJ]
r_o = apply_phase_offset(r_n, 30)
fig, axs = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
axs[0].step(kk, np.degrees(np.angle(s_qpsk[:N_TRAJ])), "b-o", where="mid", ms=4)
axs[0].set_title("Ideal QPSK phase trajectory angle(s[k]) (wrapped to (-180°, 180°])")
axs[1].step(kk, np.degrees(np.angle(r_n)), "r-o", where="mid", ms=4, label="noisy")
axs[1].step(kk, np.degrees(np.angle(s_qpsk[:N_TRAJ])), "b--", where="mid", label="ideal")
axs[1].set_title(f"Noisy phase trajectory angle(r[k]), Eb/N0 = {EBN0_ROT} dB"); axs[1].legend()
axs[2].step(kk, np.degrees(np.angle(r_o)), "m-o", where="mid", ms=4, label="offset 30°")
axs[2].step(kk, np.degrees(np.angle(r_n)), "r--", where="mid", label="no offset")
axs[2].set_title("Noisy trajectory with constant 30° offset (shift of +30°, wraps at ±180°)"); axs[2].legend()
for a in axs:
    a.set_ylim(-200, 200); a.set_ylabel("Phase (degrees)"); a.grid(True, alpha=0.4)
axs[2].set_xlabel("Symbol index k")
finish(fig, "fig06_phase_trajectory")
est = np.degrees(np.angle(np.sum(r_o * np.conj(r_n))))
print(f"\nPhase-trajectory check: estimated common rotation = {est:.4f} deg (applied 30)")
checks.append(("Estimated rotation equals applied 30 deg", abs(est - 30) < 1e-6))

r_dn = r_d0[:N_TRAJ + 1]
r_dc = apply_phase_offset(r_dn, 60)
r_df = r_dn * np.exp(1j * np.deg2rad(30 * np.arange(N_TRAJ + 1)))
dphi = lambda r: np.degrees(np.angle(r[1:] * np.conj(r[:-1])))
fig, axs = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
axs[0].stem(kk, dphi(d_tx[:N_TRAJ + 1]), basefmt=" ")
axs[0].set_title("Ideal DPSK phase difference: 0° (bit 1) or 180° (bit 0)")
axs[1].stem(kk, dphi(r_dn), basefmt=" ", linefmt="r-", markerfmt="ro", label="theta = 0")
axs[1].plot(kk, dphi(r_dc), "kx", ms=9, label="theta = 60° (constant)")
axs[1].set_title("Noisy delta-phi[k]: circles = no offset, crosses = constant 60° offset (identical)"); axs[1].legend()
axs[2].stem(kk, dphi(r_df), basefmt=" ", linefmt="m-", markerfmt="mo", label="drift 30°/symbol")
axs[2].plot(kk, dphi(r_dn), "ko", mfc="none", ms=6, label="no drift")
axs[2].set_title("30°/symbol phase drift (frequency offset): delta-phi shifted by +30°"); axs[2].legend()
for a in axs:
    a.set_ylim(-200, 200); a.set_ylabel("delta-phi (degrees)"); a.grid(True, alpha=0.4)
axs[2].set_xlabel("Symbol index k")
finish(fig, "fig07_differential_phase")

# %% Decision statistics (Part I)
m1, m0 = bits == 1, bits == 0
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
print(f"\n--- BPSK decision statistic Re(r), Eb/N0 = {EBN0_DEMO} dB (theory: mean +-cos(theta), std {math.sqrt(N0_b/2):.4f}) ---")
for ax, th in zip(axs.flat, [0, 30, 60, 90]):
    st = np.real(apply_phase_offset(r_bpsk, th))
    plot_stat_hist(ax, st, m1, m0, f"BPSK Re(r), theta = {th}°", "Decision statistic Re(r)", "bit 1 sent", "bit 0 sent")
    ov = (np.sum(st[m1] < 0) + np.sum(st[m0] >= 0)) / st.size
    print(f"theta={th:2d}: mean(1)={st[m1].mean():+.4f} mean(0)={st[m0].mean():+.4f} std={st[m1].std():.4f} overlap(=BER)={ov:.4f}")
finish(fig, "fig08_bpsk_statistics")

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
print(f"\n--- QPSK decision statistics (class = sign of the component sent), std theory {math.sqrt(N0_q/2):.4f} ---")
for j, th in enumerate([0, 30]):
    rr = apply_phase_offset(r_qpsk, th); sI, sQ = rr.real, rr.imag
    plot_stat_hist(axs[0, j], sI, b2 == 0, b2 == 1, f"QPSK Re(r), theta = {th}°", "Decision statistic Re(r)", "I=+1 (b2=0)", "I=-1 (b2=1)")
    plot_stat_hist(axs[1, j], sQ, b1 == 0, b1 == 1, f"QPSK Im(r), theta = {th}°", "Decision statistic Im(r)", "Q=+1 (b1=0)", "Q=-1 (b1=1)")
    print(f"theta={th:2d}: I means (+,-)=({sI[b2==0].mean():+.4f},{sI[b2==1].mean():+.4f}) std={sI[b2==0].std():.4f}")
finish(fig, "fig09_qpsk_statistics")

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
print(f"\n--- DPSK decision statistic Re(z), Eb/N0 = {EBN0_DEMO} dB ---")
for ax, (thc, dr) in zip(axs.flat, [(0, 0), (60, 0), (0, 30), (0, 60)]):
    rr = r_d0 * np.exp(1j * np.deg2rad(thc + dr * np.arange(r_d0.size)))
    bh, z = dpsk_detect(rr); sD = z.real
    plot_stat_hist(ax, sD, m1, m0, f"DPSK Re(z): const {thc}°, drift {dr}°/symbol", "Decision statistic Re(r[k] r*[k-1])", "bit 1 sent", "bit 0 sent")
    print(f"const={thc:2d} drift={dr:2d}: mean(1)={sD[m1].mean():+.4f} mean(0)={sD[m0].mean():+.4f} std={sD[m1].std():.4f} BER={np.mean(bh != bits):.4f}")
finish(fig, "fig10_dpsk_statistics")

# %% BER vs Eb/N0 (Part J)
nE = EBN0_VEC.size
ber_sim = np.zeros((3, nE)); ser_q = np.zeros(nE)
for i, e in enumerate(EBN0_VEC):
    ber_sim[0, i] = sim_bpsk(N_BER, e)
    ber_sim[1, i], ser_q[i] = sim_qpsk(N_BER, e)
    ber_sim[2, i] = sim_dpsk(N_BER, e)
eth = np.arange(0, 14.01, 0.25); g = db2lin(eth)
th_bpsk = Qf(np.sqrt(2 * g)); th_dpsk = 0.5 * np.exp(-g); th_ser = 1 - (1 - Qf(np.sqrt(2 * g))) ** 2
nz = lambda v: np.where(v == 0, np.nan, v)

fig, ax = plt.subplots(figsize=(9, 6.4))
ax.semilogy(eth, th_bpsk, "b-", label="Theory BPSK = Gray QPSK BER: Q(sqrt(2Eb/N0))")
ax.semilogy(eth, th_dpsk, "r-", label="Theory DBPSK: 0.5 exp(-Eb/N0)")
ax.semilogy(eth, th_ser, "k--", label="Theory QPSK SER: 1-(1-Q)^2")
ax.semilogy(EBN0_VEC, nz(ber_sim[0]), "bo", ms=8, mfc="none", label="Sim coherent BPSK")
ax.semilogy(EBN0_VEC, nz(ber_sim[1]), "gs", ms=9, mfc="none", label="Sim Gray QPSK (BER)")
ax.semilogy(EBN0_VEC, nz(ber_sim[2]), "r^", ms=8, mfc="none", label="Sim DPSK")
ax.semilogy(EBN0_VEC, nz(ser_q), "kx", ms=8, label="Sim QPSK SER (not BER)")
ax.axhline(1 / N_BER, color="gray", ls=":", label="1/N_bits resolution")
ax.set_xlim(0, 14); ax.set_ylim(1e-6, 1); ax.grid(True, which="both", alpha=0.4)
ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("Error rate (BER unless stated)")
ax.set_title(f"BER vs Eb/N0 ({N_BER:.0e} bits per point; missing points = 0 errors)"); ax.legend(fontsize=8, loc="lower left")
finish(fig, "fig11_ber_comparison")
print("\n--- BER vs Eb/N0 (simulation | theory) ---")
print("Eb/N0 | BPSK sim  theory | QPSK sim  theory | DPSK sim  theory | QPSK SER sim  theory")
for i, e in enumerate(EBN0_VEC):
    gg = float(db2lin(e)); q = float(Qf(math.sqrt(2 * gg)))
    print(f"{e:4d}  | {ber_sim[0,i]:.2e} {q:.2e} | {ber_sim[1,i]:.2e} {q:.2e} | {ber_sim[2,i]:.2e} {0.5*math.exp(-gg):.2e} | {ser_q[i]:.2e} {1-(1-q)**2:.2e}")

# %% BER under phase offsets (Part J)
offs = [0] + PHASE_DEG
ber_ph = np.zeros((3, len(offs), nE))
for i, e in enumerate(EBN0_VEC):
    for j, th in enumerate(offs):
        ber_ph[0, j, i] = sim_bpsk(N_PH, e, th)
        ber_ph[1, j, i] = sim_qpsk(N_PH, e, th)[0]
        ber_ph[2, j, i] = sim_dpsk(N_PH, e, th, 0)
names = ["Coherent BPSK", "Gray QPSK (coherent)", "DPSK (constant offset)"]
mk = ["o", "s", "^", "d", "v"]; lc = COLS + ["k"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
for s_i, ax in enumerate(axs):
    for j, th in enumerate(offs):
        ax.semilogy(EBN0_VEC, nz(ber_ph[s_i, j]), "-" + mk[j], color=lc[j], ms=6, label=f"theta = {th}°")
        if s_i < 2:
            tt = bpsk_theory(th, eth) if s_i == 0 else qpsk_theory(th, eth)
            ax.semilogy(eth, tt, "--", color="gray", lw=0.9)
    ax.set_xlim(0, 12); ax.set_ylim(1e-5, 1); ax.grid(True, which="both", alpha=0.4)
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("Bit error rate (BER)")
    ax.set_title(names[s_i] + ": BER vs Eb/N0 (dashed = theory)", fontsize=10); ax.legend(fontsize=8, loc="lower left")
finish(fig, "fig12_ber_phase_offsets")

sw = np.zeros((4, SWEEP_DEG.size))
for j, th in enumerate(SWEEP_DEG):
    sw[0, j] = sim_bpsk(N_PH, EBN0_FIX, th)
    sw[1, j] = sim_qpsk(N_PH, EBN0_FIX, th)[0]
    sw[2, j] = sim_dpsk(N_PH, EBN0_FIX, th, 0)
    sw[3, j] = sim_dpsk(N_PH, EBN0_FIX, 0, th)
fig, ax = plt.subplots(figsize=(9, 6.2))
ax.plot(SWEEP_DEG, sw[0], "bo-", label="Coherent BPSK (sim)")
ax.plot(SWEEP_DEG, sw[1], "gs-", label="Gray QPSK (sim)")
ax.plot(SWEEP_DEG, sw[2], "r^-", label="DPSK, constant offset (sim)")
ax.plot(SWEEP_DEG, sw[3], "m*-", label="DPSK, per-symbol drift (sim)")
ax.plot(SWEEP_DEG, bpsk_theory(SWEEP_DEG, EBN0_FIX), "b--", label="BPSK theory")
ax.plot(SWEEP_DEG, qpsk_theory(SWEEP_DEG, EBN0_FIX), "g--", label="QPSK theory")
ax.set_xlim(0, 180); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.4); ax.legend(loc="upper left", fontsize=8)
ax.set_xlabel("Phase offset theta (degrees); for the drift curve: degrees per symbol"); ax.set_ylabel("Bit error rate (BER)")
ax.set_title(f"BER vs carrier phase error at Eb/N0 = {EBN0_FIX} dB")
finish(fig, "fig13_ber_vs_angle")
print(f"\n--- BER vs angle at Eb/N0 = {EBN0_FIX} dB ---")
print("theta:        " + "".join(f"{v:6d}" for v in SWEEP_DEG))
for nm, row in zip(["BPSK", "QPSK", "DPSK(const)", "DPSK(drift)"], sw):
    print(f"{nm:<13s} " + "".join(f"{v:6.3f}" for v in row))

# %% Noise / normalisation checks and summary
nn = add_awgn(np.zeros(1_000_000), 0.37)
checks.append(("AWGN: var(Re)~N0/2, var(Im)~N0/2, E|n|^2~N0 (N0=0.37)",
               abs(nn.real.var() / 0.185 - 1) < 0.01 and abs(nn.imag.var() / 0.185 - 1) < 0.01 and abs(np.mean(np.abs(nn) ** 2) / 0.37 - 1) < 0.01))
checks.append(("BER inputs have equal length", bits.size == bits_hat_bpsk.size == bits_hat_qpsk.size == bits_hat_dpsk.size))
print("\n=== SELF-CHECK SUMMARY ===")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"Figures saved in ./{OUTDIR}" if SAVE_FIGS else "")
