#!/usr/bin/env python3
"""Generate AUROC plots for SNOMED/diagnostic predictions across variants.

Reads `results/eval/<variant>/diagnostic_preds.csv` and computes AUROC per-label and saves figures to `docs/figures/`.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from multiprocessing import Pool, cpu_count
from uuid import uuid4
import os
import re
import functools


def compute_auc(y_true, y_score):
    mask = (~np.isnan(y_score)) & (~np.isnan(y_true))
    y_true = y_true[mask]
    y_score = y_score[mask]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return float('nan')
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]
    cum_TP = np.cumsum(y_true_sorted)
    cum_FP = np.cumsum(1 - y_true_sorted)
    uniq_scores, idx = np.unique(y_score_sorted, return_index=True)
    tpr = np.concatenate(([0.0], cum_TP[idx - 1] / P if P > 0 else np.zeros_like(idx), [1.0])) if P > 0 else np.array([0.0,1.0])
    fpr = np.concatenate(([0.0], cum_FP[idx - 1] / N if N > 0 else np.zeros_like(idx), [1.0])) if N > 0 else np.array([0.0,1.0])
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    auc_val = np.trapz(tpr_s, fpr_s)
    return auc_val


def benjamini_hochberg(pvals):
    """Return adjusted p-values using Benjamini-Hochberg FDR.
    pvals: list or numpy array
    returns: numpy array of p_adj in same order
    """
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    adj = pvals * n / ranks
    # ensure monotonic
    for i in range(n-2, -1, -1):
        adj[i] = min(adj[i], adj[i+1])
    adj[adj > 1.0] = 1.0
    return adj


def _init_worker_shared_args(y_true_, y_a_, y_b_, idx_mode='seeded', seed=42):
    global _BOOT_Y_TRUE, _BOOT_Y_A, _BOOT_Y_B, _BOOT_IDX_MODE, _BOOT_SEED
    _BOOT_Y_TRUE = y_true_
    _BOOT_Y_A = y_a_
    _BOOT_Y_B = y_b_
    _BOOT_IDX_MODE = idx_mode
    _BOOT_SEED = seed


def _init_worker_shared_shm(meta_true, meta_a, meta_b, idx_mode, seed):
    """Initializer for worker using SharedMemory segments. meta_* are dicts with name, shape, dtype"""
    global _BOOT_Y_TRUE, _BOOT_Y_A, _BOOT_Y_B, _BOOT_IDX_MODE, _BOOT_SEED, _SHM_HANDLES
    _BOOT_IDX_MODE = idx_mode
    _BOOT_SEED = seed
    try:
        from multiprocessing import shared_memory
        _SHM_HANDLES = []
        m_true = shared_memory.SharedMemory(name=meta_true['name'])
        arr_true = np.ndarray(tuple(meta_true['shape']), dtype=np.dtype(meta_true['dtype']), buffer=m_true.buf)
        _BOOT_Y_TRUE = arr_true
        _SHM_HANDLES.append(m_true)
        m_a = shared_memory.SharedMemory(name=meta_a['name'])
        arr_a = np.ndarray(tuple(meta_a['shape']), dtype=np.dtype(meta_a['dtype']), buffer=m_a.buf)
        _BOOT_Y_A = arr_a
        _SHM_HANDLES.append(m_a)
        m_b = shared_memory.SharedMemory(name=meta_b['name'])
        arr_b = np.ndarray(tuple(meta_b['shape']), dtype=np.dtype(meta_b['dtype']), buffer=m_b.buf)
        _BOOT_Y_B = arr_b
        _SHM_HANDLES.append(m_b)
    except Exception as e:
        print('Failed to attach shared memory in worker init:', e)
        raise


def _init_worker_shared_memmap(path_true, path_a, path_b, idx_mode, seed):
    global _BOOT_Y_TRUE, _BOOT_Y_A, _BOOT_Y_B, _BOOT_IDX_MODE, _BOOT_SEED
    _BOOT_IDX_MODE = idx_mode
    _BOOT_SEED = seed
    try:
        _BOOT_Y_TRUE = np.load(path_true, mmap_mode='r')
        _BOOT_Y_A = np.load(path_a, mmap_mode='r')
        _BOOT_Y_B = np.load(path_b, mmap_mode='r')
    except Exception as e:
        print('Failed to attach memmap in worker init:', e)
        raise


def _bootstrap_worker_idx(idx_arr):
    """Worker: compute AUC delta for a single bootstrap index array recorded in numpy"""
    ta = _BOOT_Y_TRUE[idx_arr]
    aa = _BOOT_Y_A[idx_arr]
    bb = _BOOT_Y_B[idx_arr]
    va = compute_auc(ta, aa)
    vb = compute_auc(ta, bb)
    if math.isnan(va) or math.isnan(vb):
        return None
    return va - vb


def _bootstrap_worker_iter(iter_idx):
    """Worker: generate bootstrap indices deterministically from seed+iter_idx and compute delta"""
    n = _BOOT_Y_TRUE.shape[0]
    rs = np.random.RandomState(_BOOT_SEED + int(iter_idx))
    idx_arr = rs.randint(0, n, size=n).astype(np.int32)
    ta = _BOOT_Y_TRUE[idx_arr]
    aa = _BOOT_Y_A[idx_arr]
    bb = _BOOT_Y_B[idx_arr]
    va = compute_auc(ta, aa)
    vb = compute_auc(ta, bb)
    if math.isnan(va) or math.isnan(vb):
        return None
    return va - vb


def bootstrap_auc_diff(y_true, y_a, y_b, n_boot=1000, seed=42, n_jobs=1, mmap_mode='none', mmap_dir='/tmp', idx_generation='seeded', y_true_meta=None, y_a_meta=None, y_b_meta=None):
    """Bootstrap paired AUC difference between variant a and b.
    Returns (obs_delta, p_value_two_sided, ci_lower, ci_upper, auc_a, auc_b)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    # Zero check: require both classes present
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    obs_a = compute_auc(y_true, y_a)
    obs_b = compute_auc(y_true, y_b)
    obs_delta = obs_a - obs_b
    # Make arrays and seed available to serial path and worker initializers
    global _BOOT_Y_TRUE, _BOOT_Y_A, _BOOT_Y_B, _BOOT_SEED, _BOOT_IDX_MODE
    _BOOT_Y_TRUE = y_true
    _BOOT_Y_A = y_a
    _BOOT_Y_B = y_b
    _BOOT_SEED = seed
    _BOOT_IDX_MODE = idx_generation
    # Multiprocessing plan: if n_jobs > 1 passed in via global settings, worker will process subset
    # Build tasks: either precompute index arrays or use iteration indexes which workers will deterministically sample
    if idx_generation == 'precompute':
        all_idxs = [rng.randint(0, n, size=n).astype(np.int32) for _ in range(n_boot)]
    else:
        all_idxs = list(range(n_boot))
    deltas = []
    valid_count = 0
    if n_jobs is None or n_jobs <= 1:
        for idx in all_idxs:
            if idx_generation == 'precompute':
                delta = _bootstrap_worker_idx(idx)
            else:
                delta = _bootstrap_worker_iter(idx)
            if delta is None:
                continue
            deltas.append(delta)
            valid_count += 1
    else:
        # Use multiprocessing pool with shared arrays
        try:
            # Use chosen shared/memmap init initializer
            if mmap_mode == 'shm' and y_true_meta is not None and 'name' in y_true_meta:
                initfn = _init_worker_shared_shm
                initargs = (y_true_meta, y_a_meta, y_b_meta, idx_generation, seed)
            elif mmap_mode == 'memmap' and y_true_meta is not None and 'path' in y_true_meta:
                initfn = _init_worker_shared_memmap
                initargs = (y_true_meta['path'], y_a_meta['path'], y_b_meta['path'], idx_generation, seed)
            else:
                initfn = _init_worker_shared_args
                initargs = (y_true, y_a, y_b, idx_generation, seed)
            with Pool(processes=n_jobs, initializer=initfn, initargs=initargs) as pool:
                worker_fn = _bootstrap_worker_idx if idx_generation == 'precompute' else _bootstrap_worker_iter
                for res in pool.imap_unordered(worker_fn, all_idxs, chunksize=max(1, len(all_idxs)//(n_jobs*2))):
                    if res is None:
                        continue
                    deltas.append(res)
                    valid_count += 1
        except Exception as e:
            print('Multiprocessing bootstrap failed, falling back to serial loop:', e)
            for idx in all_idxs:
                delta = _bootstrap_worker_idx(idx)
                if delta is None:
                    continue
                deltas.append(delta)
                valid_count += 1
    if valid_count == 0:
        return obs_delta, float('nan'), float('nan'), float('nan'), obs_a, obs_b
    deltas = np.array(deltas)
    # Two-sided p-value
    frac_le0 = np.mean(deltas <= 0)
    frac_gt0 = np.mean(deltas > 0)
    p_val = 2.0 * min(frac_le0, frac_gt0)
    # CI: basic percentile of deltas
    lower = np.percentile(deltas, 2.5)
    upper = np.percentile(deltas, 97.5)
    return obs_delta, p_val, lower, upper, obs_a, obs_b


def compute_midrank(x):
    """Compute midranks for DeLong's method (handles ties by midrank)."""
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    # compute ranks
    r = 1
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # mid rank
        mid = 0.5 * (r + r + (j - i))
        ranks[order[i:j + 1]] = mid
        r += j - i + 1
        i = j + 1
    return ranks


def fast_delong(y_true, *y_scores_list):
    """Compute AUCs and covariance matrix for multiple classifiers using DeLong's method.
    y_true: 1D array of 0/1
    y_scores_list: one or more 1D arrays (same length as y_true) - returns aucs, cov_matrix
    """
    y_true = np.asarray(y_true, dtype=int)
    m = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    if m == 0 or n == 0:
        return [float('nan')] * len(y_scores_list), np.full((len(y_scores_list), len(y_scores_list)), np.nan)

    k = len(y_scores_list)
    aucs = np.zeros(k)
    v10_list = []
    v01_list = []
    for idx, y_scores in enumerate(y_scores_list):
        y_scores = np.asarray(y_scores, dtype=float)
        pos = y_scores[y_true == 1]
        neg = y_scores[y_true == 0]
        # concatenate and compute midranks for this classifier
        concatenated = np.concatenate([pos, neg])
        ranks = compute_midrank(concatenated)
        r_pos = ranks[:m]
        r_neg = ranks[m:]
        auc_val = (np.sum(r_pos) - m * (m + 1) / 2.0) / (m * n)
        aucs[idx] = auc_val
        v10 = (r_pos - np.arange(1, m + 1)) / float(n)
        v01 = 1.0 - (r_neg - np.arange(1, n + 1)) / float(m)
        v10_list.append(v10)
        v01_list.append(v01)

    # covariance matrix
    cov = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cov_v10 = np.cov(v10_list[i], v10_list[j], ddof=1)[0, 1]
            cov_v01 = np.cov(v01_list[i], v01_list[j], ddof=1)[0, 1]
            cov[i, j] = cov_v10 / float(m) + cov_v01 / float(n)
    return aucs, cov


def delong_roc_test(y_true, y_scores_a, y_scores_b):
    """Compute DeLong test (z-test) comparing two correlated ROC AUCs using same labels.
    Returns auc_a, auc_b, z, p_two_sided
    """
    aucs, cov = fast_delong(y_true, y_scores_a, y_scores_b)
    auc_a = aucs[0]
    auc_b = aucs[1]
    var_a = cov[0, 0]
    var_b = cov[1, 1]
    cov_ab = cov[0, 1]
    if math.isnan(auc_a) or math.isnan(auc_b) or math.isnan(var_a) or math.isnan(var_b):
        return auc_a, auc_b, float('nan'), float('nan')
    se = math.sqrt(max(var_a + var_b - 2.0 * cov_ab, 0.0))
    if se == 0:
        return auc_a, auc_b, float('nan'), float('nan')
    z = (auc_a - auc_b) / se
    # two-sided p-value
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return auc_a, auc_b, z, p


def read_csv(path):
    with open(path, 'r') as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader]
    return rows


def _create_shm_meta(arr):
    try:
        from multiprocessing import shared_memory
    except Exception:
        return None, None
    arr = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    meta = {'name': shm.name, 'shape': arr.shape, 'dtype': str(arr.dtype), 'nbytes': arr.nbytes}
    return meta, shm


def _create_memmap_meta(arr, dirpath, prefix=None):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = 'memmap_' + str(uuid4())
    # sanitize prefix to safe filename
    safe_prefix = re.sub('[^0-9a-zA-Z_-]', '_', prefix)
    fpath = dirpath / f'{safe_prefix}.npy'
    # Save and reopen as memmap to standardize
    np.save(fpath, arr)
    meta = {'path': str(fpath), 'shape': arr.shape, 'dtype': str(arr.dtype)}
    return meta


def _cleanup_shm(shm_handles):
    for shm in shm_handles:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


def parse_labels_from_header(header):
    # header contains label_pred, label_true pairs
    labels = []
    for h in header:
        if h.endswith('_pred'):
            labels.append(h[:-5])
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['baseline', 'hybrid', 'physics'])
    parser.add_argument('--eval_dir', type=str, default='results/eval')
    parser.add_argument('--out_dir', type=str, default='docs/figures')
    parser.add_argument('--bootstrap_iters', type=int, default=1000, help='Number of bootstrap iterations for pairwise testing')
    parser.add_argument('--p_adjust', type=str, default='bh', choices=['none', 'bh'], help='Multiple-testing correction for p-values')
    parser.add_argument('--n_jobs', type=int, default=max(1, cpu_count() // 2), help='Number of parallel workers for bootstrap (0 serial)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrapping and sampling')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top labels to plot for AUC differences')
    parser.add_argument('--use_delong', action='store_true', help='Use DeLong correlated test for pairwise comparisons (more precise)')
    parser.add_argument('--bootstrap_mmap', type=str, choices=['none', 'shm', 'memmap'], default='none', help='Use shared memory (shm) or memmap for bootstrap arrays')
    parser.add_argument('--mmap_dir', type=str, default='/tmp', help='Directory to write memmap files if using memmap')
    parser.add_argument('--bootstrap_idx_generation', type=str, choices=['seeded', 'precompute'], default='seeded', help='How to generate bootstrap indices: seeded per-iteration or precompute all indices')
    parser.add_argument('--labels_json', type=str, default=None, help='Optional labels.json mapping file (list of SNOMED codes)')
    parser.add_argument('--scp_csv', type=str, default=None, help='Optional SCP statements CSV to map SNOMED code -> description')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auroc_by_variant = {}
    labels_order = None
    # also build maps keyed by variant -> {sample_idx: {label:(pred,true)}}
    preds_by_variant = {}
    labels_by_variant = {}
    for v in args.variants:
        csvp = eval_dir / v / 'diagnostic_preds.csv'
        if not csvp.exists():
            print(f"Missing diagnostic_preds.csv for variant {v}, expected at {csvp}")
            continue
        rows = read_csv(csvp)
        header = list(rows[0].keys()) if len(rows) > 0 else []
        labels = parse_labels_from_header(header)
        labels_by_variant[v] = labels
        labels_order = labels if labels_order is None else labels_order
        preds_map = {}
        y_true = {l: [] for l in labels}
        y_score = {l: [] for l in labels}
        for ridx, r in enumerate(rows):
            for l in labels:
                t = r.get(f'{l}_true')
                s = r.get(f'{l}_pred')
                try:
                    y_true[l].append(int(t) if t is not None else 0)
                except Exception:
                    y_true[l].append(0)
                try:
                    y_score[l].append(float(s) if s is not None else np.nan)
                except Exception:
                    y_score[l].append(np.nan)
            # Map sample index to (preds, true) for alignment
            if 'sample_idx' in r and r.get('sample_idx') not in (None, ''):
                sample_idx = int(r.get('sample_idx'))
            else:
                sample_idx = ridx
            preds_map[sample_idx] = {l: (y_score[l][-1], y_true[l][-1]) for l in labels}
        preds_by_variant[v] = preds_map
        label_aucs = {}
        for l in labels:
            label_aucs[l] = compute_auc(np.array(y_true[l]), np.array(y_score[l]))
        auroc_by_variant[v] = label_aucs

    # Build final labels list as either labels_order if present or union of labels across variants
    all_labels = set()
    for v, ls in labels_by_variant.items():
        all_labels.update(ls)
    if labels_order is None:
        labels = sorted(list(all_labels))
    else:
        labels = labels_order

    # Ensure preds_by_variant contains all labels for all sample entries (fill missing with NaNs)
    for v in preds_by_variant:
        for sid, mapping in preds_by_variant[v].items():
            for l in labels:
                if l not in mapping:
                    mapping[l] = (float('nan'), float('nan'))
    if labels_order is None:
        print('No labels found; exiting')
        return
    labels = labels_order
    variants = list(auroc_by_variant.keys())
    auc_matrix = np.zeros((len(variants), len(labels)))
    for i, v in enumerate(variants):
        for j, l in enumerate(labels):
            val = auroc_by_variant.get(v, {}).get(l, float('nan'))
            auc_matrix[i, j] = 0.5 if np.isnan(val) else val

    # Try to build human-readable tick labels if scp CSV provided
    scp_map = {}
    if args.scp_csv is not None:
        try:
            import csv as _csv
            with open(args.scp_csv, 'r') as fh:
                reader = _csv.DictReader(fh)
                for r in reader:
                    # The SCP file lists the code in the first CSV column (no header for that column in some exports)
                    # Prefer 'description' column where available
                    first_col = list(r.values())[0] if len(r) > 0 else None
                    code = first_col
                    desc = r.get('description') or r.get('SCP-ECG Statement Description') or ''
                    scp_map[code] = desc
        except Exception as e:
            print('Failed to read scp_csv:', e)
    # Optionally load labels.json to remap label_i -> SNOMED code if needed
    labels_json_map = None
    if args.labels_json is not None:
        try:
            import json as _json
            with open(args.labels_json, 'r') as fh:
                labels_json_map = _json.load(fh)
                if not isinstance(labels_json_map, list):
                    labels_json_map = None
        except Exception as e:
            print('Failed to read labels_json:', e)

    # Plot AUROC heatmap (code-only tick labels, descriptions in a side legend)
    # Dynamic figsize to accommodate many labels
    fig_w = max(12, int(len(labels) * 0.35))
    fig_h = 6
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Use GridSpec with a narrow legend column on the right
    try:
        # Reserve an extra column for the colorbar to avoid overlapping the legend
        gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 0.06, 0.22], wspace=0.04)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[0, 2])
    except Exception:
        # Fallback if GridSpec unavailable
        ax_main = fig.add_subplot(1, 1, 1)
        ax_leg = None

    im = ax_main.imshow(auc_matrix, aspect='auto', vmin=0.0, vmax=1.0, cmap='viridis')
    # Place the colorbar in a dedicated axes column to avoid legend overlap
    if 'ax_cbar' in locals():
        fig.colorbar(im, cax=ax_cbar, label='AUROC')
    else:
        fig.colorbar(im, ax=ax_main, label='AUROC')
    ax_main.set_yticks(range(len(variants)))
    ax_main.set_yticklabels(variants)

    # Helper to shorten descriptions for legend
    def _short(s, n=80):
        if s is None:
            return ''
        s = str(s)
        return s if len(s) <= n else s[: n-3] + '...'

    codes = []
    desc_map = {}
    import re as _re
    for l in labels:
        # If labels are of the form label_0 and a labels.json was provided, remap
        if labels_json_map is not None:
            m = _re.match(r'label_(\d+)$', str(l))
            if m:
                idx = int(m.group(1))
                if idx < len(labels_json_map):
                    lcode = labels_json_map[idx]
                else:
                    lcode = l
            else:
                lcode = l
        else:
            lcode = l
        desc = scp_map.get(lcode) or scp_map.get(l)
        codes.append(str(lcode))
        if desc:
            desc_map[str(lcode)] = _short(desc)

    ax_main.set_xticks(range(len(labels)))
    ax_main.set_xticklabels(codes, rotation=45, ha='right', fontsize=8)
    ax_main.set_title('Per-label AUROC across variants')

    # Build a textual legend with code: short description
    if ax_leg is not None:
        import textwrap as _textwrap
        ax_leg.axis('off')
        all_codes = list(codes)
        n_items = len(all_codes)
        # Choose columns based on item count (aim for <=12 items/column)
        cols = min(4, max(1, int(math.ceil(n_items / 12.0))))
        n_per_col = int(math.ceil(n_items / float(cols)))
        # Precompute x positions normalized in axis space [0,1]
        total_w = 0.95
        margin = 0.02
        col_w = total_w / cols
        legend_fontsize = 7 if cols <= 2 else 6
        for col in range(cols):
            start = col * n_per_col
            sub = all_codes[start:start + n_per_col]
            # Per-item y coordinate top->bottom with small vertical margin
            if len(sub) > 1:
                step = 0.92 / (len(sub) - 1)
            else:
                step = 0
            for i_s, c in enumerate(sub):
                desc = desc_map.get(c, '')
                # wrap description to keep legend width manageable
                wrapped = _textwrap.fill(desc, width=28)
                label_text = f"{c}: {wrapped}" if desc else f"{c}:"
                x_pos = margin + col * col_w
                y_pos = 0.98 - (i_s * step)
                ax_leg.text(x_pos, y_pos, label_text, va='top', ha='left', fontsize=legend_fontsize, family='monospace', transform=ax_leg.transAxes)
        ax_leg.text(margin, 1.02, 'Label legend', va='bottom', ha='left', fontsize=9, weight='bold', transform=ax_leg.transAxes)
    else:
        # If no separate legend axis, print a debug output
        try:
            print('Label legend:')
            for c in codes:
                print(c, '-', desc_map.get(c, ''))
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(out_dir / 'diagnostic_auroc_heatmap.png', dpi=200)
    print('Saved diagnostic AUROC heatmap to', out_dir / 'diagnostic_auroc_heatmap.png')

    # Also save a bar chart with top-10 labels by mean AUROC
    mean_auc = np.nanmean(auc_matrix, axis=0)
    top_idx = np.argsort(-mean_auc)[:10]
    top_labels = [labels[i] for i in top_idx]
    top_vals = mean_auc[top_idx]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_labels)), top_vals, color='tab:blue')
    plt.xticks(range(len(top_labels)), top_labels, rotation=45, ha='right')
    plt.ylabel('Mean AUROC across variants')
    plt.title('Top-10 SNOMED labels by mean AUROC')
    plt.tight_layout()
    plt.savefig(out_dir / 'diagnostic_top10_mean_auc.png', dpi=200)
    print('Saved diagnostic top10 AUROC chart to', out_dir / 'diagnostic_top10_mean_auc.png')

    # Save ROC curves for the top label across variants
    if len(top_labels) > 0:
        top_label = top_labels[0]
        plt.figure(figsize=(6,6))
        for v in variants:
            csvp = eval_dir / v / 'diagnostic_preds.csv'
            if not csvp.exists():
                continue
            rows = read_csv(csvp)
            scores = np.array([float(r.get(f'{top_label}_pred') or np.nan) for r in rows])
            trues = np.array([int(r.get(f'{top_label}_true') or 0) for r in rows])
            auc_val = compute_auc(trues, scores)
            # Compute simple ROC points for plotting
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(trues, scores)
            except Exception:
                # fallback simple estimator
                fpr = np.linspace(0,1,50)
                tpr = np.linspace(0,1,50)
            plt.plot(fpr, tpr, label=f"{v} (AUC={auc_val:.3f})")

        plt.plot([0,1],[0,1], linestyle='--', color='gray')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC for top diagnostic label: {top_label}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(out_dir / f'diagnostic_top_label_{top_label}_roc.png', dpi=200)
        print('Saved top-label ROC to', out_dir / f'diagnostic_top_label_{top_label}_roc.png')

    # Next: compute per-label pairwise statistical comparisons using bootstrap
    # We'll compare pairwise across all variants present
    pairs = []
    for i in range(len(variants)):
        for j in range(i+1, len(variants)):
            pairs.append((variants[i], variants[j]))

    pair_results = []
    per_variant_stats_map = []
    all_pvals = []
    for l in labels:
        # Build aligned arrays per variant
        # Find intersection of sample idxs across variants for which predictions and true labels are present
        valid_idx_sets = []
        for v in variants:
            pred_map = preds_by_variant[v]
            valid_idx = set([sid for sid, mapping in pred_map.items() if (not math.isnan(mapping.get(l, (np.nan, np.nan))[0])) and (not math.isnan(mapping.get(l, (np.nan, np.nan))[1]))])
            valid_idx_sets.append(valid_idx)
        common_samples = set.intersection(*valid_idx_sets) if len(valid_idx_sets) > 0 else set()
        if len(common_samples) == 0:
            continue
        idxs = sorted(list(common_samples))
        y_true_arr = None
        scores_by_variant = {}
        for v in variants:
            preds_map = preds_by_variant[v]
            scores = np.array([preds_map.get(i, {}).get(l, (float('nan'), float('nan')))[0] for i in idxs], dtype=float)
            trues = np.array([preds_map.get(i, {}).get(l, (float('nan'), float('nan')))[1] for i in idxs], dtype=float)
            scores_by_variant[v] = scores
            if y_true_arr is None:
                y_true_arr = trues
            else:
                # sanity check that trues match
                if not np.all(y_true_arr == trues):
                    print(f"Warning: true labels mismatch for label {l} across variants; using intersection's first variant's labels")
        # Setup shared/memmap arrays for y_true and per-variant scores if requested
        shm_handles = []
        y_true_meta = None
        variant_meta = {}
        if args.bootstrap_mmap == 'shm':
            try:
                y_true_meta, y_true_shm = _create_shm_meta(y_true_arr)
                shm_handles.append(y_true_shm)
                for v in variants:
                    meta, sh = _create_shm_meta(scores_by_variant[v])
                    variant_meta[v] = meta
                    shm_handles.append(sh)
            except Exception as e:
                print('Failed to create shared memory; falling back to none:', e)
                y_true_meta = None
                variant_meta = {}
        elif args.bootstrap_mmap == 'memmap':
            try:
                y_true_meta = _create_memmap_meta(y_true_arr, args.mmap_dir, prefix=f"y_true_{l}")
                for v in variants:
                    variant_meta[v] = _create_memmap_meta(scores_by_variant[v], args.mmap_dir, prefix=f"{v}_{l}")
            except Exception as e:
                print('Failed to create memmap; falling back to none:', e)
                y_true_meta = None
                variant_meta = {}
        # per-variant AUCs and CIs via bootstrap
        per_variant_stats = {}
        for v in variants:
            obs_auc = compute_auc(y_true_arr, scores_by_variant[v])
            _, _, lower, upper, auc_a, _ = bootstrap_auc_diff(y_true_arr, scores_by_variant[v], scores_by_variant[v], n_boot=args.bootstrap_iters // 5 if args.bootstrap_iters >= 200 else max(50, args.bootstrap_iters//5), seed=args.seed, n_jobs=args.n_jobs)
            per_variant_stats[v] = {'auc': obs_auc, 'ci_lower': lower, 'ci_upper': upper}

        # Pairwise tests
        for va, vb in pairs:
            a_scores = scores_by_variant[va]
            b_scores = scores_by_variant[vb]
            obs_delta, p_val, ci_low, ci_high, auc_a, auc_b = bootstrap_auc_diff(y_true_arr, a_scores, b_scores, n_boot=args.bootstrap_iters, seed=args.seed, n_jobs=args.n_jobs)
            # Optionally compute DeLong p-value for paired test
            p_val_delong = float('nan')
            if args.use_delong:
                try:
                    _, _, z, p_val_d = delong_roc_test(y_true_arr, a_scores, b_scores)
                    p_val_delong = p_val_d
                except Exception as e:
                    print(f"DeLong test failed for label {l} between {va} and {vb}: {e}")
            pair_results.append({
                'label': l,
                'variant_a': va,
                'variant_b': vb,
                'auc_a': auc_a,
                'auc_b': auc_b,
                'delta': obs_delta,
                'p_value': p_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_pos': int(np.sum(y_true_arr == 1)),
                'n_neg': int(np.sum(y_true_arr == 0))
            })
            pair_results[-1]['p_value_delong'] = p_val_delong
            if not math.isnan(p_val):
                all_pvals.append(p_val)
        # save per-variant stats for this label to accumulate
        for v in variants:
            per_variant_stats_map.append({
                'label': l,
                'variant': v,
                'auc': per_variant_stats[v]['auc'],
                'ci_lower': per_variant_stats[v]['ci_lower'],
                'ci_upper': per_variant_stats[v]['ci_upper'],
                'n_pos': int(np.sum(y_true_arr == 1)),
                'n_neg': int(np.sum(y_true_arr == 0))
            })
        # cleanup shared resources for this label
        if len(shm_handles) > 0:
            _cleanup_shm(shm_handles)
        if args.bootstrap_mmap == 'memmap' and y_true_meta is not None:
            try:
                os.remove(y_true_meta['path'])
            except Exception:
                pass
            for v in variant_meta:
                try:
                    os.remove(variant_meta[v]['path'])
                except Exception:
                    pass

    # Adjust p-values with Benjamini-Hochberg FDR across all pairwise tests
    if len(all_pvals) > 0 and args.p_adjust != 'none':
        adj = benjamini_hochberg(all_pvals)
        # Assign adjusted pvals back to pair_results in order
        k = 0
        for pr in pair_results:
            if not math.isnan(pr['p_value']):
                pr['p_value_adj'] = adj[k]
                k += 1
            else:
                pr['p_value_adj'] = float('nan')
    else:
        for pr in pair_results:
            pr['p_value_adj'] = pr['p_value']

    # Save pairwise results CSV
    csv_out = out_dir / 'diagnostic_label_pairwise_stats.csv'
    with open(csv_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        header = ['label', 'variant_a', 'variant_b', 'auc_a', 'auc_b', 'delta', 'p_value', 'p_value_adj', 'p_value_delong', 'ci_low', 'ci_high', 'n_pos', 'n_neg']
        w.writerow(header)
        for pr in pair_results:
            row_vals = [pr.get(h, '') for h in header]
            w.writerow(row_vals)
    print('Saved per-label pairwise statistics to', csv_out)

    # Print top significant differences
    sig = [pr for pr in pair_results if (not math.isnan(pr['p_value_adj'])) and pr['p_value_adj'] < 0.05]
    sig = sorted(sig, key=lambda x: x['p_value_adj'])
    if len(sig) > 0:
        print('\nTop significant per-label pairwise differences:')
        for s in sig[:10]:
            print(f"Label {s['label']}: {s['variant_a']} vs {s['variant_b']} | delta={s['delta']:.4f} | p_adj={s['p_value_adj']:.3g} | aucs=({s['auc_a']:.3f},{s['auc_b']:.3f})")

    # Save per-variant AUC + CI CSV
    csv_var_out = out_dir / 'diagnostic_label_variant_auc_cis.csv'
    with open(csv_var_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        header = ['label', 'variant', 'auc', 'ci_lower', 'ci_upper', 'n_pos', 'n_neg']
        w.writerow(header)
        for r in per_variant_stats_map:
            w.writerow([r[h] for h in header])
    print('Saved per-variant AUC CIs to', csv_var_out)

    # Plot AUC differences for top-k significant labels
    top_k = args.top_k
    # Choose labels by smallest adjusted p-value if available, else by mean AUC
    if len(sig) > 0:
        top_labels = [s['label'] for s in sig][:top_k]
    else:
        mean_auc = np.nanmean(auc_matrix, axis=0)
        top_idx = np.argsort(-mean_auc)[:top_k]
        top_labels = [labels[i] for i in top_idx]

    if len(top_labels) > 0:
        ncols = min(2, len(top_labels))
        nrows = (len(top_labels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for k_idx, lab in enumerate(top_labels):
            r = k_idx // ncols
            c = k_idx % ncols
            ax = axes[r][c]
            vals = []
            lowers = []
            uppers = []
            for v in variants:
                # find entry
                recs = [x for x in per_variant_stats_map if x['label'] == lab and x['variant'] == v]
                if len(recs) == 0:
                    vals.append(np.nan)
                    lowers.append(np.nan)
                    uppers.append(np.nan)
                else:
                    rec = recs[0]
                    vals.append(rec['auc'])
                    lowers.append(rec['ci_lower'])
                    uppers.append(rec['ci_upper'])
            x = np.arange(len(variants))
            vals = np.array(vals)
            lowers = np.array(lowers)
            uppers = np.array(uppers)
            # Ensure non-negative error bars (some CIs may be NaN or inverted due to small bootstrap samples)
            lower_err = np.maximum(vals - lowers, 0.0)
            upper_err = np.maximum(uppers - vals, 0.0)
            yerr = np.stack([lower_err, upper_err])
            ax.bar(x, vals, yerr=yerr, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(variants, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.set_ylabel('AUROC')
            ax.set_title(f'AUCs for {lab}')
        plt.tight_layout()
        out_fig = out_dir / 'diagnostic_topk_auc_differences.png'
        plt.savefig(out_fig, dpi=200)
        print('Saved AUC difference plots for top-k labels to', out_fig)


if __name__ == '__main__':
    main()
