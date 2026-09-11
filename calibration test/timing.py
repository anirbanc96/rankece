"""
Timing comparison for calibration tests.
"""

import os
import time

import numpy as np
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from utils import *
from tests import (
    rank_ece_asymptotic_test,
    rank_ece_finite_test,
    skce_linear_test,
    skce_ustat_test,
)

DEFAULT_SAVE_DIR = "results"


METHODS = {
    "rankECE (A)": lambda Z, Y: rank_ece_asymptotic_test(Z, Y),
    "rankECE (F)": lambda Z, Y: rank_ece_finite_test(Z, Y),
    "SKCE-L (G)": lambda Z, Y: skce_linear_test(
        Z, Y, kernel="gaussian", band="median"
    ),
    "SKCE-L (L)": lambda Z, Y: skce_linear_test(
        Z, Y, kernel="laplace", band="median"
    ),
    "SKCE-U (G)": lambda Z, Y: skce_ustat_test(
        Z, Y, kernel="gaussian", band="median", n_rep=50
    ),
    "SKCE-U (L)": lambda Z, Y: skce_ustat_test(
        Z, Y, kernel="laplace", band="median", n_rep=50
    ),
}


def time_one(fn, Z, Y, n_runs, fresh_data_each_run, prob_gen, n, rho, seed_offset=0):
    """
    Time one method repeatedly.

    The first run is discarded to reduce warm-up effects.
    Returns the mean runtime in milliseconds.
    """
    times = []

    for i in range(n_runs):
        # allows for generating freh data in each run if true.
        if fresh_data_each_run:
            np.random.seed(seed_offset + i)
            Z_i, Y_i = gen_data(n, rho, prob_fn=H1_PROB_FUNCS[prob_gen])
        else:
            Z_i, Y_i = Z, Y

        start = time.perf_counter()

        try:
            fn(Z_i, Y_i)
        except Exception:
            pass

        elapsed = time.perf_counter() - start

        if i >= 0:
            times.append(elapsed)

    return np.mean(times) * 1000 if times else np.nan


def time_table(n_values,rho=0.5,n_runs=25,fresh_data_each_run=False,prob_gen = "p = sin",seed=0,):
    """
    Construct a timing table.

    Rows    : methods
    Columns : sample sizes
    Entries : mean runtime (ms)
    """
    table = pd.DataFrame(
        index=list(METHODS.keys()),
        columns=n_values,
        dtype=float,
    )

    for n in n_values:
        np.random.seed(seed)
        Z_fixed, Y_fixed = gen_data(n, rho, prob_fn=H1_PROB_FUNCS[prob_gen])

        for name, fn in METHODS.items():
            table.loc[name, n] = time_one(
                fn,
                Z_fixed,
                Y_fixed,
                n_runs=n_runs,
                fresh_data_each_run=fresh_data_each_run,
                prob_gen = prob_gen,
                n=n,
                rho=rho,
                seed_offset=seed,
            )

    table.index.name = "Method"
    table.columns.name = "n"

    return table


def print_table(table, title=""):
    if title:
        print(f"\n{'=' * 10} {title} {'=' * 10}\n")

    print(table.round(3))
    print()


def save_table(table,
               save_dir=DEFAULT_SAVE_DIR,
               filename="timing_table_sin.csv"):
    """Save timing table as CSV."""
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)
    table.to_csv(path)

    print(f"Saved CSV table to {path}")


def save_table_pdf(table,
                   save_dir=DEFAULT_SAVE_DIR,
                   filename="timing_table_sin.pdf"):
    """Save timing table as a formatted PDF."""
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(path)
    elements = []

    elements.append(
        Paragraph("Time Comparison (ms)",
                  styles["Title"])
    )
    elements.append(Spacer(1, 12))

    data = [["Method"] + [str(c) for c in table.columns]]

    for method in table.index:
        row = [method]
        row.extend(f"{table.loc[method, c]:.3f}" for c in table.columns)
        data.append(row)

    pdf_table = Table(data)

    pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))

    elements.append(pdf_table)

    doc.build(elements)

    print(f"Saved PDF table to {path}")


if __name__ == "__main__":

    n_values = [100, 200, 300, 400, 500]

    prob_gen = "p = Z - Z^15"
    filename_csv = "timing_table_ZZ15.csv"
    filename_pdf = "timing_table_ZZ15.pdf"

    table = time_table(
        n_values=n_values,
        rho=0.5,
        n_runs=200,
        fresh_data_each_run=False,
        prob_gen = prob_gen
    )

    print_table(
        table,
        title="Time Comparison"
    )

    save_table(table, filename = filename_csv)
    save_table_pdf(table, filename = filename_pdf)