import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tabulate import tabulate

DATAFILE = "Data_WithWO_V8.xlsx"


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def test_normality_rejected(values):
    _, pval = scipy.stats.shapiro(values)
    return pval < 0.05, pval


def compare_groups(
    df: pd.DataFrame, voi: str, descr: str, time_correction=True
) -> dict:
    """
    Compare variable of interest between two groups using an OLS regression or Mann-Whitney U test.
    The groups are defined by the "Phenotype" column in the DataFrame, which should be 0 or 1.

    Args:
        df: DataFrame with data (should contain columns "Phenotype", "Timepoint", "NeuronType" and voi)
        voi: Variable of interest (column name in df)
        descr: Description of the variable of interest (added to output)

    Returns:
        Dictionary with test results.
    """
    assert "Phenotype" in df.columns, "Phenotype column missing"
    if time_correction:
        assert "Timepoint" in df.columns, "Timepoint column missing"
    assert "NeuronType" in df.columns, "NeuronType column missing"
    assert voi in df.columns, f"{voi} column missing"
    assert df["Phenotype"].isin([0, 1]).all(), "Phenotype should be 0 or 1"

    df["Phenotype"] = df["Phenotype"].astype(int)

    # we first assume normality and use a linear regression including covariates
    if time_correction:
        model = smf.ols(f"{voi} ~ Phenotype + C(Timepoint) + C(NeuronType)", data=df)
    else:
        model = smf.ols(f"{voi} ~ Phenotype + C(NeuronType)", data=df)
    model = model.fit()
    pval = model.pvalues["Phenotype"]
    method = "OLS regression"
    x = df[df["Phenotype"] == 0][voi]
    y = df[df["Phenotype"] == 1][voi]
    residuals = model.resid
    d = cohen_d(y, x)

    # check if residuals are normally distributed, if not use Mann-Whitney U test to compute p-value
    normality_rejected, normality_pval = test_normality_rejected(residuals)
    if normality_rejected:
        print(
            f"Normality rejected for {descr} (p={normality_pval:.5f}), using Mann-Whitney U test instead"
        )

        # first regress out covariates
        if time_correction:
            model = smf.ols(f"{voi} ~ C(Timepoint) + C(NeuronType)", data=df).fit()
        else:
            model = smf.ols(f"{voi} ~ C(NeuronType)", data=df).fit()
        residuals = model.resid

        _, pval = scipy.stats.mannwhitneyu(
            residuals[df["Phenotype"] == 0],
            residuals[df["Phenotype"] == 1],
        )
        method = "Mann-Whitney U test"

    return {
        "descr": descr,
        "method": method,
        "pval": pval,
        "normality_rejected": normality_rejected,
        "normality_pval": normality_pval,
        "d": d,
        "n_pat": len(y),
        "n_ctrl": len(x),
        "covariates": (
            ["Timepoint", "NeuronType"] if time_correction else ["NeuronType"]
        ),
    }


def microglia_and_culture_health() -> pd.DataFrame:
    """
    Does microglia addition improve culture health compared to without microglia?
    In healthy cultures we expect:
        - more NeuN+ neurons (mature),
        - increased protrusions in axons,
        - increased protrusions in dendrites,
        - decreased debris/increased viable nuclei.

    We test all four aspects separately and combine p-values using Fisher's method.

    Due to potential compensatory mechanisms, we do not hypothesize a direction of effect.
    All tests are two-sided.
    """
    stats = []

    # Part 1: Neurons
    df = pd.read_excel(DATAFILE, "WithWO_Neurons")
    df["Phenotype"] = pd.Series(df["Phenotype"] == "withMG", dtype=int)
    stats.append(
        compare_groups(
            df,
            "NeunposNumber",
            "Neuronal health (NeuN+)",
        )
    )

    # Part 2: Axons
    df = pd.read_excel(DATAFILE, "WithWO_HumanandAxons")
    df["Phenotype"] = pd.Series(df["Phenotype"] == "withMG", dtype=int)
    df["neurite_segments"] = df[
        "mean_Neurite.Segments...Number.of.Objects"
    ]  # renaming required for smf
    stats.append(
        compare_groups(
            df,
            "neurite_segments",
            "Axons (Neurite.Segments...Number.of.Objects)",
        )
    )

    # Part 3: Dendrites
    df = pd.read_excel(DATAFILE, "WithWO_DendriteandSynapse")
    df["Phenotype"] = pd.Series(df["Phenotype"] == "withMG", dtype=int)
    df["neurite_segments"] = df[
        "Neurite.Segments...Number.of.Objects"
    ]  # renaming required for smf
    stats.append(
        compare_groups(
            df,
            "neurite_segments",
            "Dendrites (Neurite.Segments...Number.of.Objects)",
        )
    )

    # Part 4: Debris
    df = pd.read_excel(DATAFILE, "WithWO_Debris")
    df["Phenotype"] = pd.Series(df["Phenotype"] == "withMG", dtype=int)
    stats.append(
        compare_groups(
            df,
            "AllDebris",
            "Debris (AllDebris)",
        )
    )

    # Aggregate p-values.
    # We use Fisher's method to combine p-values from different tests.
    # This results in a single p-value, avoiding loss of power due to multiple testing.
    all_pvals = [s["pval"] for s in stats]
    logsum_pval = -2 * np.log(all_pvals).sum()
    aggregate_pval = scipy.stats.chi2.sf(logsum_pval, 2 * len(all_pvals))

    stats.append(
        {
            "descr": "AGGREGATE",
            "method": "Fisher",
            "pval": aggregate_pval,
        }
    )

    print(f"Microglia impact culture health | Aggregate p={aggregate_pval:.2e}")
    return pd.DataFrame(stats)


def compute_average_per_well(df: pd.DataFrame, property: str) -> pd.DataFrame:
    res = []
    for well in df["well"].unique():
        well_df = df.query(f"well == '{well}'")

        def _get_property(property):
            """Get property value for a well, checking it's the same over all rows."""
            vals = well_df[property].unique()
            assert len(vals) == 1, f"Multiple values for {property}: {vals}"
            return vals[0]

        well_res = {
            "well": well,
            f"mean_{property}": well_df[property].mean(),
            "Phenotype": _get_property("Phenotype"),
        }
        well_res["NeuronType"] = _get_property("NeuronType")
        res.append(well_res)

    res = pd.DataFrame(res)

    return res


def microglia_and_activity() -> pd.DataFrame:
    """
    Does microglia addition change activity compared to without microglia?
    """

    # Load and format data (one row per channel per timepoint)
    df = pd.read_excel("Data_WithWO_V8.xlsx", sheet_name="WithWO_Activity")
    df = df.rename(columns={"Active.Channel": "ActiveChannel", "Well.Label": "well"})
    df["Phenotype"] = pd.Series(df["Phenotype"] == "MG", dtype=int)
    df = df.query("time == 12")
    # df = df.query("time >= 7")
    print(f"Loaded {len(df)} raw measurements")

    # Compute the average "Active.Channel" per well
    df = compute_average_per_well(df, "ActiveChannel")
    print(f"Found data for {len(df)} wells")

    return pd.DataFrame(
        [
            compare_groups(
                df,
                "mean_ActiveChannel",
                "mean_ActiveChannel",
                time_correction=False,
            )
        ]
    )


def posthoc(sheet: str, dynamics=False) -> pd.DataFrame:
    """
    Run group comparison for all variables of interest in a given sheet.
    """
    df = pd.read_excel(DATAFILE, sheet_name=sheet)
    if dynamics:
        df["Phenotype"] = pd.Series(df["Phenotype"] == "MG", dtype=int)
        df["well"] = df["Well.Label"]
    else:
        df["Phenotype"] = pd.Series(df["Phenotype"] == "withMG", dtype=int)

    def clean_column_name(col):
        return (
            col.replace(" ", "_")
            .replace("...", ".")
            .replace(".", "_")
            .replace("__", "_")
            .replace("_µs_", "_micros_")
        )

    df = df.rename(columns=clean_column_name)

    if dynamics:
        df = df.query("time == 12")

    stats = []
    for col in df.columns:
        if col not in [
            "Phenotype",
            "Timepoint",
            "NeuronType",
            "WellName",
            "Well_Label",
            "Channel_Label",
            "time",
            "well",
            "mean_Neurite_Segments_Number_of_Objects",  # already included
            "Neurite_Segments_Number_of_Objects",  # already included
            "Active_Channel",  # already included
            "AllDebris",  # already included
        ] and not col.startswith("X"):
            if dynamics:
                stats.append(
                    compare_groups(
                        compute_average_per_well(df, col),
                        f"mean_{col}",
                        f"mean_{col}",
                        time_correction=False,
                    )
                )
            else:
                stats.append(
                    compare_groups(
                        df,
                        col,
                        col,
                        time_correction=True,
                    )
                )
    return pd.DataFrame(stats)


def neuron_background():
    df = pd.read_excel(DATAFILE, sheet_name="WithWO_Microglia")
    df = df.rename(columns={"TotalMicrogliaNo.": "TotalMicrogliaNo"})
    res = []
    for col in [
        "TotalMicrogliaNo",
        "AverageMicrogliaSize",
        "MicrogliaWidth",
        "MicrogliaRoundness",
        "MicrogliaLength",
        "MicrogliaWtoL",
    ]:
        model = smf.ols(
            f"{col} ~ C(NeuronType) + C(Timepoint) + C(NeuronType) : C(Timepoint)",
            data=df,
        ).fit()

        anova_table = sm.stats.anova_lm(model, typ=2)

        normality_rejected, normality_pval = test_normality_rejected(model.resid)

        res.append(
            {
                "Sheet": "WithWO_Microglia",
                "descr": f"{col} <> Neurontype",
                "method": "two-way ANOVA",
                "pval": anova_table["PR(>F)"]["C(NeuronType)"],
                "normality_rejected": normality_rejected,
                "normality_pval": normality_pval,
                "n": len(df),
                "covariates": ["Timepoint"],
            }
        )
        res.append(
            {
                "Sheet": "WithWO_Microglia",
                "descr": f"{col} <> Timepoint",
                "method": "two-way ANOVA",
                "pval": anova_table["PR(>F)"]["C(Timepoint)"],
                "normality_rejected": normality_rejected,
                "normality_pval": normality_pval,
                "n": len(df),
                "covariates": ["Timepoint"],
            }
        )
        res.append(
            {
                "Sheet": "WithWO_Microglia",
                "descr": f"{col} <> Neurontype:Timepoint",
                "method": "two-way ANOVA",
                "pval": anova_table["PR(>F)"]["C(NeuronType):C(Timepoint)"],
                "normality_rejected": normality_rejected,
                "normality_pval": normality_pval,
                "n": len(df),
                "covariates": [""],
            }
        )

    return pd.DataFrame(res)


if __name__ == "__main__":
    stats = []

    # MAIN ANALYSIS
    stats.append(microglia_and_culture_health())
    stats.append(microglia_and_activity())
    stats = pd.concat(stats, ignore_index=True)
    stats.to_csv("stats_main.csv", index=False)
    print("\nMAIN ANALYSIS")
    print(tabulate(stats, headers="keys", tablefmt="pretty"))

    # POS-HOC: statistical test per variable of interest
    stats_posthoc = []
    for sheet in [
        "WithWO_HumanandAxons",
        "WithWO_DendriteandSynapse",
        "WithWO_Debris",
        "WithWO_Activity",
    ]:
        stats_posthoc.append(posthoc(sheet, dynamics=(sheet == "WithWO_Activity")))
        stats_posthoc[-1].insert(0, "Sheet", sheet)
    stats_posthoc = pd.concat(stats_posthoc, ignore_index=True)
    stats_posthoc.to_csv("stats_posthoc.csv", index=False)
    print("\nPOST HOC RESULTS")
    print(tabulate(stats_posthoc, headers="keys", tablefmt="pretty"))

    stats_background = neuron_background()
    stats_background.to_csv("stats_neuron_background.csv", index=False)
    print("\nNEURONAL BACKGROUND ANALYSIS")
    print(tabulate(stats_background, headers="keys", tablefmt="pretty"))
