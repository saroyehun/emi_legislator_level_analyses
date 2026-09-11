import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
# Global plotting settings
plt.rcParams.update({
    "figure.figsize": (7, 5),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

PARTY_LABELS = {
    "Democratic Party": "Democrats",
    "Republican Party": "Republicans"
}

PARTY_COLORS = {
    "Democratic Party": "blue",
    "Republican Party": "red"
}

df_mc = pd.read_csv("../data/memberlevel_sbertllms.csv")
df_mc['abs_nominate_dim1'] = abs(df_mc['nominate_dim1'])

def filter_quantile_range(data, x, y, q=(0.01, 0.99)):
    df = data.dropna(subset=[x, y, "party_name"]).copy()

    x_low, x_high = df[x].quantile(q)
    y_low, y_high = df[y].quantile(q)

    return df[
        df[x].between(x_low, x_high) &
        df[y].between(y_low, y_high)
    ].copy()

def add_party_legend(ax):
    handles = []

    for party in PARTY_LABELS:
        handles.append(
            Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                color=PARTY_COLORS[party],
                label=PARTY_LABELS[party],
                markersize=2
            )
        )

    ax.legend(handles=handles, frameon=False)

def make_binned_means(data, x, y, bins=20):
    df = data.dropna(subset=[x, y, "party_name"]).copy()

    df["x_bin"] = pd.qcut(
        df[x],
        q=bins,
        duplicates="drop"
    )

    out = (
        df.groupby(["party_name", "x_bin"], observed=True)
          .agg(
              x_mean=(x, "mean"),
              y_mean=(y, "mean"),
              y_sd=(y, "std"),
              n=(y, "size")
          )
          .reset_index()
    )

    out["y_se"] = out["y_sd"] / out["n"] ** 0.5
    out["y_low"] = out["y_mean"] - 1.96 * out["y_se"]
    out["y_high"] = out["y_mean"] + 1.96 * out["y_se"]

    return out

def plot_party_relationship_binned(ax, data, title=None, bins=20, q=(0.01, 0.99)):

    parties = {
        "Republican Party": "red",
        "Democratic Party": "blue"
    }

    xvar = "abs_nominate_dim1"
    yvar = "rating_emb_pool_z"

    # filtered data: both binned points and regression use this same data
    data_plot = filter_quantile_range(data, xvar, yvar, q=q)

    binned = make_binned_means(data_plot, xvar, yvar, bins=bins)

    for party, color in parties.items():

        df_temp = data_plot.loc[
            data_plot["party_name"] == party,
            [xvar, yvar]
        ].dropna()

        df_bin = binned.loc[binned["party_name"] == party].copy()

        # binned means with CI
        ax.errorbar(
            df_bin["x_mean"],
            df_bin["y_mean"],
            yerr=[
                df_bin["y_mean"] - df_bin["y_low"],
                df_bin["y_high"] - df_bin["y_mean"]
            ],
            fmt="o",
            markersize=1,
            linewidth=1,
            capsize=1,
            color=color,
            alpha=0.3,      # lower transparency
            elinewidth=0.5,
            label=PARTY_LABELS[party]
        )

        # regression line on same filtered data
        sns.regplot(
            data=df_bin,
            x="x_mean",
            y="y_mean",
            scatter=False,
            #truncate=True,
            ci=95,
            color=color,
            line_kws={"linewidth": 1.0},
            ax=ax
        )

    ax.set_xlabel("Ideological extremity")
    ax.set_ylabel("Mean EMI")
    ax.set_xlim(-0.05, 0.95)
    if title:
        ax.set_title(title)

    add_party_legend(ax)


fig, ax = plt.subplots(figsize=(5, 2.5))

plot_party_relationship_binned(
    ax,
    df_mc,
    title=None,
    bins=50,
    q=(0.0, 1.0)
)

plt.tight_layout()

plt.savefig("../output/congress_individualvsnominate_pooled_binned.pdf", dpi=300, bbox_inches="tight")
plt.savefig("../output/congress_individualvsnominate_pooled_binned.svg", dpi=300, bbox_inches="tight")
plt.savefig("../output/congress_individualvsnominate_pooled_binned.png", dpi=300, bbox_inches="tight")

plt.close()

fig, axes = plt.subplots(
    2, 1,
    figsize=(7, 12),
    sharex=True,
    sharey=True
)

plot_party_relationship_binned(
    axes[0],
    df_mc[df_mc["chamber"] == "House"],
    title="House",
    bins=50,
    q=(0.0, 1.0)
)

plot_party_relationship_binned(
    axes[1],
    df_mc[df_mc["chamber"] == "Senate"],
    title="Senate",
    bins=50,
    q=(0.0, 1.0)
)

plt.tight_layout()

plt.savefig("../output/congress_individualvsnominate_chambers_binned.pdf", dpi=300, bbox_inches="tight")
plt.savefig("../output/congress_individualvsnominate_chambers_binned.svg", dpi=300, bbox_inches="tight")
plt.savefig("../output/congress_individualvsnominate_chambers_binned.png", dpi=300, bbox_inches="tight")

plt.close()
