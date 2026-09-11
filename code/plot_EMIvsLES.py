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
                markersize=8
            )
        )

    ax.legend(handles=handles, frameon=False)


def plot_emi_les(ax, data, title=None):

    parties = {
        "Republican Party": "red",
        "Democratic Party": "blue"
    }

    for party, color in parties.items():

        df_temp = (
            data.loc[data["party_name"] == party,
                     ["rating_emb_pool_z", "log_les"]]
            .dropna()
        )

        ax.scatter(
            df_temp["rating_emb_pool_z"],
            df_temp["log_les"],
            s=2,
            alpha=0.05,
            color=color,
            label=PARTY_LABELS[party]
        )

        sns.regplot(
            data=df_temp,
            x="rating_emb_pool_z",
            y="log_les",
            scatter=False,
            color=color,
            ci=95,
            line_kws={"linewidth": 1.5},
            ax=ax
        )

    ax.set_xlabel("EMI")
    ax.set_ylabel("LES (log scale)")

    if title:
        ax.set_title(title)

    add_party_legend(ax)
    
df_mc = pd.read_csv("../data/memberlevel_sbertllms.csv")  
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

plot_emi_les(ax, df_mc)

plt.savefig(
    "../output/congress_individualEMIvsLes_pooled.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "../output/congress_individualEMIvsLes_pooled.svg",
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    "../output/congress_individualEMIvsLes_pooled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# House and senate

fig, axes = plt.subplots(
    2, 1,
    figsize=(7, 12),
    sharex=True,
    sharey=True
)

plot_emi_les(
    axes[0],
    df_mc[df_mc["chamber"] == "House"],
    title="House"
)

plot_emi_les(
    axes[1],
    df_mc[df_mc["chamber"] == "Senate"],
    title="Senate"
)

axes[0].set_xlabel("")

axes[1].legend().remove()

plt.tight_layout()

plt.savefig(
    "../output/congress_individualEMIvsLes_chambers.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    "../output/congress_individualEMIvsLes_chambers.svg",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "../output/congress_individualEMIvsLes_chambers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
