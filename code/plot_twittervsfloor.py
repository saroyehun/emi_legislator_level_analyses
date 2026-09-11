import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

twitter_congress = pd.read_csv("../data/twitter_speech_sbert_llms.csv")
# aggregate
df_plot = (
    twitter_congress
    .groupby(["bioguide_id", "year_quarter", "channel", "party_name", "chamber"])["emi_mean"]
    .mean()
    .reset_index()
)

# reshape wide
df_wide = (
    df_plot
    .pivot_table(
        index=["bioguide_id", "year_quarter", "party_name", "chamber"],
        columns="channel",
        values="emi_mean"
    )
    .reset_index()
    .dropna(subset=["Twitter", "Floor"])
)

# major parties only
df_wide = df_wide[
    df_wide["party_name"].isin(["Republican Party", "Democratic Party"])
].copy()

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

def plot_twitter_floor(ax, data, title=None, marginal_means=True):
    parties = {
        "Republican Party": "red",
        "Democratic Party": "blue"
    }

    for party, color in parties.items():
        df_temp = data.loc[
            data["party_name"] == party,
            ["Twitter", "Floor"]
        ].dropna()

        ax.scatter(
            df_temp["Twitter"],
            df_temp["Floor"],
            s=5,
            alpha=0.15,
            color=color,
            label=PARTY_LABELS[party]
        )

        sns.regplot(
            data=df_temp,
            x="Twitter",
            y="Floor",
            scatter=False,
            ci=95,
            color=color,
            line_kws={"linewidth": 1.5},
            ax=ax
        )

        if marginal_means:
            ax.axvline(
                df_temp["Twitter"].mean(),
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )
            ax.axhline(
                df_temp["Floor"].mean(),
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )

    ax.set_xlabel("Twitter EMI")
    ax.set_ylabel("Floor EMI")

    if title:
        ax.set_title(title)

    add_party_legend(ax)
    
def plot_party_channel_distributions(data, axes):
    channel_colors = {
        "Twitter": "darkorange",
        "Floor": "black"
    }

    for ax, party in zip(axes, PARTY_LABELS):
        df_temp = data.loc[
            data["party_name"] == party,
            ["Twitter", "Floor"]
        ].dropna()

        for channel, color in channel_colors.items():
            sns.kdeplot(
                data=df_temp,
                x=channel,
                color=color,
                linewidth=1.8,
                fill=False,
                label=channel,
                ax=ax
            )

        ax.set_title(PARTY_LABELS[party])
        ax.set_xlabel("EMI")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)


# pooled
fig, ax = plt.subplots(constrained_layout=True)

plot_twitter_floor(ax, df_wide)

plt.savefig(
    "../output/twitter_floor_emi_pooled.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "../output/twitter_floor_emi_pooled.svg",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    "../output/twitter_floor_emi_pooled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# by chamber
fig, axes = plt.subplots(
    2, 1,
    figsize=(7, 12),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

plot_twitter_floor(
    axes[0],
    df_wide[df_wide["chamber"] == "H"],
    title="House"
)

plot_twitter_floor(
    axes[1],
    df_wide[df_wide["chamber"] == "S"],
    title="Senate"
)

axes[0].set_xlabel("")
axes[1].legend().remove()

#plt.tight_layout()

plt.savefig(
    "../output/twitter_floor_emi_chambers.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "../output/twitter_floor_emi_chambers.svg",
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    "../output/twitter_floor_emi_chambers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
