"""
analysis.py — Part 3: where are we high or low, and what are we missing?

Two questions the owner asked, answered from the dataset (Part 1) once every rival
item has a name on our list (Part 2):

1. **Prices.** For everything we and at least one rival both sell, in the same
   size: our price next to each rival's, their average, and where we stand —
   below all of them (the best place to be), below or above their average, or
   above all of them.
2. **Gaps.** What we sell that nobody else does (a reason to come to us), and what
   the rivals sell that we don't — ranked by how many of them sell it, because
   something all three offer is a stronger hint than something one does.

The docstrings say what each function returns; tests/test_pipeline.py says how it
is checked.
"""

import pandas as pd

from .settings import (
    OUR_CAFE, SIZES, AT_AVERAGE, BELOW_ALL, BELOW_AVERAGE, ABOVE_AVERAGE, ABOVE_ALL
)


def add_canonical(menus: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """`menus` plus a `canonical` column: the one name every café's item goes by.

    Our rows: their own `item`. Rival rows: the `matched_to` from `matches` for
    that (`cafe`, `item`) — missing for items the owner has not reviewed yet.
    Every row of `menus` is kept, in order; `menus` itself is not changed.
    """
    out = menus.merge(
        matches[["cafe", "item", "matched_to"]].drop_duplicates(),
        on=["cafe", "item"],
        how="left",
    )
    out["canonical"] = out["matched_to"].where(out["cafe"] != OUR_CAFE, out["item"])
    return out.drop(columns=["matched_to"])


def price_table(menus: pd.DataFrame) -> pd.DataFrame:
    """Price per (canonical item, size) per café.  *(Provided.)*

    Rows are (`canonical`, `size`), sizes in menu order (S, M, L, Each); columns
    are the cafés; values are prices. Rows with no canonical name (not reviewed
    yet) are left out.
    """
    known = menus.dropna(subset=["canonical"])
    table = known.pivot_table(index=["canonical", "size"], columns="cafe",
                              values="price", aggfunc="first")
    order = {size: rank for rank, size in enumerate(SIZES)}
    return table.sort_index(key=lambda level: level.map(order) if level.name == "size"
                            else level)


def price_position(ours: float, low: float, average: float, high: float) -> str:
    """Where our price stands against the rivals' prices for the same item and size.

    `low`, `average` and `high` are the rivals' lowest, average and highest price.
    Returns one of the five labels in settings — the names, not the text:

        ours below `low`                 -> BELOW_ALL       "👍👍 Lowest Price"
        ours above `high`                -> ABOVE_ALL       "👎👎 Highest Price"
        otherwise, ours below `average`  -> BELOW_AVERAGE   "👍 Lower than average"
        otherwise, ours above `average`  -> ABOVE_AVERAGE   "👎 Higher than average"
        otherwise (ours equals average)  -> AT_AVERAGE      "👉👈 Average"

    Check them in that order: with one rival, low, average and high are the same.
    """
    if ours < low:
        return BELOW_ALL
    elif ours > high:
        return ABOVE_ALL
    elif ours < average:
        return BELOW_AVERAGE
    elif ours > average:
        return ABOVE_AVERAGE
    else:
        return AT_AVERAGE


def compare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Our price next to every rival's, for every item and size we both sell.

    `prices` is price_table's result. Returns one row per (canonical, size) where
    we have a price *and* at least one rival does, in price_table's order, with
    columns:

        canonical  size  our_price  <one column per rival>  rival_avg  our_position

    - `our_price`: the settings.OUR_CAFE column, renamed.
    - the rival columns: every other column of `prices`, in the same order
      (missing where that rival doesn't sell it).
    - `rival_avg`: the average of the rivals' prices, ignoring missing ones,
      rounded to 2 places.
    - `our_position`: price_position(our_price, the rivals' lowest, rival_avg, the
      rivals' highest) for that row.

    Plain index.
    """
    comparison = prices.reset_index()
    comparison = comparison[comparison[OUR_CAFE].notna()].copy()

    rival_columns = [cafe for cafe in prices.columns if cafe != OUR_CAFE]
    rivals = comparison[rival_columns]
    comparison = comparison[rivals.notna().any(axis=1)].copy()
    rivals = comparison[rival_columns]

    our_price = comparison[OUR_CAFE].rename("our_price")
    rival_avg = rivals.mean(axis=1, skipna=True).round(2).rename("rival_avg")

    def row_position(row: pd.Series) -> str:
        '''
        Return our position for this row, using price_position.  The rivals' prices
        are the non-missing values in the rival columns for this row.
        '''
        rival_values = row[rival_columns].dropna()
        if rival_values.empty:
            return AT_AVERAGE
        return price_position(
            row[OUR_CAFE],
            rival_values.min(),
            rival_values.mean(),
            rival_values.max(),
        )

    our_position = comparison.apply(row_position, axis=1).rename("our_position")

    return pd.concat(
        [
            comparison[["canonical", "size"]],
            our_price,
            rivals,
            rival_avg,
            our_position,
        ],
        axis=1,
    ).reset_index(drop=True)


def gap_analysis(menus: pd.DataFrame) -> pd.DataFrame:
    """Which items only we sell, which only rivals sell, and which we both do.

    `menus` must have a `canonical` column (add_canonical). Rival rows with no
    canonical name (unreviewed) are left out. Returns one row per (canonical, kind):

        canonical  kind  where  rivals  sold_by

    - `where`: "only us", "only them" or "both" — an outer merge of our items with
      the rivals' items with `indicator=True`, its `_merge` column relabelled.
    - `rivals`: how many different rivals sell it (0 for "only us").
    - `sold_by`: those rivals' names, sorted, joined with ", " ("" for "only us").

    Sorted by `where` ("only them" first, then "only us", then "both"), then
    `rivals` (most first), then `canonical`; plain index.
    """
    ours = menus[menus["cafe"] == OUR_CAFE][["canonical", "kind"]].drop_duplicates()
    theirs = (
        menus[menus["cafe"] != OUR_CAFE][["canonical", "kind", "cafe"]]
        .drop_duplicates()
    )

    ours_unique = ours.drop_duplicates()
    theirs_unique = theirs[["canonical", "kind"]].drop_duplicates()
    merged = ours_unique.merge(
        theirs_unique,
        on=["canonical", "kind"],
        how="outer",
        indicator=True,
    )
    merged["where"] = merged["_merge"].map(
        {"left_only": "only us", "right_only": "only them", "both": "both"}
    )

    rival_summary = (
        theirs.groupby(["canonical", "kind"], as_index=False)
        .agg(
            rivals=("cafe", "nunique"),
            sold_by=("cafe", lambda values: ", ".join(sorted(pd.unique(values)))),
        )
    )

    out = merged[["canonical", "kind", "where"]].merge(
        rival_summary, on=["canonical", "kind"], how="left"
    )
    out["rivals"] = out["rivals"].fillna(0).astype(int)
    out["sold_by"] = out["sold_by"].fillna("")
    out.loc[out["where"] == "only us", "sold_by"] = ""
    out.loc[out["where"] == "only us", "rivals"] = 0
    out["where"] = pd.Categorical(
        out["where"],
        categories=["only them", "only us", "both"],
        ordered=True,
    )

    ordered = out.sort_values(
        by=["where", "rivals", "canonical"],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    ordered["where"] = ordered["where"].astype(str)
    return ordered[["canonical", "kind", "where", "rivals", "sold_by"]]
