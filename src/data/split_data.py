import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_grouped_splits(df, target_col="laptime", train_size=0.7, val_size=0.15, random_state=42):
    """
    Split by (raceid, driver), so the same race+driver pair never appears in train and val/test.
    Returns: train_df, val_df, test_df
    """
    df = df.copy()
    df["group_id"] = df["raceid"].astype(str) + "_" + df["driver"]

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    groups = df["group_id"]

    # train vs temp
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_y = temp_df[target_col]
    temp_groups = temp_df["group_id"]

    # val vs test from temp
    remaining_fraction = val_size / (1.0 - train_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=remaining_fraction, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, temp_y, groups=temp_groups))

    train_df = df.iloc[train_idx].copy()
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # final sanity check: no overlapping group_id across splits
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    return train_df, val_df, test_df
