import re
import matplotlib.pyplot as plt
import itertools as it
from rdkit import Chem
from tqdm import tqdm
from math import ceil, sqrt
from pathlib import Path
from collections import Counter, defaultdict
from compare_runs import get_clear_val_canon_smiles

plt.rcParams["text.usetex"] = True


def main():
    all_generated_files = list(Path(".").rglob("*generated_smiles.txt")) + list(
        Path(".").rglob("*generated_samples.txt")
    )

    for max_num_atoms in tqdm(
        [
            30,
            25,
            20,
            12,
            8,
        ],
        desc="Plotting molecule history",
    ):
        filtered_files = [
            x
            for x in all_generated_files
            if not str(x).startswith(".backup")
            and (
                f"{max_num_atoms}_generated_smiles.txt" in str(x)
                or f"_{max_num_atoms}/generated_samples.txt" in str(x)
            )
        ]
        plot_history(max_num_atoms, filtered_files)


def load_smiles_as_canon(max_num_atoms: int) -> list:
    smiles = []
    for line in open(f"/app/data/molecules/size_{max_num_atoms}/valid.smi"):
        # mol = Chem.MolFromSmiles(line.strip())
        canon = Chem.CanonSmiles(line)
        smiles.append(canon)
    return smiles


def convert_smiles_to_canon(smiles: list) -> list:
    converted = []
    errored = 0
    for smile in smiles:
        try:
            canon = Chem.CanonSmiles(smile)
        except:
            errored += 1
            print("Can not convert", smile)
            continue
        converted.append(canon)
    if errored:
        print("Errored", errored)
    return converted


def plot_history(
    max_num_atoms: int,
    files: list,
):
    print(f"Plotting history for {max_num_atoms} atoms")
    val_canons = load_smiles_as_canon(max_num_atoms)

    group_to_methods = {
        "everything": (
            "DiGress discrete",
            "DiGress continuous",
            "Data Efficient Grammar",
            "MoLeR",
        ),
    }

    group_to_fig_axe = {
        key: plt.subplots(1, 1, figsize=(12, 12)) for key in group_to_methods.keys()
    }

    for _, ax in group_to_fig_axe.values():
        ax.set_ylim([0, 1])
        ax.set_xlabel("Number of generated unique molecules sorted by frequency")
        ax.set_ylabel("Ratio of molecules generated from validation dataset")

    molecules_counters = []
    methods = set()

    method_to_linestyle = {
        "DiGress discrete": ("-", "tab:cyan"),
        "DiGress continuous": ("-", "orange"),
        "Data Efficient Grammar": ("-", "tab:olive"),
        "MoLeR": ("-", "tab:blue"),
        # f"ds-q-{max_num_atoms}-graphpot": ("-", "wheat"),
        # f"ds-q-{max_num_atoms}-extragraphpot": ("-", "aqua"),
        # f"ds-q-{max_num_atoms}": ("-", "tab:red"),
        # f"ds-q-{max_num_atoms} max-sum": ("-.", "tab:green"),
        # f"ds-q-{max_num_atoms} sum-max": ("-.", "tab:purple"),
        # f"orig-{max_num_atoms}": (":", "tab:brown"),
        # f"orig-seventh-depth-{max_num_atoms}": (":", "tab:pink"),
    }
    method_to_order = defaultdict(lambda: 1)

    # End of configs

    method_to_data = {}
    method_to_highest_generated = defaultdict(lambda: 0)
    allowed_methods = [m for methods_ in group_to_methods.values() for m in methods_]

    for file in files:
        method = filepath_to_title(file)
        print("Processing", method, file)
        methods.add(method)

        if method not in allowed_methods:
            print(method, " not in ", allowed_methods)
            continue

        style, color = method_to_linestyle.get(method, ("-", "gray"))

        history = get_molecule_history(file)
        history_dedup = dedup_by(history, lambda x: x)

        molecule_counter = Counter(history)
        molecules_counters.append((method, molecule_counter))

        generated_molecules_sorted = sorted(
            molecule_counter.keys(),
            key=lambda x: molecule_counter[x],
            reverse=True,
        )

        sorted_val_sizes, sorted_val_percs = get_cumulative_perc_deduplicated(
            generated_molecules_sorted, val_canons
        )

        # If there are multiple same runs, take one that were running the longest (generated most molecules)
        if method_to_highest_generated[method] >= len(sorted_val_sizes):
            print("Warning - skipping", method, file)
            continue

        method_to_highest_generated[method] = len(sorted_val_sizes)
        method_to_data[method] = (
            method,
            style,
            color,
            history_dedup,
            sorted_val_sizes,
            sorted_val_percs,
        )

    try:
        max_generated_number = max([len(data[3]) for data in method_to_data.values()])
    except ValueError:
        print("No data to plot for ", max_num_atoms)
        return

    for group, methods in group_to_methods.items():
        fig, axe = group_to_fig_axe[group]
        for method in methods:
            if method not in method_to_data:
                continue
            (
                method_2,
                style,
                color,
                history_dedup,
                sorted_val_sizes,
                sorted_val_percs,
            ) = method_to_data[method]
            assert method == method_2, (method, method_2)
            label = method
            axe.plot(
                sorted_val_sizes[:max_generated_number],
                sorted_val_percs[:max_generated_number],
                style,
                label=label,
                alpha=1.0,
                color=color,
            )

    for group, (fig, ax) in group_to_fig_axe.items():
        ax.legend(loc="lower right")
        fig.savefig(f"molecule_history_{max_num_atoms}_{group}.jpg")


def get_cumulative_perc_deduplicated(gen_smiles, dataset_smiles):
    dataset_smiles_set = set(dataset_smiles)

    percs = []
    sizes = []
    subset = set()

    for molecule in tqdm(gen_smiles):
        subset.add(molecule)
        percs.append(len(subset & dataset_smiles_set) / len(dataset_smiles))
        sizes.append(len(subset))

    return sizes, percs


def get_molecule_history(filepath: str) -> list:
    history = []
    with open(filepath) as file:
        for idx, line in tqdm(enumerate(file), desc=f"Reading {filepath}"):
            line = line.strip()
            if not line:
                continue
            items = line.split()

            if len(items) == 3:
                smile = items[2]
            elif len(items) == 1:
                smile = items[0]
            else:
                raise Exception(f"Unknown format, {items}")

            assert len(items) in (1, 3)
            history.append(smile)

    return convert_smiles_to_canon(history)


def dedup_by(smiles: list, callable=lambda x: x) -> list:
    seen = set()
    deduped = []
    for smile in smiles:
        if callable(smile) not in seen:
            deduped.append(smile)
            seen.add(callable(smile))
    return deduped


def filepath_to_title(filepath: Path) -> str:
    filepath_str = str(filepath)

    if "DiGress" and "continuous" in filepath_str:
        return "DiGress continuous"

    elif "DiGress" and "discrete" in filepath_str:
        return "DiGress discrete"

    elif "moler" in filepath_str.lower():
        return "MoLeR"

    elif "data_efficient_grammar" in filepath_str.lower():
        return "Data Efficient Grammar"


def flat(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


gnnhs_regex = r"-gnnhs\d+"


if __name__ == "__main__":
    main()
