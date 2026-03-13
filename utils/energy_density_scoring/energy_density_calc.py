"""
Energy Density Calculation for Rank-Based Scoring
"""
import csv
from typing import List, Tuple

def read_irmsd_values(file_path: str) -> List[Tuple[str, str, float]]:
    """Read the irmsd values from file and returns a list of tuples.

    Args:
        file_path: path to the pairwise irmsd values csv file.
        
    Returns:
        irmsd_values: list of tuples with (m1, m2, irmsd).
    """
    irmsd_values = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            irmsd_values.append((row[0], row[1], float(row[2])))
    return irmsd_values

def read_total_irmsd(energy_file: str) -> dict:
    """Read total irmsd per model."""
    total_dict = {}
    with open(energy_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_dict[row["merged_id"]] = float(row["total_irmsd"])
    return total_dict

def create_neighbor_dict(irmsd_values):
    """Create symmetric neighbor dictionary."""
    neighbors = {}

    for m1, m2, value in irmsd_values:
        neighbors.setdefault(m1, []).append((m2, value))
        neighbors.setdefault(m2, []).append((m1, value))

    return neighbors


def rank_based_scoring(neighbor_dict, total_irmsd_dict):
    """
    Compute rank-based score for each model.
    """
    final_scores = {}

    for model, neighbor_list in neighbor_dict.items():
        if model not in total_irmsd_dict:
            continue

        score = total_irmsd_dict[model]

        for neighbor_id, pair_irmsd in neighbor_list:
            if neighbor_id not in total_irmsd_dict:
                continue
            
            score += total_irmsd_dict[neighbor_id] / (pair_irmsd + 1)

        final_scores[model] = score

    ranked = sorted(final_scores.items(), key=lambda x: x[1])
    return ranked

def rank_based_main(
    irmsd_file,
    energy_file,
    output_file=None,
):
    irmsd_values = read_irmsd_values(irmsd_file)
    neighbor_dict = create_neighbor_dict(irmsd_values)
    total_irmsd_dict = read_total_irmsd(energy_file)

    ranked_models = rank_based_scoring(
        neighbor_dict,
        total_irmsd_dict
    )

    # Output
    output_lines = []
    for rank, (model, score) in enumerate(ranked_models, start=1):
        output_lines.append(f"{rank},{model},{score}")

    if output_file:
        with open(output_file, "w") as f:
            f.write("rank,model,score\n")
            f.write("\n".join(output_lines))
        print(f"Ranking written to {output_file}")
    else:
        print("\n".join(output_lines))

    return ranked_models
