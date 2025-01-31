"""
Name: plot_energies_normalized.py
Function: This script reads TSV files from a directory, processes the data, normalizes the energy values, and plots the normalized energy values.
Date: 12-12-2024
Author: Nils Smit
"""

"""
Example usage:
python plot_energies_normalized.py /path/to/directory/with/energy/files
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

def read_tsv_files(directory_path):
    """Reads all TSV files in the given directory.
    
    Args:
        directory_path (str): Path to the directory containing TSV files.
        
    Returns:
        list: List of TSV files in the directory.
    """
    tsv_files = [f for f in os.listdir(directory_path) if f.endswith('.tsv')]
    return tsv_files

def process_tsv_data(tsv_files, directory_path):
    """Processes the data for each TSV file.
    
    Args:
        tsv_files (list): List of TSV files.
        directory_path (str): Path to the directory containing TSV files.
    
    Returns:
        pd.DataFrame: Processed data for all TSV files.
    """
    processed_data = []
    for file in tsv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path, sep='\t')

        # Remove the 'Name' column that contains the name of the model file
        df_numeric = df.drop(columns=['Name'])

        # Calculate the vdW energy as the difference between repulsive and attractive vdW energy
        df_numeric['vdW energy'] = df_numeric['repulsive vdW energy (unweighted)'] - df_numeric['attractive vdW energy (unweighted)']
        
        # Rename the columns for clarity
        df_numeric['coulombic electrostatic energy'] = df_numeric['coulombic electrostatic energy (unweighted)']
        df_numeric['generalized Born approximation electrostatics energy'] = df_numeric['generalized Born approximation electrostatics energy (unweighted)']

        # Calculate the average of the top 5 rows for each column
        top_5_rows = df_numeric.head(5)
        avg_data = top_5_rows.mean(axis=0)
        processed_data.append([file] + avg_data.tolist())

    columns = ['Name'] + df_numeric.columns.tolist()
    processed_df = pd.DataFrame(processed_data, columns=columns)
    return processed_df

def normalize_data(processed_df):
    columns_to_normalize = [
        "total weighted energy", "vdW energy", 
        "pairwise potential energy (unweighted)"
    ]
    
    normalized_df = processed_df.copy()

    # Step 1: Normalize all other energies first
    for col in columns_to_normalize:
        min_val = processed_df[col].min()
        max_val = processed_df[col].max()
        if max_val != min_val:
            normalized_df[col] = 1 - ((processed_df[col] - min_val) / (max_val - min_val))
        else:
            normalized_df[col] = 1

    # Normalize vdW energy
    min_vdw = normalized_df['vdW energy'].min()
    max_vdw = normalized_df['vdW energy'].max()
    if max_vdw != min_vdw:
        normalized_df['vdW energy'] = 1 - (
            (normalized_df['vdW energy'] - min_vdw) / (max_vdw - min_vdw)
        )
    else:
        normalized_df['vdW energy'] = 1
    
    # Normalize Coulombic energy
    min_coulombic = normalized_df['coulombic electrostatic energy'].min()
    max_coulombic = normalized_df['coulombic electrostatic energy'].max()
    if max_coulombic != min_coulombic:
        normalized_df['coulombic electrostatic energy'] = (
            (max_coulombic - normalized_df['coulombic electrostatic energy']) 
            / (max_coulombic - min_coulombic)
        )
    else:
        normalized_df['coulombic electrostatic energy'] = 1
    
    # Normalize Generalized Born energy
    min_gb = normalized_df['generalized Born approximation electrostatics energy'].min()
    max_gb = normalized_df['generalized Born approximation electrostatics energy'].max()
    if max_gb != min_gb:
        normalized_df['generalized Born approximation electrostatics energy'] = (
            (max_gb - normalized_df['generalized Born approximation electrostatics energy']) 
            / (max_gb - min_gb)
        )
    else:
        normalized_df['generalized Born approximation electrostatics energy'] = 1

    # Step 2: Weight the energies (using unnormalized values)
    coulombic_weight = 600
    gb_weight = 60

    # Apply weights to unnormalized energies before summing
    normalized_df['Weighted Electrostatics'] = (
        processed_df['coulombic electrostatic energy (unweighted)'] * coulombic_weight +
        processed_df['generalized Born approximation electrostatics energy (unweighted)'] * gb_weight
    )

    # Step 3: Normalize the weighted electrostatics to ensure it's between 0 and 1
    min_weighted = normalized_df['Weighted Electrostatics'].min()
    max_weighted = normalized_df['Weighted Electrostatics'].max()
    if max_weighted != min_weighted:
        normalized_df['Weighted Electrostatics'] = 1 - (
            (normalized_df['Weighted Electrostatics'] - min_weighted) / (max_weighted - min_weighted)
        )
    else:
        normalized_df['Weighted Electrostatics'] = 1
    
    # Step 4: Calculate the total energy, including the weighted electrostatics
    normalized_df['Total Energy'] = normalized_df[['total weighted energy', 'vdW energy',
                                                   'coulombic electrostatic energy', 
                                                   'generalized Born approximation electrostatics energy', 
                                                   'pairwise potential energy (unweighted)', 'Weighted Electrostatics']].sum(axis=1)

    # Step 5: Sort by Total Energy
    normalized_df_sorted = normalized_df.sort_values(by='Total Energy', ascending=False)
    
    return normalized_df_sorted

def plot_stacked_bar(normalized_values, labels):
    """Creates and shows a stacked bar plot for the normalized energy values.
    
    Args:
        normalized_values (pd.DataFrame): Normalized energy values.
        labels (list): List of labels for the x-axis.
    """
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized_values.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    # Set the plot titles and labels
    ax.set_title('Normalized energies of SwiftTCRs top results', fontsize=18)
    ax.set_xlabel('Case', fontsize=14)
    ax.set_ylabel('Normalized Energy Values', fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
    
    # Add a legend
    ax.legend(title="Energy Type", loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_individual_energy_category(normalized_df_sorted, energy_categories):
    """Plots the individual energy categories.
    
    Args:
        normalized_df_sorted (pd.DataFrame): Normalized and sorted data.
        energy_categories (list): List of energy categories to plot.
    """
    # Create a color map for the energy categories
    colormap = plt.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, len(energy_categories)))
    energy_colors = dict(zip(energy_categories, colors))

    # Plot the individual energy categories
    for category in energy_categories:
        # Sort the data based on the selected category
        sorted_data = normalized_df_sorted[['Name', category]].sort_values(by=category, ascending=False)
        
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(sorted_data['Name'], sorted_data[category], color=energy_colors[category], edgecolor='black')
        
        # Set plot titles and labels
        ax.set_title(f'Normalized {category} by Case', fontsize=18)
        ax.set_xlabel('Case', fontsize=14)
        ax.set_ylabel('Normalized Energy Value (0-1)', fontsize=14)
        labels = sorted_data['Name'].str.replace('.tsv', '')
        ax.set_xticks(range(len(sorted_data)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

def plot_energy_with_selected_columns(normalized_df_sorted, energy_columns, plot_title, x_ticks=True):
    """Plots a stacked bar chart with selected energy columns and sorts the bars by total energy of those columns.
    
    Args:
        normalized_df_sorted (pd.DataFrame): Normalized and sorted data.
        energy_columns (list): List of energy columns to plot.
        plot_title (str): Title of the plot.
        x_ticks (bool): Whether to show
    """
    # Calculate the total energy based only on the selected columns
    normalized_df_sorted['Total Energy Selected'] = normalized_df_sorted[energy_columns].sum(axis=1)
    
    # Sort the dataframe based on the total energy of the selected columns
    sorted_data = normalized_df_sorted.sort_values(by='Total Energy Selected', ascending=False)

    # Extract the energy data for plotting
    energy_data = sorted_data[energy_columns]
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    energy_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    # Set plot titles and labels
    ax.set_title(plot_title, fontsize=18)
    ax.set_xlabel('Case', fontsize=14)
    ax.set_ylabel('Normalized Energy Values', fontsize=14)

    # Set the x-ticks and labels
    labels = sorted_data["Name"].str.replace('.tsv', '')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
    
    # Add a legend
    ax.legend(title="Energy Type", loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_combination_stacked_bar(normalized_df_sorted, combinations):
    """Generates all combinations of energy components and plots a stacked bar for each.
    
    Args:
        normalized_df_sorted (pd.DataFrame): Normalized and sorted data.
        combinations (list): List of combinations of energy components
    """
    for combination in combinations:
        # Calculate the total energy based on the selected combination
        normalized_df_sorted['Total Energy (' + ', '.join(combination) + ')'] = normalized_df_sorted[list(combination)].sum(axis=1)
        sorted_data = normalized_df_sorted.sort_values(by='Total Energy (' + ', '.join(combination) + ')', ascending=False)
        energy_data_combination = sorted_data[list(combination)]
        
        # Create the stacked bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        energy_data_combination.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        # Set plot titles and labels
        ax.set_title(f'Normalized Energy Components: {", ".join(combination)}', fontsize=18)
        ax.set_xlabel('Case', fontsize=14)
        ax.set_ylabel('Normalized Energy Values', fontsize=14)
        labels = sorted_data["Name"].str.replace('.tsv', '')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
        
        # Add a legend
        ax.legend(title="Energy Type", loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()


def plot_weighted_and_normalized_electrostatics(normalized_df_sorted):
    """Calculates weighted electrostatic energy, normalizes it, and plots it with total weighted energy.
    
    Args:
        normalized_df_sorted (pd.DataFrame): Normalized and sorted data.
    """
    # Use the already calculated 'Weighted Electrostatics' from normalize_data()
    # No need to recalculate it here.

    min_electrostatics = normalized_df_sorted['Weighted Electrostatics'].min()
    max_electrostatics = normalized_df_sorted['Weighted Electrostatics'].max()
    if max_electrostatics != min_electrostatics:
        normalized_df_sorted['Normalized Weighted Electrostatics'] = (
        (max_electrostatics - normalized_df_sorted['Weighted Electrostatics']) /
        (max_electrostatics - min_electrostatics)
        )
    else:
        normalized_df_sorted['Normalized Weighted Electrostatics'] = 1

    normalized_df_sorted['New Total Energy'] = (
        normalized_df_sorted['total weighted energy'] + 
        normalized_df_sorted['Normalized Weighted Electrostatics']
    )

    sorted_data = normalized_df_sorted.sort_values(by='New Total Energy', ascending=False)
    
    plot_data = sorted_data[['total weighted energy', 'Normalized Weighted Electrostatics']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_title('Normalized SwiftTCR Results', fontsize=18)
    ax.set_xlabel('Case', fontsize=14)
    ax.set_ylabel('Normalized Values', fontsize=14)
    
    labels = sorted_data["Name"].str.replace('.tsv', '')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
    
    ax.legend(title="Metrics", loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    directory_path = sys.argv[1]  # Example directory path
    
    # Read and process the data
    tsv_files = read_tsv_files(directory_path)
    processed_df = process_tsv_data(tsv_files, directory_path)

    # Normalize the data
    normalized_df_sorted = normalize_data(processed_df)

    # Plot Stacked Bar of Normalized Energies
    normalized_values = normalized_df_sorted[['total weighted energy', 'vdW energy', 
                                              'coulombic electrostatic energy', 
                                              'generalized Born approximation electrostatics energy', 
                                              'pairwise potential energy (unweighted)']]
    labels = normalized_df_sorted["Name"].str.replace('.tsv', '')
    plot_stacked_bar(normalized_values, labels)

    # Plot Individual Energy Categories
    energy_categories = [
        "total weighted energy", 
        "vdW energy", 
        "coulombic electrostatic energy", 
        "generalized Born approximation electrostatics energy",
        "pairwise potential energy (unweighted)"
    ]
    plot_individual_energy_category(normalized_df_sorted, energy_categories)

    # Plot Energy with Selected Columns
    energy_columns = ['total weighted energy', 'coulombic electrostatic energy']
    plot_energy_with_selected_columns(normalized_df_sorted, energy_columns,
                                      'Normalized Total Weighted Energy and Coulombic Electrostatic Energy')

    # Plot Energy with Generalized Born Electrostatic Energy
    energy_columns_gb = ['total weighted energy', 'generalized Born approximation electrostatics energy']
    plot_energy_with_selected_columns(normalized_df_sorted, energy_columns_gb,
                                      'Normalized Total Weighted Energy and Generalized Born Electrostatic Energy')

    # Plot All Energies Except Generalized Born
    energy_columns_all = ['total weighted energy', 'vdW energy', 'coulombic electrostatic energy', 'pairwise potential energy (unweighted)']
    plot_energy_with_selected_columns(normalized_df_sorted, energy_columns_all,
                                      'Normalized Energy Components (Excluding Generalized Born Electrostatic Energy)')
    
    # Plot the total weighted energy and weighted electrostatic energies    
    plot_weighted_and_normalized_electrostatics(normalized_df_sorted)


    # # Plot All Combinations of Energy Components
    # energy_columns_all = [
    #     'total weighted energy', 'vdW energy', 'coulombic electrostatic energy', 'pairwise potential energy (unweighted)', 
    #     'generalized Born approximation electrostatics energy'
    # ]
    # combinations = []
    # for r in range(1, len(energy_columns_all) + 1):
    #     combinations.extend(itertools.combinations(energy_columns_all, r))

    # plot_combination_stacked_bar(normalized_df_sorted, combinations)