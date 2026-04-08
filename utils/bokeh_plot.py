"""
Name: bokeh_plot.py
Date: 08-Apr-2026
Author: Wieke Krösschell

Function:
Showing the calculated angles (calculated with calc_incident_crossing_angle.py) in an interactive scatterplot

INPUT:
- <path/to/anglesfile>.csv  :   CSV file containing the model name, crossing and incident angles
- <list of cases to highlight>  :   All cases to highlight with pdb extension and divided by comma's
- <title of plots>          :   Desired title of the plots

OUTPUT:
- angles.html   :   Interactive scatterplot with html extension
- angles.pdf    :   Non-interactive scatterplot as PDF

Example usage:
python bokeh_plot.py angles.csv "2ian.pdb,2iam.pdb,2pxy.pdb,6cql.pdb,6cqq.pdb,6cqr.pdb" "Angles of STCRDab dataset"

"""

# Importing the modules 
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, gridplot
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import export_png
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

def bokeh_plot(input_file, highlight_list, title_plots, legend, highlight_legend):
    #  Save as a .html file
    output_file("angles.html")

    # Use highlight_list as hardcoded
    # highlight_list = ["2ian.pdb", "2iam.pdb", "2pxy.pdb", "6cql.pdb", "6cqq.pdb", "6cqr.pdb"]

    # Init of the plot
    graph = figure(title = title_plots,
                x_range = (0, 100),
                y_range = (0, 60),
                x_axis_label = 'Crossing angle',
                y_axis_label = 'Incident angle') 

    # Extract angles from original csv file
    df_new = pd.read_csv(csv_path)

    # Creating highlight dataframe
    df_highlight =pd.DataFrame()
    df = df_new[df_new['modelname'].isin(highlight_list)]
    df_highlight = pd.concat([df_highlight, df], ignore_index=True)

    # Creating input for bokeh plot
    src_new = ColumnDataSource(df_new)
    src_highlight = ColumnDataSource(df_highlight)

    # Plot the angles
    graph.scatter(source = src_new, x='crossing_angle', y='incident_angle', size=9, color='blue', legend_label=f"{legend}")
    graph.scatter(source = df_highlight, x='crossing_angle', y='incident_angle', size=9, color='yellow', legend_label=f"{highlight_legend}")

    # Show legend
    graph.legend.location = "top_center"
    graph.legend.click_policy = "hide"

    hover = HoverTool(tooltips=[('Name', '@modelname'),
                                ('Crossing angle', '@crossing_angle'),
                                ('Incident angle', '@incident_angle')])
    graph.add_tools(hover)

    # Create the plot with reverse angles
    graph_reverse = figure(
        x_range=(-140, -40),
        y_range=graph.y_range,
        toolbar_location=None,
        x_axis_label = 'Crossing angle',
        y_axis_label = 'Incident angle'
    )

    # Shows the reversly docked models
    graph_reverse.scatter(source = src_new, x='crossing_angle', y='incident_angle', size=9, color='blue', legend_label=f"{legend}")
    graph_reverse.scatter(source = df_highlight, x='crossing_angle', y='incident_angle', size=9, color='yellow', legend_label=f"{highlight_legend}")
    graph_reverse.add_tools(hover)
    show(gridplot([[graph, graph_reverse]]))
    print('angles.html created')

def pdf_file(input_file, highlight_list, title_plots, legend, highlight_legend):
    # Load data
    df_new = pd.read_csv(input_file)

    # Highlight dataframe
    df_highlight = df_new[df_new['modelname'].isin(highlight_list)]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Creating plot
    ax1.scatter(df_new['crossing_angle'], df_new['incident_angle'],
                s=30, color='blue', label=f'{legend}')

    ax1.scatter(df_highlight['crossing_angle'], df_highlight['incident_angle'],
                s=30, color='yellow', label=f'{highlight_legend}')

    ax1.set_title(title_plots)
    ax1.set_xlabel("Crossing angle")
    ax1.set_ylabel("Incident angle")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 60)
    ax1.grid(True, linestyle='--', color='grey', alpha=0.5)
    ax1.legend()

    # Reverse angles plot
    ax2.scatter(df_new['crossing_angle'], df_new['incident_angle'],
                s=30, color='blue', label=f'{legend}')

    ax2.scatter(df_highlight['crossing_angle'], df_highlight['incident_angle'],
                s=30, color='yellow', label=f'{highlight_legend}')

    ax2.set_title(f"{title_plots}, reversely docked")
    ax2.set_xlim(-140, -40)
    ax2.set_ylim(0, 60)
    ax2.set_xlabel("Crossing angle")
    ax2.set_ylabel("Incident angle")
    ax2.grid(True, linestyle='--', color='grey', alpha=0.5)
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Save as PDF
    plt.savefig('angles.pdf', format='pdf')
    print('angles.pdf created')

if __name__ =="__main__":
    # Assigns variables to input
    csv_path = str(argv[1]) # Path to CSV file 
    highlight_list = argv[2].split(sep=',') # List with highlighted cases
    title_plots = str(argv[3])

    # Sets legend names of plots, easy to change
    legend = "PDB structures from STCRDab dataset"
    highlight_legend = "Benchmark dataset chosen structures"

    bokeh_plot(csv_path, highlight_list, title_plots, legend, highlight_legend)
    pdf_file(csv_path, highlight_list, title_plots, legend, highlight_legend)
