import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pulp

def load_and_clean_data(file_path, input_cols, output_cols):
    """
    Load data from a CSV file and clean it for DEA analysis
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    input_cols : list
        Names of input columns
    output_cols : list
        Names of output columns
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe ready for DEA analysis
    """
    # Extract year from filename (assuming format like 2001.csv)
    year = os.path.basename(file_path).split('.')[0]
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Drop rows with missing values in input or output columns
    df_clean = df.dropna(subset=input_cols + output_cols)
    
    # Add year column
    df_clean['Year'] = year
    
    return df_clean

def run_output_oriented_dea(inputs, outputs):
    """
    Perform output-oriented constant returns to scale DEA
    
    Parameters:
    -----------
    inputs : numpy.ndarray
        Input data matrix (n_dmus × n_inputs)
    outputs : numpy.ndarray
        Output data matrix (n_dmus × n_outputs)
        
    Returns:
    --------
    list
        Efficiency scores for each DMU (firm)
    """
    n_dmus = inputs.shape[0]  # Number of firms
    efficiency_scores = []
    
    # For each DMU (firm), solve a linear programming problem
    for i in range(n_dmus):
        # Create the optimization model
        model = pulp.LpProblem(f"DEA_DMU_{i}", pulp.LpMaximize)
        
        # Create variables
        phi = pulp.LpVariable(f"phi_{i}", lowBound=1)  # Output expansion factor
        lambdas = [pulp.LpVariable(f"lambda_{i}_{j}", lowBound=0) for j in range(n_dmus)]
        
        # Objective function: maximize phi (efficiency)
        model += phi
        
        # Input constraints: weighted sum of inputs <= inputs of DMU i
        for input_idx in range(inputs.shape[1]):
            model += pulp.lpSum(inputs[j, input_idx] * lambdas[j] for j in range(n_dmus)) <= inputs[i, input_idx]
        
        # Output constraints: weighted sum of outputs >= phi * outputs of DMU i
        for output_idx in range(outputs.shape[1]):
            model += pulp.lpSum(outputs[j, output_idx] * lambdas[j] for j in range(n_dmus)) >= phi * outputs[i, output_idx]
        
        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses output
        
        # Calculate efficiency (for output-oriented models, efficiency = 1/phi)
        if pulp.LpStatus[model.status] == 'Optimal':
            eff_score = 1 / phi.value()
            efficiency_scores.append(eff_score)
        else:
            print(f"Warning: No optimal solution found for DMU {i}")
            efficiency_scores.append(None)
    
    return efficiency_scores

def perform_dea(df, input_cols, output_cols):
    """
    Perform output-oriented DEA analysis on a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with data for a specific year
    input_cols : list
        List of column names representing inputs
    output_cols : list
        List of column names representing outputs
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with firm names and efficiency scores
    """
    # Extract firm names and year
    firm_names = df['Company Name'].values
    year = df['Year'].iloc[0]
    
    # Extract inputs and outputs as numpy arrays
    inputs = df[input_cols].values
    outputs = df[output_cols].values
    
    print(f"Running DEA for Year {year}: {len(firm_names)} firms, {len(input_cols)} inputs, {len(output_cols)} outputs")
    
    # Run DEA analysis
    efficiency_scores = run_output_oriented_dea(inputs, outputs)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Company Name': firm_names,
        'Year': year,
        'EfficiencyScore': efficiency_scores
    })
    
    return results

def process_all_years(data_folder, input_cols, output_cols):
    """
    Process all years' data and combine into one panel dataset
    
    Parameters:
    -----------
    data_folder : str
        Folder containing yearly CSV files
    input_cols : list
        List of column names representing inputs
    output_cols : list
        List of column names representing outputs
        
    Returns:
    --------
    pandas.DataFrame
        Combined panel dataset with efficiency scores for all years
    """
    # Get all CSV files in the folder
    csv_files = glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return None
        
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_results = []
    
    for file_path in sorted(csv_files):
        try:
            # Load and clean data
            df = load_and_clean_data(file_path, input_cols, output_cols)
            
            # Skip if dataframe is empty after cleaning
            if len(df) == 0:
                print(f"Warning: No valid data in {file_path} after cleaning")
                continue
                
            # Perform DEA
            year_results = perform_dea(df, input_cols, output_cols)
            
            # Add to list of results
            all_results.append(year_results)
            
            print(f"Successfully processed {file_path}: {len(year_results)} firms analyzed")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Combine all results
    if all_results:
        panel_data = pd.concat(all_results, ignore_index=True)
        print(f"Combined panel data: {panel_data.shape[0]} rows, {len(panel_data['Company Name'].unique())} unique firms")
        return panel_data
    else:
        print("No results to combine. Check data files and error messages.")
        return None

def calculate_firm_averages(panel_data):
    """
    Calculate average efficiency scores for each firm across all years
    
    Parameters:
    -----------
    panel_data : pandas.DataFrame
        Panel dataset with efficiency scores
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with average efficiency scores by firm
    """
    if panel_data is None:
        return None
        
    # Group by firm and calculate statistics
    firm_averages = panel_data.groupby('Company Name')['EfficiencyScore'].agg(
        ['mean', 'std', 'min', 'max', 'count']).reset_index()
    
    # Rename columns
    firm_averages.columns = ['Company Name', 'AvgEfficiency', 'StdEfficiency', 
                            'MinEfficiency', 'MaxEfficiency', 'YearsPresent']
    
    return firm_averages

def plot_efficiency_trends(panel_data, top_n=5):
    """
    Plot efficiency trends for top N firms (by data availability)
    
    Parameters:
    -----------
    panel_data : pandas.DataFrame
        Panel dataset with efficiency scores
    top_n : int
        Number of firms to plot
    """
    if panel_data is None:
        return
        
    # Find firms with most years of data
    firm_counts = panel_data['Company Name'].value_counts().nlargest(top_n)
    firms_to_plot = firm_counts.index.tolist()
    
    # Filter data for these firms
    plot_data = panel_data[panel_data['Company Name'].isin(firms_to_plot)]
    
    # Convert Year to numeric
    plot_data['Year'] = pd.to_numeric(plot_data['Year'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    for firm in firms_to_plot:
        firm_data = plot_data[plot_data['Company Name'] == firm]
        firm_data = firm_data.sort_values('Year')  # Sort by year for connected lines
        plt.plot(firm_data['Year'], firm_data['EfficiencyScore'], marker='o', label=firm)
    
    plt.title(f'Efficiency Trends for Top {top_n} Firms (by data availability)')
    plt.xlabel('Year')
    plt.ylabel('Efficiency Score (1.0 = efficient)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('efficiency_trends.png')
    print(f"Plot saved as 'efficiency_trends.png'")

def main():
    """Main function to run the entire analysis"""
    # Configuration - MODIFY THESE AS NEEDED
    data_folder = "."  # Folder containing CSV files
    input_cols = ['Net fixed assets', 'Raw materials consumed', 'Power & fuel', 'Salaries, wages, bonus, ex gratia pf & gratuities paid']  # Input column names
    output_cols = ['Sales of goods']  # Output column names
    output_file = "efficiency_panel_data.csv"  # Output file name
    
    print("=" * 60)
    print("DEA PANEL ANALYSIS")
    print("=" * 60)
    print(f"Data folder: {data_folder}")
    print(f"Input columns: {input_cols}")
    print(f"Output columns: {output_cols}")
    print("=" * 60)
    
    # Process all years
    print("Starting DEA analysis for all years...")
    panel_data = process_all_years(data_folder, input_cols, output_cols)
    
    if panel_data is not None:
        # Save panel data
        panel_data.to_csv(output_file, index=False)
        print(f"Panel data saved to {output_file}")
        
        # Calculate firm averages
        firm_averages = calculate_firm_averages(panel_data)
        firm_averages.to_csv("firm_average_efficiency.csv", index=False)
        print("Average efficiency scores by firm saved to firm_average_efficiency.csv")
        
        # Plot efficiency trends
        print("Generating efficiency trend plot...")
        plot_efficiency_trends(panel_data)
        
        print("Analysis completed successfully!")
    else:
        print("Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()