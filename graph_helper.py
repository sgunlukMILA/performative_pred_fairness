import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import os
from datetime import datetime
import json


labels = {
    's_u_const': r"$\beta$ (S$\rightarrow$U)",
    's_a_const': r"$\alpha_S$ (S$\rightarrow$A)", 
    's_c_const': r"$\omega_S$ (S$\rightarrow$C)",  
    'a_c_const': r"$\omega_A$ (A$\rightarrow$C)", 
    's_y_const': r"$m_S$ (S$\rightarrow$Y)", 
    'c_y_const': r"$m_C$ (C$\rightarrow$Y)",
    'u_a_const': r"$\alpha_U$ (U$\rightarrow$A)",
    'u_y_const': r"$m_U$ (U$\rightarrow$Y)",
    'w_A' : r"Learned coeff of feature $A$",
    'w_C' : r"Learned coeff of feature $C$",
    'b' : r"Learned intercept $b$",
}

def initialize_graph_folder(params):
  
  # Get the current date and time as a string
  current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

  # Create the folder name with "graphs_" prefix
  folder_name = f"graphs_{current_datetime}"

  # Specify the path where you want to create the folder
  base_directory = ""  # Replace with your desired directory path

  # Combine the base directory and folder name to create the full path
  folder_path = os.path.join(base_directory, folder_name)

  # Check if the folder already exists before creating it
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_name}' created at '{folder_path}'")
  else:
    print(f"Folder '{folder_name}' already exists at '{folder_path}'")
    
  params_file_path = os.path.join(folder_path, "params.txt")

  # Write the dictionary to a JSON file
  with open(params_file_path, "w") as file:
    json.dump(params, file)

  print(f"Dictionary saved to '{params_file_path}'")
    
  return folder_name


#grapher helper functions
def line_plots(ei_data, consts, deltas, arrow_str = ''):
  """
    Generate and display line plots for Equal Improvability (EI) data.

    Args:
        ei_data (numpy.ndarray): EI data for varying cofficients and deltas.
        consts (list): List of values of the varied coefficient with which EI data was collected.
        deltas (list): List of delta values with which EI data was collected.
        arrow_str (str, optional): String for the legent to indicate the varied coefficient (default is '').
  """
  names = sorted(
            mcolors.TABLEAU_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
  names = [elem.removeprefix('tab:') for elem in names]
  for i, const in enumerate(consts):
    plt.plot(deltas, ei_data[i,:,0], color=names[i], linestyle='dashed',
             label = "protected for const = "+ str(round(const, 4)))
    plt.plot(deltas, ei_data[i,:,1], color=names[i],
             label = "not protected for const = "+ str(round(const, 4)))
  plt.xlabel('effort ($\Delta$)')
  plt.ylabel('probability of improvability')
  plt.title('Plot of P(Y(c+$\Delta$) = 1| Y(c) = 0, S = s) vs $\Delta$ effort')
  plt.legend()
  plt.show()

  for i, const in enumerate(consts):
    plt.plot(deltas, ei_data[i,:,1] - ei_data[i,:,0], color=names[i],
             label = arrow_str + " coeff = "+ str(round(const, 4)))
  plt.xlabel('effort ($\Delta$)')
  plt.ylabel('difference in probability of improving')
  plt.title('Plot of P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 1) - P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 0) vs $\Delta$ effort')
  plt.legend()
  plt.show()


def heat_map_effort(sim_params, results, type_graph, delta=0, save=False, folder = ""):
  """
    Generate and display a heatmap for improvement probability data across different variables.

    Args:
        results (numpy.ndarray): EI data while varying two coefficients.
        mask (numpy.ndarray): Mask to indicate missing data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        effort_str (str, optional): Description of effort regime based on delta (default is 'None').
        type_sim (str, optional): Type of experiment for the filename (default is 'NA').
        delta (float, optional): Effort change delta (default is 0).
        save (bool, optional): If True, save the generated plot (default is False).
  """

  # Initialize data
  results = np.flip(results, axis = 0)
  var_I = np.flip(sim_params["vars_0"], axis = 0)
  var_J = sim_params["vars_1"]
    
  # Select appropriate min and max for heatmap
  if (False and np.min(results) >= 0 and np.max(results) <= 1):
    map = sns.heatmap(results, cmap='viridis', vmin=0, vmax=1)
  elif (False and np.min(results) >= -1 and np.max(results) <= 1):
    map = sns.heatmap(results, cmap='viridis', vmin=-1, vmax=1)
  else:
    map = sns.heatmap(results, cmap='viridis')
  map.set_facecolor('xkcd:black')

  # Set axis labels and plot title
  plt.xlabel('Variable ' + labels[sim_params["str_vars"][1]])
  plt.ylabel('Variable ' + labels[sim_params["str_vars"][0]])
  plt.title("$\Delta$ = " + str(delta) + " Effort Regime")
  plt.margins(x=0)

  # Customize the tick labels on the x and y axes
  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  # Show the heatmap
  if save:
    plt.savefig(sim_params["graph_folder"] + "/"+ type_graph + "_effort_" + str(delta).replace('.', '_'))
  plt.show()


