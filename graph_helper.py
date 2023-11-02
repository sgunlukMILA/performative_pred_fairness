import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors




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


def all_improve_heat_map_effort(results, var_I, str_var_I, var_J, str_var_J, effort_str="None", type_sim="NA", delta=0, save=False):
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
  results = np.flip(results, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  if (False and np.min(results) >= 0 and np.max(results) <= 1):
    map = sns.heatmap(results, cmap='viridis', vmin=0, vmax=1)
  elif (False and np.min(results) >= -1 and np.max(results) <= 1):
    map = sns.heatmap(results, cmap='viridis', vmin=-1, vmax=1)
  else:
    map = sns.heatmap(results, cmap='viridis')
  map.set_facecolor('xkcd:black')
  #, cbar_kws={'label': 'P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 0) - P(Y(c + $\Delta$) = 1| Y(c) = 0, S = 1)'})

  # Set axis labels and plot title
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
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
    plt.savefig("graphs/"+type_sim+"_effort_"+str(delta).replace('.', '_'))
  plt.show()


#Feature Importance Heatmaps
    
def importance_heatmap(results, var_I, str_var_I, var_J, str_var_J, title_importance="None", type_importance="NA", save=False):
  """
    Generate and display a heatmap for the importance of a single feature.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  map = sns.heatmap(results, cmap='viridis')#, mask=mask)#, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
    
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(title_importance+ " Importance")

  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
      plt.savefig("graphs/"+type_importance+"_importance")
  plt.show()

def diff_importance_heatmap(results, var_I, str_var_I, var_J, str_var_J, title_importance="None", type_importance="NA", save=False):
  """
    Generate and display a heatmap for the difference in importance of features A and C.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  var_I = np.flip(var_I, axis = 0)
  limit = max(np.max(results), np.min(results)*-1)
  palette = (sns.color_palette("Spectral_r", as_cmap=True))
  map = sns.heatmap(results, cmap=palette, vmin=-1*limit, vmax=limit)#, mask=mask)#, vmin=0, vmax=1)
  map.set_facecolor('xkcd:black')
  
  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(title_importance+ " Importance")

  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
      plt.savefig("graphs/"+type_importance+"_importance")
  plt.show()

def categorical_heatmap(results, var_I, str_var_I, var_J,  str_var_J, type_importance="None", title_importance="NA", save=False):
  """
    Generates and displays a categorical heatmap for feature importance, i.e. when coefficient of A is greater than
    coefficient of C and vice versa.

    Args:
        results (numpy.ndarray): Feature importance data.
        var_I (numpy.ndarray): Array of variable values for the rows of the data, to be shown on the y-axis.
        str_var_I (str): String representation of the variable for the y-axis.
        var_J (numpy.ndarray): Array of variable values for columns of the data, to be shown on the x-axis.
        str_var_J (str): String representation of the variable for the x-axis.
        type_importance (str, optional): Type of experiment for the filename (default is 'None').
        title_importance (str, optional): Type of improvement for the title of the graph (default is 'NA').
        save (bool, optional): If True, save the generated plot (default is False).
  """
  results = np.flip(results, axis = 0)
  results = np.where(results < 0, -1, 1)
  var_I = np.flip(var_I, axis = 0)
  category_cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
    
  sns.heatmap(results, fmt='', cmap=sns.color_palette("Spectral_r", as_cmap=True), cbar_kws={"ticks": [-1, 0, 1]},
               vmin=-1.25, vmax=1.25)

  plt.xlabel('Variable ' + str_var_J)
  plt.ylabel('Variable ' + str_var_I)
  plt.title(type_importance)
  step_size_I = max(len(var_I) // 4, 1)
  step_size_J = max(len(var_J) // 4, 1)
  xtick_labels = [f'{val:.1f}' for val in var_J[::step_size_J]]
  ytick_labels = [f'{val:.1f}' for val in var_I[::step_size_I]]
  plt.xticks(np.arange(0, len(var_J), step_size_J) + 0.5, xtick_labels)
  plt.yticks(np.arange(0, len(var_I), step_size_I) + 0.5, ytick_labels)
  if save:
    plt.savefig("graphs/"+type_importance+"_pos_importance")
  plt.show()





