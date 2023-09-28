import numpy as np
import matplotlib.pyplot as plt


def plot_a_c_single(coordinates, labels):
    """
    Create a scatter plot with different symbols for each data point.

    Parameters:
    - coordinates (numpy.ndarray): An Nx2 array of 2D coordinates.
    - symbols (numpy.ndarray): An N-element binary array (0s and 1s) where 0 represents a circle and 1 represents a cross.
    """
    # Extract x and y coordinates
    x, y = coordinates[:, 0], coordinates[:, 1]

    # Create a scatter plot
    fig, ax = plt.subplots()

    # Plot circles for symbols with value 0
    ax.scatter(x[labels == 0], y[labels == 0], marker='o', label='Y=0', color='b', s=15)

    # Plot crosses for symbols with value 1
    ax.scatter(x[labels == 1], y[labels == 1], marker='x', label='Y=1', color='r', s=15, alpha=.5)

    ax.set_xlabel('A (ancestor) value')
    ax.set_ylabel('C (direct cause) value')
    ax.set_title('Population with majority and minority groups')
    ax.legend()

    plt.show()


def plot_a_c(coordinates1, labels1, is_improv1,
             coordinates2, labels2, is_improv2,
             classifier=None, real_boundary_maj=None, real_boundary_min=None, max_delta=None):
    """
    Create two scatter plots on top of each other with different symbols for each data point.

    Parameters:
    - coordinates1 (numpy.ndarray): An Nx2 array of 2D coordinates for the first plot.
    - labels1 (numpy.ndarray): An N-element binary array (0s and 1s) for the first plot, where 0 represents a circle and 1 represents a cross.
    - coordinates2 (numpy.ndarray): An Nx2 array of 2D coordinates for the second plot.
    - labels2 (numpy.ndarray): An N-element binary array (0s and 1s) for the second plot, where 0 represents a circle and 1 represents a cross.
    - line_params (dict): A dictionary containing the line parameters with 'w' as a 2-D array representing the slope and 'b' as a 1-D array representing the bias.
    """
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(11, 8))

    # Plot the first set of circles and crosses
    ax1.scatter(coordinates1[:, 0][(labels1 == 0) & (is_improv1 == 0)],
                coordinates1[:, 1][(labels1 == 0) & (is_improv1 == 0)],
                marker='o', label='Y=0', color='b', s=15)
    ax1.scatter(coordinates1[:, 0][(labels1 == 1) & (is_improv1 == 0)],
                coordinates1[:, 1][(labels1 == 1) & (is_improv1 == 0)],
                marker='x', label='Y=1', color='r', s=15, alpha=0.5)
    ax1.scatter(coordinates1[:, 0][(labels1 == 0) & (is_improv1 == 1)],
                coordinates1[:, 1][(labels1 == 0) & (is_improv1 == 1)],
                marker='o', label=f'Y=0 improvable, $\delta = {max_delta}$', color='g', s=15)
    ax1.scatter(coordinates1[:, 0][(labels1 == 1) & (is_improv1 == 1)],
                coordinates1[:, 1][(labels1 == 1) & (is_improv1 == 1)],
                marker='x', label=f'Y=1 improvable, $\delta = {max_delta}$', color='g', s=15, alpha=0.5)
    ax1.set_ylabel('C (direct cause) value')
    ax1.set_title('minority')
    ax1.legend()

    # Plot the second set of circles and crosses
    ax2.scatter(coordinates2[:, 0][(labels2 == 0) & (is_improv2 == 0)],
                coordinates2[:, 1][(labels2 == 0) & (is_improv2 == 0)],
                marker='o', label='Y=0', color='b', s=15)
    ax2.scatter(coordinates2[:, 0][(labels2 == 1) & (is_improv2 == 0)],
                coordinates2[:, 1][(labels2 == 1) & (is_improv2 == 0)],
                marker='x', label='Y=1', color='r', s=15, alpha=0.5)
    ax2.scatter(coordinates2[:, 0][(labels2 == 0) & (is_improv2 == 1)],
                coordinates2[:, 1][(labels2 == 0) & (is_improv2 == 1)],
                marker='o', label=f'Y=0 improvable, $\delta = {max_delta}$', color='g', s=15)
    ax2.scatter(coordinates2[:, 0][(labels2 == 1) & (is_improv2 == 1)],
                coordinates2[:, 1][(labels2 == 1) & (is_improv2 == 1)],
                marker='x', label=f'Y=1 improvable, $\delta = {max_delta}$', color='g', s=15, alpha=0.5)
    ax2.set_xlabel('A (ancestor) value')
    ax2.set_ylabel('C (direct cause) value')
    ax2.set_title('majority')
    ax2.legend()

    # Plot the second set of circles and crosses
    ax3.scatter(coordinates1[:, 0][(labels1 == 0) & (is_improv1 == 0)],
                coordinates1[:, 1][(labels1 == 0) & (is_improv1 == 0)],
                marker='o', label='Y=0', color='b', s=15)
    ax3.scatter(coordinates1[:, 0][(labels1 == 1) & (is_improv1 == 0)],
                coordinates1[:, 1][(labels1 == 1) & (is_improv1 == 0)],
                marker='x', label='Y=1', color='r', s=15, alpha=0.5)
    ax3.scatter(coordinates1[:, 0][(labels1 == 0) & (is_improv1 == 1)],
                coordinates1[:, 1][(labels1 == 0) & (is_improv1 == 1)],
                marker='o', label=f'Y=0 improvable, $\delta = {max_delta}$', color='g', s=15)
    ax3.scatter(coordinates1[:, 0][(labels1 == 1) & (is_improv1 == 1)],
                coordinates1[:, 1][(labels1 == 1) & (is_improv1 == 1)],
                marker='x', label=f'Y=1 improvable, $\delta = {max_delta}$', color='g', s=15, alpha=0.5)
    ax3.scatter(coordinates2[:, 0][(labels2 == 0) & (is_improv2 == 0)],
                coordinates2[:, 1][(labels2 == 0) & (is_improv2 == 0)],
                marker='o', color='b', s=15)
    ax3.scatter(coordinates2[:, 0][(labels2 == 1) & (is_improv2 == 0)],
                coordinates2[:, 1][(labels2 == 1) & (is_improv2 == 0)],
                marker='x', color='r', s=15, alpha=0.5)
    ax3.scatter(coordinates2[:, 0][(labels2 == 0) & (is_improv2 == 1)],
                coordinates2[:, 1][(labels2 == 0) & (is_improv2 == 1)],
                marker='o', color='g', s=15)
    ax3.scatter(coordinates2[:, 0][(labels2 == 1) & (is_improv2 == 1)],
                coordinates2[:, 1][(labels2 == 1) & (is_improv2 == 1)],
                marker='x', color='g', s=15, alpha=0.5)
    ax3.set_xlabel('A (ancestor) value')
    ax3.set_ylabel('C (direct cause) value')
    ax3.set_title('whole population')
    ax3.legend()

    # Add a global title above all subplots
    plt.suptitle("Population and classifier $f(A,C)$")
    # Add a dashed line to both plots
    if classifier:
        coef = classifier.coef_[0]  # Get the coefficients of the decision boundary
        intercept = classifier.intercept_
        x_boundary = np.linspace(min(np.min(coordinates1[:, 0]), np.min(coordinates2[:, 0])),
                                 max(np.max(coordinates1[:, 0]), np.max(coordinates2[:, 0])), 100)
        y_boundary = (-coef[0] * x_boundary - intercept) / coef[1]
        ax1.plot(x_boundary, y_boundary, 'k--', label='$\hat{f}(C,A)=0.5$')
        ax2.plot(x_boundary, y_boundary, 'k--', label='$\hat{f}(C,A)=0.5$')
        ax3.plot(x_boundary, y_boundary, 'k--', label='$\hat{f}(C,A)=0.5$')

        # Add a legend for the dashed line in both subplots
        ax1.legend(loc='upper right', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed
        ax2.legend(loc='lower left', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed
        ax3.legend(loc='lower left', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed

    if real_boundary_min:
        x_boundary = np.linspace(min(np.min(coordinates1[:, 0]), np.min(coordinates2[:, 0])),
                                 max(np.max(coordinates1[:, 0]), np.max(coordinates2[:, 0])), 100)
        y_boundary = real_boundary_min * np.ones_like(x_boundary)
        ax1.plot(x_boundary, y_boundary, 'gray', linestyle=':', label='$\mathbb{E}[Y|C,S=-1]=0.5$')
        ax3.plot(x_boundary, y_boundary, 'gray', linestyle=':', label='$\mathbb{E}[Y|C,S=-1]=0.5$')

        # Add a legend for the dashed line in both subplots
        ax1.legend(loc='upper right', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed
        ax3.legend(loc='lower left', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed

    if real_boundary_maj:
        x_boundary = np.linspace(min(np.min(coordinates1[:, 0]), np.min(coordinates2[:, 0])),
                                 max(np.max(coordinates1[:, 0]), np.max(coordinates2[:, 0])), 100)
        y_boundary = real_boundary_maj * np.ones_like(x_boundary)
        ax2.plot(x_boundary, y_boundary, 'gray', linestyle=':', label='$\mathbb{E}[Y|C,S=+1]=0.5$')
        ax3.plot(x_boundary, y_boundary, 'gray', linestyle=':', label='$\mathbb{E}[Y|C,S=+1]=0.5$')

        # Add a legend for the dashed line in both subplots
        ax2.legend(loc='upper left', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed
        ax3.legend(loc='upper left', frameon=True, fontsize='x-small')  # Adjust the 'loc' parameter as needed

    # Adjust spacing between subplots
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from SCM import ScalarLinearDecisionModel
    from improvements_fcs import find_pred_improve_lin_cost, find_best_real_improve_lin_cost

    default_params = {
        'n_samples': 1000,
        's_a_const': 1,
        'a_var': 1,
        's_c_const': 0,
        'a_c_const': 1,
        'c_var': 1,
        's_y_const': 1,
        'c_y_const': 1,
        'y_var': 0.1
    }

    # generate data
    data_train = ScalarLinearDecisionModel(default_params)
    data_train.generate_basic_data()
    data_train_AC = np.stack((data_train.A, data_train.C), axis=1)

    # train model
    model = LogisticRegression().fit(data_train_AC, data_train.Y)
    pred_fn = {'w': model.coef_.flatten(), 'b': model.intercept_[0]}

    # get improvables info
    cost_fn = {'w': np.array([1, 1]), 'b': 0}
    data_summary = {
        "c_noise": data_train.c_noise,
        "S": data_train.S,
        "AC": np.stack((data_train.A, data_train.C), axis=1),
        "Y_logit": data_train.Y_logit,
        "Y": data_train.Y
    }
    # pred_best_delta, pred_best_improvement, pred_which_improve = find_pred_improve_lin_cost(data_train, cost_fn,
    #                                                                                         pred_fn, data_summary)
    real_best_delta, real_best_improvement, real_which_improve = find_best_real_improve_lin_cost(data_train, cost_fn,
                                                                                                 data_summary)
    max_delta = 1
    data_train.store_is_improvable(real_best_delta, max_delta=max_delta)

    # plot data
    pd_data = data_train.ts_to_df()
    mino_pd_data = pd_data.loc[pd_data['S'] == 0]
    mino_ac = np.stack((mino_pd_data.A, mino_pd_data.C), axis=1)
    majo_pd_data = pd_data.loc[pd_data['S'] == 1]
    majo_ac = np.stack((majo_pd_data.A, majo_pd_data.C), axis=1)

    plot_a_c(mino_ac, mino_pd_data.Y, mino_pd_data.is_improvable,
             majo_ac, majo_pd_data.Y, majo_pd_data.is_improvable,
             classifier=model,
             real_boundary_maj=data_train.get_real_boundary(s=1),
             real_boundary_min=data_train.get_real_boundary(s=-1),
             max_delta=max_delta)

