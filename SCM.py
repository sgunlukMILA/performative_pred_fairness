import numpy as np
import pandas as pd


class ScalarLinearDecisionModel():
    """
    A class for generating simulated data based on our scalar linear decision model.

    Attributes:
        n_samples (int): Number of samples to generate.
        s_a_const (float): Coefficient for S in generating A (alpha).
        a_var (float): Variance for A noise (epsilon_A^2).
        a_noise (numpy.ndarray): Noise added to A based on a_var (u_A).
        s_c_const (float): Coefficient for S in generating C (omega_S).
        a_c_const (float): Coefficient for A in generating C (omega_A).
        c_var (float): Variance for C noise (epsilon_C^2).
        c_noise (numpy.ndarray): Noise added to C based on c_var (u_C).
        s_y_const (float): Coefficient for S in generating Y (m_S).
        c_y_const (float): Coefficient for C in generating Y (m_C).
        y_var (float): Variance for Y noise (epsilon_Y^2).
        y_noise (numpy.ndarray): Noise added to Y based on y_var (u_Y).
        S (numpy.ndarray): Binary variable indicating the treatment {0, 1}.
        S_sym (numpy.ndarray): Binary variable converted to {-1, 1}.
        A (numpy.ndarray): Simulated values of variable A.
        C (numpy.ndarray): Simulated values of variable C.
        Y (numpy.ndarray): Simulated values of binary outcome Y.

    Methods:
        __init__(params): Initialize the ScalarLinearDecisionModel using params, a dictionary containing model parameters.
        generate_basic_data(): Generates simulated data based on the model.
        generate_do_A_data(new_A): Generates data for improved set of ancestral featues.
        generate_improve_A_data(diff_A): Generates data by improving A by diff_A.
        generate_do_C_data(new_C): Generates data for improved set of causal featues.
        generate_improve_C_data(diff_C): Generates data by improving C by diff_C.
        ts_to_df(describe=False): Converts simulated data to a DataFrame with describe being a printer attribute.
    """

    def __init__(self, params):
        # n_samples, s_a_const = 1, a_var = 1, s_c_const = 0,  a_c_const = 1, c_var = 1, s_y_const = 1, c_y_const = 1, y_var = 0.1):
        self.n_samples = params['n_samples']
        self.s_a_const = params['s_a_const']
        self.a_var = params['a_var']
        self.a_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.a_var * np.ones(self.n_samples))
        self.s_c_const = params['s_c_const']
        self.a_c_const = params['a_c_const']
        self.c_var = params['c_var']
        self.c_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.c_var * np.ones(self.n_samples))
        self.s_y_const = params['s_y_const']
        self.c_y_const = params['c_y_const']
        self.y_var = params['y_var']
        self.y_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.y_var * np.ones(self.n_samples))
        self.S = np.random.binomial(n=1, p=0.75, size=self.n_samples)
        # make labels s = {-1, 1} by doing (2S-1)
        self.S_sym = 2 * self.S - 1
        self.A = np.empty((self.n_samples, 1))
        self.C = np.empty((self.n_samples, 1))
        self.Y = np.empty((self.n_samples, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_basic_data(self):
        self.A = self.S_sym * self.s_a_const + self.a_noise
        self.C = self.A * self.a_c_const + self.S_sym * self.s_c_const + self.c_noise
        y_logit = self.sigmoid(self.C * self.c_y_const + self.S_sym * self.s_y_const + self.y_noise)
        self.Y = np.random.binomial(n=1, p=y_logit)

    def generate_do_A_data(self, new_A):
        temp_C = new_A * self.a_c_const + self.S_sym * self.s_c_const + self.c_noise
        y_logit = self.sigmoid(self.C * self.c_y_const + self.S_sym * self.s_y_const + self.y_noise)
        temp_Y = np.random.binomial(n=1, p=y_logit)
        return temp_C, temp_Y

    def generate_improve_A_data(self, diff_A):
        temp_C = (self.A + diff_A) * self.a_c_const + self.S_sym * self.s_c_const + self.c_noise
        y_logit = self.sigmoid(self.C * self.c_y_const + self.S_sym * self.s_y_const + self.y_noise)
        temp_Y = np.random.binomial(n=1, p=y_logit)
        return temp_C, temp_Y

    def generate_do_C_data(self, new_C):
        y_logit = self.sigmoid(new_C * self.c_y_const + self.S_sym * self.s_y_const + self.y_noise)
        temp_Y = np.random.binomial(n=1, p=y_logit)
        return temp_Y

    def generate_improve_C_data(self, diff_C):
        y_logit = self.sigmoid((self.C + diff_C) * self.c_y_const + self.S_sym * self.s_y_const + self.y_noise)
        temp_Y = np.random.binomial(n=1, p=y_logit)
        return temp_Y

    def ts_to_df(self, describe=False):
        sim_data = pd.DataFrame()
        sim_data["S"] = self.S
        sim_data["A"] = self.A
        sim_data["C"] = self.C
        sim_data["Y"] = self.Y
        if describe:
            print(sim_data.describe())
            print(sim_data.loc[sim_data['S'] == 0].describe())
            print(sim_data.loc[sim_data['S'] == 1].describe())
        return sim_data


if __name__ == "__main__":
    # Data generation test call
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

    sample_dm = ScalarLinearDecisionModel(default_params)
    sample_dm.generate_basic_data()
    sample_data = sample_dm.ts_to_df()

    print(sample_data.describe())
    print(sample_data.loc[sample_data['S'] == 0].describe())
    print(sample_data.loc[sample_data['S'] == 1].describe())
