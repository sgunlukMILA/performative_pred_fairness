# IMPORT LIBRARIES

import numpy as np
import pandas as pd


# DATA GENERATING CLASSES

class ScalarLinearDecisionModel():
  """
    A class for generating simulated data based on our scalar linear decision model.

    Attributes:
        n_samples (int): Number of samples to generate.
        p_majority (float): Fraction of population that are majority members (i.e. E[S]). 
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
        __init__(params): Initialize the ScalarLinearDecisionModel using params, 
                          a dictionary containing model parameters.
        generate_basic_data(): Generates simulated data based on the model.
        generate_do_A_data(new_A): Generates data for improved set of ancestral featues.
        generate_improve_A_data(diff_A): Generates data by improving A by diff_A.
        generate_do_C_data(new_C): Generates data for improved set of causal featues.
        generate_improve_C_data(diff_C): Generates data by improving C by diff_C.
        ts_to_df(describe=False): Converts simulated data to a DataFrame with describe being a printer attribute.
  """
  def __init__(self, params):
    #n_samples, s_a_const = 1, a_var = 1, s_c_const = 0,  a_c_const = 1, c_var = 1, s_y_const = 1, c_y_const = 1, y_var = 0.1):
    self.n_samples = params['n_samples']
    self.features_dim = 2
    self.intervention_dim = 2
    self.p_majority = params['p_majority']
    self.s_a_const = params['s_a_const']
    self.a_var = params['a_var']
    self.a_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.a_var*np.ones(self.n_samples))
    self.s_c_const = params['s_c_const']
    self.a_c_const = params['a_c_const']
    self.c_var = params['c_var']
    self.c_noise = np.random.normal(loc = np.zeros(self.n_samples), scale=self.c_var*np.ones(self.n_samples))
    self.s_y_const = params['s_y_const']
    self.c_y_const = params['c_y_const']
    self.y_var = params['y_var']
    #y_noise is no longer used, only bern sampling instead
    self.y_noise = np.random.normal(loc = np.zeros(self.n_samples), scale = self.y_var*np.ones(self.n_samples))
    self.S = np.random.binomial(n=1, p = self.p_majority, size=self.n_samples)
    #make labels s = {-1, 1} by doing (2S-1)
    self.S_sym = 2*self.S - 1
    self.A = np.empty((self.n_samples,1))
    self.C = np.empty((self.n_samples,1))
    self.Y_logit = np.empty((self.n_samples,1))
    self.Y = np.empty((self.n_samples,1))
    self.mask = np.ones((self.n_samples), dtype=bool)
    self.is_improvable = None
    #self.necessary_improv = np.empty((self.n_samples, 2))
  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))
  def generate_basic_data(self):
    self.A = self.S_sym*self.s_a_const + self.a_noise
    self.C = self.A*self.a_c_const + self.S_sym*self.s_c_const + self.c_noise
    self.Y_logit = self.C*self.c_y_const + self.S_sym*self.s_y_const
    self.Y = np.random.binomial(n=1, p = self.sigmoid(self.Y_logit)) #np.where(self.Y_logit > 0, 1, 0) #
  def generate_improve_data(self, data, diff_vec):
    temp_A = diff_vec[0] + data['A'] 
    temp_C = diff_vec[1] + (temp_A*self.a_c_const + (2*data['S'] - 1)*self.s_c_const + data['c_noise']) 
    return np.array([temp_A, temp_C])
  def generate_y_logit(self, data, diff_vec):
    temp_A = diff_vec[:, 0] + data["A"] 
    temp_C = diff_vec[:, 1] + (temp_A*self.a_c_const + (2*data['S'] - 1)*self.s_c_const + data['c_noise']) 
    return temp_C*self.c_y_const + (2*data['S'] - 1)*self.s_y_const
  def get_data_df(self):
    sim_data = pd.DataFrame()
    sim_data["S"] = self.S
    sim_data["c_noise"] = self.c_noise
    sim_data["A"] = self.A
    sim_data["C"] = self.C
    sim_data["Y_logit"] = self.Y_logit
    sim_data["Y"] = self.Y
    sim_data["is_improvable"] = self.is_improvable
    return sim_data
  def get_features(self):
    return np.stack((self.A, self.C), axis=1)
#np.stack((self.A.reshape((filter_shape,1)), self.C.reshape((filter_shape,1))), axis=1)
  def ts_to_df(self, describe = False):
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

  def get_real_boundary(self, s):
        """
        the real boundary where the true data generation process assigns E[Y]=0.5
        This quantity depends on what Y is conditioned on.
        The simplest one to derive (and arguably the "truest" one) is E[Y|C,S]=0.5, which we use for now

        For a given S, the scalar value of C is returned, which yields E[Y|C,S]=0.5
        Args:
            s: value of S variable \in -1, 1

        Returns: value of C which yields E[Y|C,S] = 0.5

        """
        return -(self.s_y_const/self.c_y_const) * s
    
  def store_is_improvable(self, real_best_delta, max_delta):
    """
        computes and stores a bool array, indicating whether each agent can improve with a budget of max_delta
        Args:
            real_best_delta: np.array, shape: (n,) lowest delta with which each agent can increase their P(Y) past 0.5,
            if it is < 0.5 (should be computed outside this class)
            max_delta: max effort allowed

        Returns:

    """
    self.is_improvable = (real_best_delta < max_delta) & (real_best_delta > 0)



# DATA GENERATING CLASS

class NewScalarLinearDecisionModel():
  """
    A class for generating simulated data based on our scalar linear decision model.

    Attributes:
        n_samples (int): Number of samples to generate.
        p_majority (float): Fraction of population that are majority members (i.e. E[S]).
        s_a_const (float): Coefficient for S in generating A (alpha).
        a_var (float): Variance for A noise (epsilon_A^2).
        a_noise (numpy.ndarray): Noise added to A based on a_var (u_A).
        a_c_const (float): Coefficient for A in generating C (omega_A).
        c_var (float): Variance for C noise (epsilon_C^2).
        c_noise (numpy.ndarray): Noise added to C based on c_var (u_C).
        c_y_const (float): Coefficient for C in generating Y (m_C).
        s_spur_const(float): Coefficient for S in generating Spurious (beta_S).
        y_spur_const(float): Coefficient for Y in generating Spurious (beta_Y).
        u_var (float): Variance for U noise (epsilon_spur^2).
        spur_noise (numpy.ndarray): Noise added to Spurious based on spur_var (u_spur).
        S (numpy.ndarray): Binary variable indicating the treatment {0, 1}.
        S_sym (numpy.ndarray): Binary variable converted to {-1, 1}.
        A (numpy.ndarray): Simulated values of variable A.
        C (numpy.ndarray): Simulated values of variable C.
        Y (numpy.ndarray): Simulated values of binary outcome Y.

    Methods:
        __init__(params): Initialize the NewScalarLinearDecisionModel using params, 
                          a dictionary containing model parameters.
        generate_basic_data(): Generates simulated data based on the model.
        generate_do_A_data(new_A): Generates data for improved set of ancestral featues.
        generate_improve_A_data(diff_A): Generates data by improving A by diff_A.
        generate_do_C_data(new_C): Generates data for improved set of causal featues.
        generate_improve_C_data(diff_C): Generates data by improving C by diff_C.
        ts_to_df(describe=False): Converts simulated data to a DataFrame with describe being a printer attribute.
  """
  def __init__(self, params):
    self.n_samples = params['n_samples']
    self.features_dim = params['input_dim']
    self.intervention_dim = params['intervention_dim']
    self.p_majority = params['p_majority']
    self.s_u_const = params['s_u_const']
    self.u_var = params['u_var']
    self.u_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.u_var*np.ones(self.n_samples))
    self.s_a_const = params['s_a_const']
    self.u_a_const = params['u_a_const']
    self.a_var = params['a_var']
    self.a_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.a_var*np.ones(self.n_samples))
    self.s_c_const = params['s_c_const']
    self.c_var = params['c_var']
    self.c_noise = np.random.normal(loc = np.zeros(self.n_samples), scale=self.c_var*np.ones(self.n_samples))
    self.c_y_const = params['c_y_const']
    self.u_y_const = params['u_y_const']
    #y_noise is no longer used, only bern sampling instead
    self.S = np.random.binomial(n=1, p = self.p_majority, size=self.n_samples)
    #make labels s = {-1, 1} by doing (2S-1)
    self.S_sym = 2*self.S - 1
    self.U = np.empty((self.n_samples,1))
    self.A = np.empty((self.n_samples,1))
    self.C = np.empty((self.n_samples,1))
    self.Y_logit = np.empty((self.n_samples,1))
    self.Y = np.empty((self.n_samples,1))
    self.mask = np.ones((self.n_samples), dtype=bool)
    self.is_improvable = None
  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))
  def generate_basic_data(self):
    self.U = np.random.binomial(n=1, p=(0.5 + 0.25*self.S_sym*self.s_u_const)*np.ones(self.n_samples))
    self.A = self.S_sym*self.s_a_const + self.U*self.u_a_const + self.a_noise
    self.C = self.S_sym*self.s_c_const + self.c_noise
    self.Y_logit = self.C*self.c_y_const + self.U*self.u_y_const 
    self.Y = np.random.binomial(n=1, p = self.sigmoid(self.Y_logit)) #np.where(self.Y_logit > 0, 1, 0) #
  def generate_improve_data(self, data, diff_vec):
    temp_U = data['U']
    if self.intervention_dim == 3:
      temp_U += diff_vec[2] 
    temp_A = diff_vec[0] + (2*data['S'] - 1)*self.s_a_const + temp_U*self.u_a_const + data['a_noise'] 
    temp_C = diff_vec[1] + (2*data['S'] - 1)*self.s_c_const + data['c_noise']
    if self.features_dim == 2:
      return np.array([temp_A, temp_C])
    elif self.features_dim == 3:
      return np.array([temp_A, temp_C, temp_U])
    else:
      print("ERROR: Invalid feature dimension")
      return None
  def generate_y_logit(self, data, diff_vec):
    temp_U = data['U']
    if self.intervention_dim == 3:
      temp_U += diff_vec[2] 
    temp_A = diff_vec[:, 0] + (2*data['S'] - 1)*self.s_a_const + temp_U*self.u_a_const + data['a_noise'] 
    temp_C = diff_vec[:, 1] + (2*data['S'] - 1)*self.s_c_const + data['c_noise']
    return temp_C*self.c_y_const + temp_U*self.u_y_const 
  def get_data_df(self):
    sim_data = pd.DataFrame()
    sim_data["S"] = self.S
    sim_data["a_noise"] = self.a_noise
    sim_data["c_noise"] = self.c_noise
    sim_data["u_noise"] = self.u_noise
    sim_data["A"] = self.A
    sim_data["C"] = self.C
    sim_data["U"] = self.U
    sim_data["U_binary"] = self.U #- self.u_noise
    sim_data["Y_logit"] = self.Y_logit
    sim_data["Y"] = self.Y
    sim_data["is_improvable"] = self.is_improvable
    return sim_data
  def get_features(self):
    if self.features_dim == 2:
      return np.stack((self.A, self.C), axis=1)
    elif self.features_dim == 3:
      return np.stack((self.A, self.C, self.U), axis=1)
    else:
      print("ERROR: Invalid feature dimension")
      return None
  def ts_to_df(self, describe = False):
    sim_data = pd.DataFrame()
    sim_data["S"] = self.S
    sim_data["A"] = self.A
    sim_data["C"] = self.C
    sim_data["Y"] = self.Y
    sim_data["U"] = self.U
    if describe:
      print(sim_data.describe())
      print(sim_data.loc[sim_data['S'] == 0].describe())
      print(sim_data.loc[sim_data['S'] == 1].describe())
    return sim_data

  def get_real_boundary(self, s, u = 0.5):
        """
        the real boundary where the true data generation process assigns E[Y]=0.5
        This quantity depends on what Y is conditioned on.
        The simplest one to derive (and arguably the "truest" one) is E[Y|C,S]=0.5, which we use for now

        For a given S, the scalar value of C is returned, which yields E[Y|C,S]=0.5
        Args:
            s: value of S variable \in -1, 1

        Returns: value of C which yields E[Y|C,S,U] = 0.5

        """
        return -(self.u_y_const/self.c_y_const) * u
    
  def store_is_improvable(self, real_best_delta, max_delta):
    """
        computes and stores a bool array, indicating whether each agent can improve with a budget of max_delta
        Args:
            real_best_delta: np.array, shape: (n,) lowest delta with which each agent can increase their P(Y) past 0.5,
            if it is < 0.5 (should be computed outside this class)
            max_delta: max effort allowed

        Returns:

    """
    self.is_improvable = (real_best_delta < max_delta) & (real_best_delta > 0)



# DATA GENERATING CLASS

class SelectionBiasDecisionModel():
  """
    A class for generating simulated data based on the selection bias version of the scalar linear decision model.

    Attributes:
        n_samples (int): Number of samples to generate.
        p_majority (float): Fraction of population that are majority members (i.e. E[S]).
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
        __init__(params): Initialize the ScalarLinearDecisionModel using params, 
                          a dictionary containing model parameters.
        generate_basic_data(): Generates simulated data based on the model.
        generate_do_A_data(new_A): Generates data for improved set of ancestral featues.
        generate_improve_A_data(diff_A): Generates data by improving A by diff_A.
        generate_do_C_data(new_C): Generates data for improved set of causal featues.
        generate_improve_C_data(diff_C): Generates data by improving C by diff_C.
        ts_to_df(describe=False): Converts simulated data to a DataFrame with describe being a printer attribute.
  """
  def __init__(self, params):
    #n_samples, s_a_const = 1, a_var = 1, s_c_const = 0,  a_c_const = 1, c_var = 1, s_y_const = 1, c_y_const = 1, y_var = 0.1):
    self.n_samples = params['n_samples']
    self.features_dim = 2
    self.p_majority = params['p_majority']
    self.s_a_const = params['s_a_const']
    self.a_var = params['a_var']
    self.a_noise = np.random.normal(loc=np.zeros(self.n_samples), scale=self.a_var*np.ones(self.n_samples))
    self.s_c_const = params['s_c_const']
    self.a_c_const = params['a_c_const']
    self.c_var = params['c_var']
    self.c_noise = np.random.normal(loc = np.zeros(self.n_samples), scale=self.c_var*np.ones(self.n_samples))
    self.s_y_const = params['s_y_const']
    self.c_y_const = params['c_y_const']
    self.y_var = params['y_var']
    #y_noise is no longer used, only bern sampling instead
    self.y_noise = np.random.normal(loc = np.zeros(self.n_samples), scale = self.y_var*np.ones(self.n_samples))
    self.S = np.random.binomial(n=1, p = self.p_majority, size=self.n_samples)
    #make labels s = {-1, 1} by doing (2S-1)
    self.S_sym = 2*self.S - 1
    self.A = np.empty((self.n_samples,1))
    self.C = np.empty((self.n_samples,1))
    self.D = np.empty((self.n_samples,1))
    self.Y_logit = np.empty((self.n_samples,1))
    self.Y = np.empty((self.n_samples,1))
    self.mask = None
    self.is_improvable = None
  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))
  def generate_basic_data(self):
    self.A = self.S_sym*self.s_a_const + self.a_noise
    self.C = self.A*self.a_c_const + self.S_sym*self.s_c_const + self.c_noise
    self.Y_logit = self.C*self.c_y_const 
    self.Y = np.random.binomial(n=1, p = self.sigmoid(self.Y_logit)) #np.where(self.Y_logit > 0, 1, 0) #
    self.D = np.random.binomial(n=1, p = self.sigmoid(self.Y_logit + self.S_sym*self.s_y_const))
    #np.where(self.Y_logit + self.S_sym*self.s_y_const > 0, 1, 0)#
    self.mask = (self.D == 1)
  def generate_improve_data(self, data, diff_vec):
    temp_A = data['A'] + diff_vec[0]
    temp_C = diff_vec[1] + (temp_A*self.a_c_const + (2*data['S'] - 1)*self.s_c_const + data['c_noise']) 
    return np.array([temp_A, temp_C])#
  def generate_y_logit(self, data, diff_vec):
    temp_A = data['A'] + diff_vec[0]
    temp_C = diff_vec[1] + (temp_A*self.a_c_const + (2*data['S'] - 1)*self.s_c_const + data['c_noise']) 
    return temp_C*self.c_y_const
  def get_data_df(self):
    sim_data = pd.DataFrame()
    sim_data["S"] = self.S
    sim_data["a_noise"] = self.a_noise
    sim_data["c_noise"] = self.c_noise
    sim_data["A"] = self.A
    sim_data["C"] = self.C
    sim_data["Y_logit"] = self.Y_logit
    sim_data["Y"] = self.Y
    sim_data["is_improvable"] = self.is_improvable
    return sim_data
  def ts_to_df(self, describe = False):
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

  def get_real_boundary(self, s):
        """
        the real boundary where the true data generation process assigns E[Y]=0.5
        This quantity depends on what Y is conditioned on.
        The simplest one to derive (and arguably the "truest" one) is E[Y|C,S]=0.5, which we use for now

        For a given S, the scalar value of C is returned, which yields E[Y|C,S]=0.5
        Args:
            s: value of S variable \in -1, 1

        Returns: value of C which yields E[Y|C,S] = 0.5

        """
        return -(self.s_y_const/self.c_y_const) * s

  def store_is_improvable(self, real_best_delta, max_delta):
    """
        computes and stores a bool array, indicating whether each agent can improve with a budget of max_delta
        Args:
            real_best_delta: np.array, shape: (n,) lowest delta with which each agent can increase their P(Y) past 0.5,
            if it is < 0.5 (should be computed outside this class)
            max_delta: max effort allowed

        Returns:

    """
    self.is_improvable = (real_best_delta < max_delta) & (real_best_delta > 0)
    
    
if __name__ == "__main__":
    # Data generation test call
    default_params = {
        'n_samples': 1000, 
        'input_dim': 2,
        'intervention_dim': 2,
        'p_majority': 0.5,
        's_u_const': 0,
        'u_var': 1,
        's_a_const': 1, 
        'u_a_const': 1,
        'a_var': 1, 
        's_c_const': 1,  
        'a_c_const': 1, 
        'c_var': 1, 
        's_y_const': 1, 
        'c_y_const': 1,
        'u_y_const': 1,
        'y_var': 0.1,
    
}

    sample_dm = ScalarLinearDecisionModel(default_params)
    sample_dm.generate_basic_data()
    sample_data = sample_dm.ts_to_df()

    print(sample_data.describe())
    print(sample_data.loc[sample_data['S'] == 0].describe())
    print(sample_data.loc[sample_data['S'] == 1].describe())
  





