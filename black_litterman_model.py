import pandas as pd
import numpy as np

sectors = ['AGG', 'AOR', 'XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV']

#Calculates all the log returns
#Notes:
    #df.shift(1) - finds the previous day's price for each ETF
    #dropna() - removes any rows with NaN values that may result from the shift operation. 
    #This ensures we don't get an error for the first day which doen't have a previous day to compare to.
def log_returns(DataFrame):
    import pandas as pd
    import numpy as np
    log_returns = np.log(DataFrame/DataFrame.shift(1)).dropna()
    return log_returns


def cov_matrix(log_returns):
    import pandas as pd
    import numpy as np
    cov_matrix = log_returns.cov() * 252 #Calculates the covariance matrix of the log returns
    return cov_matrix

def eighty_twenty(cov_matrix):
    import pandas as pd
    #Define weights with explicit Ticker labels
    weights_dict = {
        'XLK': 0.18, 'XLP': 0.09, 'XLB': 0.09, 'XLF': 0.09, 
        'XLV': 0.09, 'XLU': 0.04, 'XLI': 0.09, 'AGG': 0.20, 'AOR': 0.13
    }
    #Create a Series and reindex it to match the Covariance Matrix exactly
    w_series = pd.Series(weights_dict).reindex(cov_matrix.index)
    return w_series

def sixty_fourty(cov_matrix):
    import pandas as pd
    #Define weights with explicit Ticker labels
    weights_dict = {
        'XLK': 0.14, 'XLP': 0.06, 'XLB': 0.06, 'XLF': 0.07, 
        'XLV': 0.07, 'XLU': 0.03, 'XLI': 0.07, 'AGG': 0.40, 'AOR': 0.10
    }
    #Create a Series and reindex it to match the Covariance Matrix exactly
    w_series = pd.Series(weights_dict).reindex(cov_matrix.index)
    return w_series

def ninety_ten(cov_matrix):
    import pandas as pd
    #Define weights with explicit Ticker labels
    weights_dict = {
        'XLK': 0.20, 'XLP': 0.10, 'XLB': 0.10, 'XLF': 0.10, 
        'XLV': 0.10, 'XLU': 0.05, 'XLI': 0.10, 'AGG': 0.10, 'AOR': 0.15
    }
    #Create a Series and reindex it to match the Covariance Matrix exactly
    w_series = pd.Series(weights_dict).reindex(cov_matrix.index)
    return w_series

#Fix WEIGHTS
def simm_bench(cov_matrix):
    import pandas as pd
    #Define weights with explicit Ticker labels
    weights_dict = {
        'XLK': 0.18, 'XLP': 0.09, 'XLB': 0.09, 'XLF': 0.09, 
        'XLV': 0.09, 'XLU': 0.04, 'XLI': 0.09, 'AGG': 0.20, 'AOR': 0.13
    }
    #Create a Series and reindex it to match the Covariance Matrix exactly
    w_series = pd.Series(weights_dict).reindex(cov_matrix.index)
    return w_series

def benchmark_variance(market_series, covariance_matrix):
    #Pandas handles the alignment for you here
    return market_series.T @ covariance_matrix @ market_series

#Calculation to find lambda
def lambda_risk_aversion(benchmark_variance, benchmark):
    E_RM = 0;
    if benchmark == "90/10": E_RM = 0.09
    elif benchmark == "80/20": E_RM = 0.085
    else: E_RM = 0.0725
    RF = 0.0421
    
    return (E_RM-RF)/benchmark_variance

#Calculate the implied Equilubrium returns 
#'@' performs matrix multiplication

def implied_returns(lambda_risk_aversion, cov_matrix, market):
    import numpy as np
    import pandas as pd
    return lambda_risk_aversion * cov_matrix @ market

def view_vectors(views, sectors):
    import numpy as np
    import pandas as pd
    num_views = len(views)
    num_assets = len(sectors)
    
    #Initialize P and Q
    P = np.zeros((num_views, num_assets))
    Q = np.zeros(num_views)
    
    for i, view in enumerate(views):
        #Handle the Pick Matrix (P)
        #Set the target (Winner)
        P[i, sectors.index(view['target'])] = 1
        
        #Set the subsidiary (Loser) if it exists
        if view['subsidiary']:
            P[i, sectors.index(view['subsidiary'])] = -1
            
        #Handle the View Vector (Q)
        #This always takes the expected return value
        Q[i] = view['return']
        
    return P, Q

#Calculate Omega (The uncertainty of the analyst views)
#This creates a diagonal matrix representing the variance of your views
#We use the 'He-Litterman' method: Omega = diag(P * (tau * Sigma) * P.T)

def omega(P_matrix, covariance_matrix):
    import numpy as np
    TAU = 0.025
    
    omega = np.diag(np.diag(P_matrix @ (TAU * covariance_matrix) @ P_matrix.T))
    return omega

def mu_bl(cov_matrix, P_matrix, Q_matrix, omega, implied_returns):
    TAU = 0.025
    
    #Calculate the 'Prior' precision (Tau * Sigma)^-1
    precision_prior = np.linalg.inv(TAU * cov_matrix)

    #Calculate the 'View' precision (P.T * Omega^-1 * P)
    precision_view = P_matrix.T @ np.linalg.inv(omega) @ P_matrix

    #Calculate the combined returns (Mu_BL)
    #Formula: [(Prior_Prec + View_Prec)^-1] @ [Prior_Prec @ Pi + P.T @ Omega^-1 @ Q]
    term1 = np.linalg.inv(precision_prior + precision_view)
    term2 = (precision_prior @ implied_returns) + (P_matrix.T @ np.linalg.inv(omega) @ Q_matrix)
    
    mu_bl = term1 @ term2
    return mu_bl