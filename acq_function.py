import numpy as np
from scipy.stats import norm
from torch import mean, std
def acq_fn(predictions,ytrain=None,stds=None,explore_type='its'):

    if explore_type == 'ei':
        ei_calibration_factor = 5.
        max_y = np.max(ytrain)
        factored_stds = stds/ei_calibration_factor
        gam = [(predictions[i]-max_y)/factored_stds[i] for i in range(len(mean))]
        ei = [factored_stds[i]*(gam[i] * norm.cdf(gam[i])+norm.pdf(gam[i])) for i in range(predictions)]
        return ei
    elif explore_type == 'its':
        return np.random.normal(predictions,stds)

    return None
    
