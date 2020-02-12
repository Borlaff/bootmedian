"""
Bootmedian: A program to calculate robust median by using bootstrapping
simulations on arrays.
v-3.0 - First release to github
v-3.5 - Tesing weights for AMP_equalizer

Inputs:
sample_input: The input sample for which you want to calculate the median.
              It can include NaN.
nsimul: The number of bootstrapping simulations. Increase it little by little
        to not freeze your computer.

errors: Boolean switch. If 1, it will calculate the 1,2,3 sigma confidence
        intervals for the median. If 0, the same output array will be returned,
        but with zeroes.

Output: The output is a len(output) = 7 array.
        np.array([median, s1_up, s1_down, s2_up, s2_down, s3_up, s3_down])
        Where median is the median value for the input sample and sX_up/down
        are its X = 1, 2, 3 sigma up/down confidence limits.

"""

import numpy as np
import multiprocessing
import bottleneck as bn
import pandas as pd

sigma1 = 0.682689492137086
sigma2 = 0.954499736103642
sigma3 = 0.997300203936740

s1_down_q = (1-sigma1)/2
s1_up_q = 1 - s1_down_q
s2_down_q = (1-sigma2)/2
s2_up_q = 1 - s2_down_q
s3_down_q = (1-sigma3)/2
s3_up_q = 1 - s3_down_q

def bootstrap_resample(X, weights=False):
    dataframe = pd.DataFrame(X)

    if not isinstance(weights, bool):
        if bn.nansum(weights) == 0:
            weights = np.ones(len(weights))
        weights = weights/np.max(weights)
        weights_pd = pd.Series(weights)
        sample_pd = dataframe.sample(len(X), weights=weights_pd, replace=True)
        X_resample = np.ndarray.flatten(np.array(sample_pd))
    else:
        sample_pd = dataframe.sample(len(X), replace=True)
        X_resample = np.ndarray.flatten(np.array(sample_pd))

    return X_resample

def mean_bootstrap(argument):
    # arguments = sample, indexes, i
    sample = argument[0]
    weights = argument[1]
    if (len(argument) == 3):
        std1 = argument[2]
        sample = np.random.normal(loc=sample, scale=std1)
    X_resample = bootstrap_resample(X=sample, weights=weights)
    median_boot = bn.nanmean(X_resample)
    return mean_boot


def median_bootstrap(argument):
    # arguments = sample, indexes, i
    sample = argument[0]
    weights = argument[1]
    if (len(argument) == 3):
        std1 = argument[2]
        sample = np.random.normal(loc=sample, scale=std1)
    X_resample = bootstrap_resample(X=sample, weights=weights)
    median_boot = bn.nanmedian(X_resample)
    return median_boot


def bootmedian(sample_input, nsimul=1000, weights=False, errors=1, std=False, verbose=False, nthreads=7, mode="median"):
    if verbose:
        print("Bootmedian v.3.0")

    print_std = False

    if(len(sample_input) == 0):
        output = {"median": np.nan, "s1_up": np.nan, "s1_down": np.nan,
                  "s2_up": np.nan, "s2_down": np.nan, "s3_up": np.nan,
                  "s3_down": np.nan, "std1_up": np.nan, "std1_down": np.nan}
        return(output)

    sample_0 = np.ndarray.flatten(sample_input)
    sample = sample_0[~(np.isnan(sample_0))]

    if not isinstance(weights, bool):
        weights_0 = np.ndarray.flatten(weights)
        weights = weights_0[~(np.isnan(sample_0))]

    if not isinstance(std, bool):
        std_0 = np.ndarray.flatten(std)
        std = std_0[~(np.isnan(sample_0))]
        print_std = True


    n = len(sample)
    if verbose:
        print("The input sample is n = "+str(n)+" large")
    median_boot = np.empty(nsimul)
    if(n == 0):
        if verbose:
            print("Avoiding numeric simulations, the result is zero")
        output = {"median": np.nan, "s1_up": np.nan, "s1_down": np.nan,
                  "s2_up": np.nan, "s2_down": np.nan, "s3_up": np.nan,
                  "s3_down": np.nan, "std1_up": np.nan, "std1_down": np.nan}
        return(output)


    zip_sample = [sample]*nsimul

    if isinstance(weights, bool):
        weights = np.ones(len(sample))

    if isinstance(std, bool):
        std = np.zeros(len(sample))

    weights = weights/bn.nanmax(weights)
    zip_weights = [weights]*nsimul
    zip_std = [std]*nsimul

    arguments = zip(zip_sample, zip_weights, zip_std)

    #return(arguments)
    if nthreads != 1:
        if multiprocessing.cpu_count() > 1:
            num_cores = multiprocessing.cpu_count() - 1
        else:
            num_cores = 1
        #if multiprocessing.cpu_count() > 8:
        #    num_cores = 8

        if verbose:
            print("A total of "+str(num_cores)+" workers joined the cluster!")
        print("A total of "+str(num_cores)+" workers joined the cluster!")

        pool = multiprocessing.Pool(processes=num_cores)
        median_boot = pool.map(median_bootstrap, arguments)
        pool.terminate()

    else:
        median_boot = np.zeros(nsimul)
        median_boot[:] = np.nan
        for i in range(nsimul):
            if mode=="median":
                median_boot[i] = median_bootstrap(arguments[i])
            if mode=="mean":
                median_boot[i] = mean_bootstrap(arguments[i])

    #print(median_boot)
    if mode=="median":
        median = bn.nanmedian(median_boot)
    if mode=="mean":
        median = bn.nanmean(median_boot)

    if(errors == 1):
        s1_up = np.percentile(median_boot, s1_up_q*100)
        s1_down = np.percentile(median_boot, s1_down_q*100)
        s2_up = np.percentile(median_boot, s2_up_q*100)
        s2_down = np.percentile(median_boot, s2_down_q*100)
        s3_up = np.percentile(median_boot, s3_up_q*100)
        s3_down = np.percentile(median_boot, s3_down_q*100)

    if(print_std):
        std1_up = np.percentile(sample_input, s1_up_q*100)
        std1_down = np.percentile(sample_input, s1_down_q*100)
    else:
        std1_up = 0
        std1_down = 0

    if(errors == 0):
        s1_up = 0
        s1_down = 0
        s2_up = 0
        s2_down = 0
        s3_up = 0
        s3_down = 0

    if(errors == 0.5):
        s1_up = np.percentile(median_boot, s1_up_q*100)
        s1_down = np.percentile(median_boot, s1_down_q*100)
        s2_up = 0
        s2_down = 0
        s3_up = 0
        s3_down = 0

    output = {"median": median, "s1_up": s1_up, "s1_down": s1_down,
              "s2_up": s2_up, "s2_down": s2_down, "s3_up": s3_up,
              "s3_down": s3_down, "std1_up": std1_up, "std1_down": std1_down}
#    return(np.array([median, s1_up, s1_down, s2_up, s2_down, s3_up, s3_down,
#                     std1_up, std1_down]))

    return(output)
