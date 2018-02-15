"""
Bootmedian: A program to calculate robust median by using bootstrapping
simulations on arrays.
v-3.0 - First release to github

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


def median_bootstrap(argument):
    # arguments = sample, indexes, i
    sample = argument[0]
    indexes = np.random.randint(low=0,high=len(sample),size=len(sample))
    x = sample[indexes]
    median_boot = bn.nanmedian(x)
    return median_boot


def bootmedian(sample_input, nsimul, errors=1):
    print("Bootmedian v.3.0")
    sigma1 = 0.682689492137086
    sigma2 = 0.954499736103642
    sigma3 = 0.997300203936740

    if(len(sample_input) == 0):
        return(np.array([0,0,0,0,0,0,0]))

    s1_down_q = (1-sigma1)/2
    s1_up_q = 1 - s1_down_q
    s2_down_q = (1-sigma2)/2
    s2_up_q = 1 - s2_down_q
    s3_down_q = (1-sigma3)/2
    s3_up_q = 1 - s3_down_q

    sample_0 = np.ndarray.flatten(sample_input)
    sample = sample_0[~(np.isnan(sample_0))]
    n = len(sample)
    print("The input sample is n = "+str(n)+" large")
    median_boot = np.empty(nsimul)
    if(n == 0):
        print("Avoiding numeric simulations, the result is zero")
        return(np.array([0, 0, 0, 0, 0, 0, 0]))

    num_cores = multiprocessing.cpu_count() - 2
    print("A total of "+str(num_cores)+" workers joined the cluster!")

    zip_sample = [sample]*nsimul

    arguments = zip(zip_sample,range(nsimul))

    #return(arguments)
    pool = multiprocessing.Pool(processes=num_cores)
    median_boot = pool.map(median_bootstrap, arguments)
    #print(median_boot)

    median = bn.nanmedian(median_boot)
    if(errors == 1):
        s1_up = np.percentile(median_boot,s1_up_q*100)
        s1_down = np.percentile(median_boot,s1_down_q*100)
        s2_up = np.percentile(median_boot,s2_up_q*100)
        s2_down = np.percentile(median_boot,s2_down_q*100)
        s3_up = np.percentile(median_boot,s3_up_q*100)
        s3_down = np.percentile(median_boot,s3_down_q*100)

    if(errors == 0):
        s1_up = 0
        s1_down = 0
        s2_up = 0
        s2_down = 0
        s3_up = 0
        s3_down = 0

    pool.terminate()
    return(np.array([median, s1_up, s1_down, s2_up, s2_down, s3_up, s3_down]))
