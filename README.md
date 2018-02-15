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
