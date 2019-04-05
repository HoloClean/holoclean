from pyitlib import discrete_random_variable as drv

from utils import NULL_REPR

def compute_norm_cond_entropy_corr(data_df, attrs_from, attrs_to):
    """
    Computes the correlations between attributes by calculating
    the normalized conditional entropy between them. The conditional
    entropy is asymmetric, therefore we need pairwise computation.

    The computed correlations are stored in a dictionary in the format:
    {
      attr_a: { cond_attr_i: corr_strength_a_i,
                cond_attr_j: corr_strength_a_j, ... },
      attr_b: { cond_attr_i: corr_strength_b_i, ...}
    }

    :return a dictionary of correlations
    """
    corr = {}
    # Compute pair-wise conditional entropy.
    for x in attrs_from:
        corr[x] = {}
        for y in attrs_to:
            # Set correlation to 1 for same attributes.
            if x == y:
                corr[x][y] = 1.0
                continue

            xy_df = data_df[[x, y]]
            xy_df = xy_df.loc[~(xy_df[x] == NULL_REPR) & ~(xy_df[y] == NULL_REPR)]
            x_vals = xy_df[x]
            x_domain_size = x_vals.nunique()

            # Set correlation to 0.0 if entropy of x is 1 (only one possible value).
            if x_domain_size == 1 or len(xy_df) == 0:
                corr[x][y] = 0.0
                continue

            # Compute the conditional entropy H(x|y) = H(x,y) - H(y).
            # H(x,y) denotes H(x U y).
            # If H(x|y) = 0, then y determines x, i.e., y -> x.
            # Use the domain size of x as a log base for normalization.
            y_vals = xy_df[y]

            x_y_entropy = drv.entropy_conditional(x_vals, y_vals, base=x_domain_size).item()

            # The conditional entropy is 0 for strongly correlated attributes and 1 for
            # completely independent attributes. We reverse this to reflect the correlation.
            corr[x][y] = 1.0 - x_y_entropy
    return corr
