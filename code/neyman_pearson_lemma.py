import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gaussian(mu, sigma, alpha, ax, 
                  crit_value=None, 
                  pdf_color='red',
                  pdf_label=None, 
                  shaded_color='lightcoral',
                  shaded_label=None, 
                  upper_tail=True):
    # Generate data points for the x-axis
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

    # Calculate the probability density function (pdf) for the Gaussian distribution
    y = norm.pdf(x, mu, sigma)

    # Plot the bell curve
    ax.plot(x, y, color=pdf_color, label=f'N({mu}, {sigma**2})' if pdf_label is None else pdf_label)

    # Find x-value corresponding to P(X > alpha)
    if(crit_value is None):
        x_alpha = norm.ppf(1 - alpha, mu, sigma)
    else:
        x_alpha = crit_value

    # Shade the region where P(X > alpha)
    if(upper_tail):
        x_shade = np.linspace(x_alpha, mu + 4 * sigma, 1000)
    else:
        x_shade = np.linspace(x_alpha, mu - 4 * sigma, 1000)
    y_shade = norm.pdf(x_shade, mu, sigma)
    ax.fill_between(x_shade, y_shade, color=shaded_color, alpha=0.5, 
                    label=f'P(X > {x_alpha:.2f})' if shaded_label is None else shaded_label)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Probability Density')

    # Return values
    return x_alpha


if __name__ == '__main__':
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x_alpha = plot_gaussian(-1, 1, 0.05, ax, 
                            pdf_label='H0 density',
                            pdf_color='red', 
                            shaded_label='α',
                            shaded_color='lightcoral')
    x_beta = plot_gaussian(2, 1, 0.05, ax, 
                           pdf_label='H1 density',
                           pdf_color='green', 
                           shaded_label='β',
                           shaded_color='green',
                           crit_value=x_alpha, 
                           upper_tail=False)

    # Show the plot
    plt.legend()
    plt.savefig(r'../figures/typeI_typeII_errors.png')
    plt.show()


    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    x_alpha = plot_gaussian(-1, 1, 0.05, ax[0], 
                            pdf_label='H0 density',
                            pdf_color='red', 
                            shaded_label='α',
                            shaded_color='lightcoral')
    x_beta = plot_gaussian(2, 1, 0.05, ax[0], 
                           pdf_label='H1 density',
                           pdf_color='green', 
                           shaded_label='β',
                           shaded_color='green',
                           crit_value=x_alpha, 
                           upper_tail=False)
    ax[0].set_title('Type I & Type II errors when pvalue=0.05')
    
    x_alpha = plot_gaussian(-1, 1, 0.01, ax[1], 
                            pdf_label='H0 density',
                            pdf_color='red', 
                            shaded_label='α',
                            shaded_color='lightcoral')
    x_beta = plot_gaussian(2, 1, 0.01, ax[1], 
                           pdf_label='H1 density',
                           pdf_color='green', 
                           shaded_label='β',
                           shaded_color='green',
                           crit_value=x_alpha, 
                           upper_tail=False)
    ax[1].set_title('Type I & Type II errors when pvalue=0.01')

    # Show the plot
    plt.legend()
    plt.savefig(r'../figures/typeI_typeII_errors_tradeoff.png')
    plt.show()