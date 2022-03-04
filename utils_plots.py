

import matplotlib.pyplot as plt
import numpy as np


def label_u(label):
    """Transform labels into readable words for plots

    Args:
        label (str): A or N

    Returns:
        str: apnea or normal respiration
    """

    return 'Apnea' if label == 'A' else 'Normal Respiration' if label == 'N' else ''


def plot_correlation_pts(points, labels, idx_start=4, idx_end=13):
    """plot correlation points

    Args:
        points (values): _description_
        labels (list or array): _description_
        idx_start (int, optional): start range at. Defaults to 4.
        idx_end (int, optional): end range at. Defaults to 13.
    """

    if idx_start is None:
        idx_start = 0
    if idx_end is None:
        idx_end = len(points)

    my_dpi = 100

    plt.figure(figsize=(1350 / my_dpi, 3000 / my_dpi), dpi=my_dpi)
    plt.tight_layout()

    for i in range(idx_start, idx_end):
        choose_idx = i
        plt.subplot(15,1,i+1)
        plt.plot(points[choose_idx],label=label_u(labels[choose_idx]))
        plt.scatter(np.arange(len(points[choose_idx])),points[choose_idx])
        plt.ylim(-0.8,1.3)
        
        plt.legend(loc='lower left')
        plt.show()


def plot_ae_io_patient(input_sigs, output_sigs):
    """Plot input and output signals of one patient all stacked

    Args:
        input_sigs (array): Array of input samples of one user
        output_sigs (array): Array of output sample of one user
    """

    my_dpi = 100
    
    plt.figure(figsize=(1300 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    plt.tight_layout()
    plt.plot(np.hstack(input_sigs), label='Input')

    plt.plot(np.hstack(output_sigs), label='Output')
    plt.xlabel("time (s)")
    plt.ylabel("ADC")
    plt.title('Autoencoder Input-Output Comparison of One Patient- ')
    plt.legend()
    plt.show()


def plot_ae_io_sample(input_seg, output_seg, label):
    """plot autoencoder input and output of one sample

    Args:
        input_seg (array): one sample
        output_seg (array): decoded version of the sample
        label (str): label of sample
    """
    my_dpi = 100

    plt.figure(figsize=(1300 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    plt.tight_layout()
    plt.plot(input_seg, label='Input')
    plt.plot(output_seg, label='Output')
    plt.xlabel("time (s)")
    plt.ylabel("ADC")
    plt.title('Autoencoder Input-Output Comparison - ' + label)
    plt.legend()
    plt.show()