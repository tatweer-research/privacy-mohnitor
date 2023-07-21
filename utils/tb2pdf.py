import os.path

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import numpy as np


# Function to extract data from TensorBoard experiment
def read_tensorboard_experiment(logdir: str):
    """
    Reads a TensorBoard experiment and returns a dictionary with the data.
    Args:
        logdir (str): Path to the TensorBoard experiment.
    Returns:
        data (dict): Dictionary with the data from the TensorBoard experiment.
    """
    event_acc = EventAccumulator(logdir, size_guidance={'tensors': 0, 'scalars': 0})
    event_acc.Reload()

    # Get all tensor metrics
    tensor_tags = event_acc.Tags()["tensors"]

    data = {}
    for tag in tensor_tags:
        data[tag] = []
        for event in event_acc.Tensors(tag):
            data[tag].append(float(tf.make_ndarray(event.tensor_proto)))

    # Get all scalar metrics
    scalar_tags = event_acc.Tags()["scalars"]
    for tag in scalar_tags:
        data[tag] = []
        for event in event_acc.Scalars(tag):
            data[tag].append(event.value)
    return data


def create_pdf_from_tensorboard(logdir: str, report_dir: str):
    """
    Create a PDF report from a TensorBoard experiment.
    Args:
        logdir (str): Path to the TensorBoard experiment.
    """
    exp_data = read_tensorboard_experiment(logdir)

    # Generate a PDF report
    sns.set(style="darkgrid")

    # Create a PDF document to save the figures
    pdf_pages = PdfPages(os.path.join(report_dir, "training_report.pdf"))

    for column in exp_data.keys():
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(exp_data[column])), exp_data[column], label=column)
        plt.legend(loc="best")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"{column}")

        # Save the figure to the PDF document
        pdf_pages.savefig(plt.gcf())
        plt.close()

    # Close the PDF document
    pdf_pages.close()


if __name__ == '__main__':
    # Define the experiment ID
    logdir = r"/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/expermiments/"
    create_pdf_from_tensorboard(logdir, '')
