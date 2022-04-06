"""The 'visualize' module includes functions to create visualizations of the dataset."""

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from src.models.train_model import get_dataset


def data_visualization_univariate(retrieved_dataset, shape: bool = True, head: bool = True, describe: bool = True,
                                  group_by='', box_plot: bool = True, histograms: bool = True) -> None:
    """
    Create a visualization of different univariate characteristics of the dataset.

    :param retrieved_dataset: The parameter holding the dataset
    :param bool shape: Boolean parameter used to trigger the visualization of the dataset's shape on or off
    :param bool head: Boolean parameter used to trigger the visualization of the dataset's head on or off
    :param bool describe: Boolean parameter used to trigger the visualization of the dataset's description, containing
        the count, mean, std, min and max of the dataset, on or off
    :param str group_by: When not an empty string, groups the dataset by the category corresponding to the contents of
        the string
    :param bool box_plot: Boolean parameter used to trigger the visualization of the dataset's box plot
    :param bool histograms: Boolean parameter used to trigger the visualization of the dataset's histogram
    :return:
    """
    # shape
    if shape:
        print('\n', retrieved_dataset.shape)

    # head
    if head:
        print('\n', retrieved_dataset.head(20))

    # descriptions
    if describe:
        print('\n', retrieved_dataset.describe())

    # class distribution
    if group_by != '':
        print('\n', retrieved_dataset.groupby(group_by).size())

    # box and whisker plots
    if box_plot:
        retrieved_dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)

    # histograms
    if histograms:
        retrieved_dataset.hist()


def data_visualization_multivariate(retrieved_dataset) -> None:
    """
    Create a visualization of the multivariate characteristic of the dataset known as scatter plot matrix.

    :param retrieved_dataset: The parameter holding the dataset
    :return:
    """
    # scatter plot matrix
    scatter_matrix(retrieved_dataset)
    pyplot.show()


if __name__ == '__main__':
    retrieved_dataset = get_dataset('external')
    data_visualization_univariate(retrieved_dataset=retrieved_dataset, group_by='Genre')
    data_visualization_multivariate(retrieved_dataset)
