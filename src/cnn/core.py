def get_axis_indexes(kernel_axis_length, center_index):
    """Calculate the kernel indexes on a certain axis depending on the kernel
    center.

    Args:
        kernel_axis_length (int): The length of the single axis of the
            convolutional kernel.
        center_index (int): The index of the kernel center on a certain axis.
    """
    axis_indexes = []
    for i in range(-center_index, kernel_axis_length - center_index):
        axis_indexes.append(i)
    return axis_indexes


def get_axes_indexes(kernel_size, center_indexes):
    """Calculate the kernel axes indexes depending on the kernel center.

    Args:
        kernel_size (tuple of int): The size of the convolutional kernel. The
            first index should be on the x-axis, and the second on the y-axis.
        center_indexes (tuple of int): The kernel center indexes. The first
            index should be on the x-axis, and the second on the y-axis.
    """
    indexes_x = get_axis_indexes(
        kernel_axis_length=kernel_size[0],
        center_index=center_indexes[0]
    )
    indexes_y = get_axis_indexes(
        kernel_axis_length=kernel_size[1],
        center_index=center_indexes[1]
    )
    return indexes_x, indexes_y
