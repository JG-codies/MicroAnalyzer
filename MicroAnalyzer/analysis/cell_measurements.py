def get_cell_length(cell):
    """
    calculate the length of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the length of the given cell in pixels
    """
    return cell.length


def get_cell_width(cell):
    """
    calculate the width of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the width of the given cell in pixels
    """
    return cell.radius * 2


def get_cell_area(cell):
    """
    calculate the area of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the area of the given cell in pixels
    """
    return cell.area


def get_cell_radius(cell):
    """
    calculate the radius of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the radius of the given cell in pixels
    """
    return cell.radius


def get_cell_circumference(cell):
    """
    calculate the circumference of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the circumference of the given cell in pixels
    """
    return cell.circumference


def get_cell_surface_area(cell):
    """
    calculate the surface area of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the surface area of the given cell in pixels
    """
    return cell.surface


def get_cell_volume(cell):
    """
    calculate the volume of a given cell.
    :param cell: an optimized colicoords.Cell object
    :return: the volume of the given cell in pixels
    """
    return cell.volume
