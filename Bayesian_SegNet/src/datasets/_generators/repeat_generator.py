"""A method to repeat a generators output for multi-IO models."""


def repeat_generator(
    x_gen: 'DataGenerator',
    y_gen: 'DataGenerator',
    x_repeats: int=0,
    y_repeats: int=0
) -> 'DataGenerator':
    """
    Return a generator that repeats x and y input generators.

    Args:
        x_gen: a directory for x data
        y_gen: a generator for y data
        x_repeats: the number of times to repeat the x data (default 0)
        y_repeats: the number of times to repeat the y data (default 0)

    Returns:
        a new generator that returns a tuple of (X, y) lists with sizes:
        - (x_repeats + 1)
        - (y_repeats + 1)

    """
    # create a mapping function to repeat the x and y inputs
    def repeat_outputs(_x, _y):
        """
        Repeat the generator outputs of input generators.

        Args:
            _x: the x generator to repeat
            _y: the y generator to repeat

        Returns: a list with inputs repeated x_repeats and y_repeats times

        """
        # repeat if there is a positive repeat rate
        if x_repeats > 0:
            _x = [_x] * (x_repeats + 1)
        # repeat if there is a positive repeat rate
        if y_repeats > 0:
            _y = [_y] * (y_repeats + 1)
        # return the updated x and y values
        return _x, _y
    # return the new mapping for the generators
    return map(repeat_outputs, x_gen, y_gen)


# explicitly define the outward facing API of this module
__all__ = [repeat_generator.__name__]
