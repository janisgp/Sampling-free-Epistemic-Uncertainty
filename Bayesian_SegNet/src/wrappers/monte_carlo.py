"""A model wrapper to perform Monte Carlo simulations."""
import numpy as np
from keras.models import Model


class MonteCarlo(object):
    """A model wrapper to perform Monte Carlo simulations."""

    def __init__(self, model: Model, simulations: int, uncertainty: str='var'):
        """
        Initialize a new Monte Carlo model wrapper.

        Args:
            model: the Bayesian model to estimate mean output using Monte Carlo
            simulations: the number of simulations to estimate mean
            uncertainty: the type of uncertainty as either 'var' or 'entropy'

        Returns:
            None

        """
        # type check the model and store
        if not isinstance(model, Model):
            raise TypeError('model must be of type {}'.format(Model))
        self.model = model
        # type check the simulations parameter and store
        try:
            self.simulations = int(simulations)
        except ValueError:
            raise TypeError('simulations must be an integer')
        # type check the uncertainty parameter
        try:
            self.uncertainty = str(uncertainty)
        except ValueError:
            raise TypeError('uncertainty must be a string')
        # make sure uncertainty is a legal value
        if self.uncertainty not in {'var', 'entropy'}:
            raise ValueError('uncertainty must be either "var" or "entropy"')

    @property
    def input_shape(self):
        """Return the input shape of the model for this Monte Carlo."""
        return self.model.input_shape

    @property
    def output_shape(self):
        """Return the output shape of the model for this Monte Carlo."""
        return self.model.output_shape

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return mean target and output variance for given inputs.

        Args:
            args: the positional arguments for evaluate_generator
            kwargs: the keyword arguments for evaluate_generator

        Returns:
            a tuple of:
            - mean predictions over self.simulations passes
            - variance of predictions over self.simulations passes

        """
        # create a list to store the output predictions in
        simulations = [None] * self.simulations
        # predict for the number of simulations
        for idx in range(self.simulations):
            simulations[idx] = self.model.predict(X)
        # take the mean prediction in the simulations
        mean = np.mean(simulations, axis=0)
        # return the mean and the variance
        if self.uncertainty == 'var':
            return mean, np.mean(np.var(simulations, axis=0), axis=-1)
        # return the mean and the entropy of the means
        elif self.uncertainty == 'entropy':
            return mean, -1 * np.sum(np.log(mean) * mean, axis=-1)
        # unrecognized value, raise error
        raise ValueError('self.uncertainty must be either "var" or "entropy"')


# explicitly define the outward facing API of this module
__all__ = [MonteCarlo.__name__]
