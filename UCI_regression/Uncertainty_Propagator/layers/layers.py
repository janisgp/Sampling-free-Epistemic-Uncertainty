from tensorflow.keras.layers import Layer


class VarPropagationLayer(Layer):
    """
    General layer for variance propagation
    
    properties:
        layer: keras.layers.layer, original keras layer
        use_cov: bool, use full covariance or not
    """
    
    def __init__(self,
                 layer: Layer,
                 use_cov: bool=False,
                 low_rank_cov: int=None,
                 **kwargs):
        self.layer = layer
        self.use_cov = use_cov
        self.low_rank_cov = low_rank_cov
        super(VarPropagationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VarPropagationLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        call method using full covariance or diagonal covariance
        """
        if self.use_cov:
            if self.low_rank_cov is not None:
                out = self._call_low_rank_cov(x)
            else:
                out = self._call_full_cov(x)
        else:
            out = self._call_diag_cov(x)
        return out

    def _call_low_rank_cov(self, x):
        raise NotImplementedError
    
    def _call_full_cov(self, x):
        raise NotImplementedError
        
    def _call_diag_cov(self, x):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    
class ActivationVarPropagationLayer(VarPropagationLayer):
    """
    Specific variance propagation layer for activation functions
    
    Properties:
        inputs: tf.Tensor, input to the activation function from mean propagation stream for computing Jacobian
        exact: bool, (NOT FULLY IMPLEMENTED, only ReLIU) use exact variance propagation through non-linearities
    """
    def __init__(self, inputs, layer=None, **kwargs):
        
        if 'exact' in kwargs:
            self.exact = kwargs['exact']
            del kwargs['exact']
        else:
            self.exact = False
        
        self.inputs = inputs
        
        super(ActivationVarPropagationLayer, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        super(ActivationVarPropagationLayer, self).build(input_shape)
    
    def _call_full_cov(self, x):
        if self.exact:
            out = self._call_full_cov_exact(x)
        else:
            out = self._call_full_cov_approx(x)
        return out
        
    def _call_diag_cov(self, x):
        if self.exact:
            out = self._call_diag_cov_exact(x)
        else:
            out = self._call_diag_cov_approx(x)
        return out
        
    def _call_full_cov_exact(self, x):
        raise NotImplementedError
        
    def _call_full_cov_approx(self, x):
        raise NotImplementedError
    
    def _call_diag_cov_exact(self, x):
        raise NotImplementedError
    
    def _call_diag_cov_approx(self, x):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return input_shape