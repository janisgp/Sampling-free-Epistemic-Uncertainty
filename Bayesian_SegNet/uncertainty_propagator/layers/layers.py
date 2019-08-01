from keras.layers import Layer


class VarPropagationLayer(Layer):

    def __init__(self, layer, use_cov=False, **kwargs):
        self.layer = layer
        self.use_cov = use_cov
        super(VarPropagationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VarPropagationLayer, self).build(input_shape)

    def call(self, x):
        if self.use_cov:
            out = self._call_full_cov(x)
        else:
            out = self._call_diag_cov(x)
        return out
    
    def _call_full_cov(self, x):
        raise NotImplementedError
        
    def _call_diag_cov(self, x):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    
class ActivationVarPropagationLayer(VarPropagationLayer):

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
    
    def call(self, x):
        if self.use_cov:
            out = self._call_full_cov(x)
        else:
            out = self._call_diag_cov(x)
        return out
    
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