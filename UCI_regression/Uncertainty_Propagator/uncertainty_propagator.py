import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Concatenate, Input

from .layer_link import *
from .utils.low_rank_approximation_utils import MergeLRPropagationLayer


class UncertaintyPropagator():
    """
    Class for propagating variance through a neural network
    
    With build_model one can create models for efficient mc prediction and approximate variance propagation  
    
    Properties:
        model: keras.models.Model
        mc_samples: int, number of mc samples used for a forward propagation with mc sampling
        variance_aggregation: function for aggregating variance (e.g. in classification task over classes)
        _mc_mode: bool, use monte-carlo sampling or not in predict method
    """

    def __init__(self,
                 model,
                 mc_samples: int=10,
                 use_cov: bool=False,
                 low_rank_cov: int=None,
                 variance_aggregation=lambda x: x,
                 **kwargs):

        self.model = model
        self.mc_samples = mc_samples
        self._mc_mode = True
        self.variance_aggregation = variance_aggregation
        self.use_cov = use_cov
        self.low_rank_cov = low_rank_cov

        params = {
            'low_rank_cov': self.low_rank_cov
        }

        # build model
        self._build_model(**{
            **params,
            **kwargs
        })

    def _build_model(self, **kwargs):
        """
        Builds inference models for mc sampling and approximate variance propagation
        
        args:
            use_cov: bool, use full covariance or not (not implemented for convolutional architectures)
            kwargs:
                exact: bool, (NOT FULLY IMPLEMENTED, only ReLIU) use exact variance propagation through non-linearities
        returns:
            None
        """

        layer_dict = self.get_layer_inp_mapping()
        var_dict = {
            list(layer_dict.keys())[0]: None
        }
        visited_nodes = []
        
        var_out, var_dict = self.get_var_recursively(layer_dict,
                                           var_dict, list(layer_dict.keys())[0], 
                                           [], 
                                           visited_nodes, 
                                           **kwargs)
        ###################################
        ### TO-DO: fix returning None Types
        var_out = [var for var in var_out if var is not None] # hack
        ###################################
        self.out_nodes_with_var = []
        out_nodes_names = [node.name for node in self.model.outputs]
        for key in var_dict.keys():
            if var_dict[key] is not None and key in out_nodes_names:
                self.out_nodes_with_var.append(key)
        self.var_dict = var_dict
        
        ###############################################
        ### TO-DO: implement mc graph for multipath ###
        ###############################################

        # build output names
        # approximation:
        self.out_names_approx, self.out_names_mc = [], []
        for o in self.model.outputs:
            name = o.name
            self.out_names_approx.append('approx_' + name)
            self.out_names_mc.append('mc_' + name)
            if name in list(self.var_dict.keys()) and self.var_dict[name] is not None:
                self.out_names_approx.append('approx_var_' + name)
                self.out_names_mc.append('mc_var_' + name)

        if self.low_rank_cov is not None:
            var_out = [MergeLRPropagationLayer()(v) for v in var_out]

        # create the variance propagation model
        self.unc_model = Model(self.model.inputs, 
                               self.model.outputs + var_out)

    def get_var_recursively(self, 
                            layer_dict: dict, 
                            var_dict: dict, 
                            node_name: str, 
                            result: list, 
                            visited_nodes: list, 
                            **kwargs):
        
        layer_list = layer_dict[node_name]
        visited_nodes.append(node_name)

        for layer in layer_list:

            # get variance of note
            var = var_dict[node_name]

            layer_type = layer.__class__.__name__

            # check if current layer is a noise layer
            if layer_type in noise_layers.keys():
                init = var is None
                var = noise_layers[layer_type](layer,
                                               initial_noise=init,
                                               use_cov=self.use_cov,
                                               **kwargs)(layer.input if init else var)
            # otherwise we propagate the variance (if there has been a prior noise injection)
            elif var is not None:
                var = self._propagate_var(var, layer, self.use_cov, **kwargs)

            next_node = layer.output.name
            var_dict[next_node] = var

            if next_node not in visited_nodes:
                if next_node not in layer_dict.keys():
                    result.append(var)
                else:
                    result, var_dict = self.get_var_recursively(layer_dict,
                                                                var_dict,
                                                                next_node,
                                                                result,
                                                                visited_nodes,
                                                                **kwargs)

        return result, var_dict

    def get_layer_inp_mapping(self):

        layer_dict = dict()
        for layer in self.model.layers:

            # in case layer has several inputs we need to iterate over it
            if isinstance(layer.input, list):
                inputs = layer.input
            else:
                inputs = [layer.input]

            for inp in inputs:
                if inp.name not in layer_dict.keys():
                    layer_dict[inp.name] = [layer]
                else:
                    layer_dict[inp.name] += [layer]

        return layer_dict

    def _propagate_var(self, 
                       var, 
                       layer, 
                       use_cov, 
                       **kwargs):
        """
        propagates variance var through layer
        
        args:
            var: tf.Tensor, variance prior to layer
            layer: keras.layers.Layer, current layer
            use_cov: bool, use full covariance or not
        returns:
            tf.Tensor with variance after layer
        """
        
        layer_type = layer.__class__.__name__
        
        # if layer is a pooling layer
        if layer_type in pooling_layers.keys():
            return pooling_layers[layer_type](layer, use_cov=use_cov, **kwargs)(var)
        
        # for affine layer we potentially need to propagate through affine layer and its activation function
        elif layer_type in affine_layers.keys():
            activation_name = layer.activation.__name__
            var = affine_layers[layer_type](layer, use_cov=use_cov, **kwargs)(var)
            
            # we need the mean input to the activation function if the activation function is not the identity
            if activation_name != 'linear':
                inputs = PreActivationLayer(layer)(layer.input)
            else:
                inputs = None
            
            # build activation layer from mean input and propagate variance
            return activation_layers[activation_name](inputs=inputs, use_cov=use_cov, **kwargs)(var)
        
        # layer is only and activation function 
        elif layer_type == 'Activation':
            activation_name = layer.activation.__name__
            return activation_layers[activation_name](inputs=layer.get_input_at(0), use_cov=use_cov, **kwargs)(var)

        # correct input for lambda layer
        elif layer_type == 'Lambda':

            print('Warning: Layer not implemented:', layer_type)

            # if there are several inputs find correct slot for variance
            if isinstance(layer.input, list):
                input_list = []
                for idx in range(len(layer.input)):
                    inp = layer.input[idx]
                    if inp.get_shape().as_list() == var.get_shape().as_list():
                        input_list.append(var)
                    else:
                        input_list.append(inp)

                return Lambda(lambda x: x, name='VarPropagation_' + layer.name)(layer(input_list))

            else:
                return layer(var)

        # if variance propagation layer not implemented for this layer type: return warning
        # and apply layer directly to variance
        else:
            print('Warning: Layer not implemented:', layer_type)
            if use_cov:
                raise NotImplementedError('Layer type not implemented!')
            return layer(var)
    
    def mc_prediction(self, 
                      X, 
                      return_runtime: bool=False):
        """
        Performs prediction based on MC sampling
        
        args:
            X: np.array, input
            return_runtime: bool, return runtime or not
        returns:
            prediction and standard deviation (and runtime if return_runtime)
        """
        
        # get prediction function
        f = K.function(self.model.inputs + [K.learning_phase()],
               self.model.outputs)
        
        # create list of inputs for single input
        if not isinstance(X, list):
            X = [X]
        
        # predict with mc sampling model
        start = time.time()
        result = [f(X + [1]) for _ in range(self.mc_samples)]
        rt = time.time() - start

        # compute std and mean
        means = []
        variances = []
        for i in range(len(self.model.outputs)):
            ri = [r[i] for r in result]
            means.append(np.mean(ri, 0))
            variances.append(np.var(ri, 0))
        variances = [self.variance_aggregation(v) for v in variances]

        # filter outputs that have zero variance
        result_var = dict()
        for i in range(len(variances)):
            name = self.model.outputs[i].name
            if name in self.out_nodes_with_var:
                result_var['mc_var_' + name] = variances[i]
        
        if return_runtime:
            return means, result_var, rt
        else: 
            return means, result_var
        
    def approx_prediction(self, 
                          X, 
                          return_runtime: bool=False):
        """
        Performs prediction based on approximate variance propagation
        
        args:
            X: np.array, input
            return_runtime: bool, return runtime or not
        returns:
            prediction and standard deviation (and runtime if return_runtime)
        """
        
        # predict with variance propagation model
        start = time.time()
        result = self.unc_model.predict(X)
        rt = time.time() - start

        # split prediction and variances
        split_idx = len(self.unc_model.outputs) - len(self.out_nodes_with_var)
        preds = result[:split_idx]
        vars = result[split_idx:]

        if self.low_rank_cov is None:
            result_var = dict()
            for name in self.out_nodes_with_var:

                # get output variance name and subsequently its index
                var_node_name = self._get_var_tensor_name(name)
                for i in range(split_idx):
                    if self.unc_model.outputs[split_idx + i].name == var_node_name:
                        break

                result_var['approx_var_' + name] = vars[i]
        else:
            result_var = vars

        if return_runtime:
            return preds, result_var, rt
        else:
            return preds, result_var

    def predict(self, 
                X, 
                return_runtime: bool=False):
        """
        predicts prediction and standard deviation. Used to be able use UncertaintyPropagator interchangable 
        with keras.models.Model at inference.
        Used model for inference determined via private property _mc_mode
        
        args:
            X: np.array, input
            return_runtime: bool, return runtime or not
        returns:
            prediction and standard deviation (and runtime if return_runtime)
        """
        
        if self._mc_mode:
            return self.mc_prediction(X, return_runtime=return_runtime)
        else:
            return self.approx_prediction(X, return_runtime=return_runtime)

    def get_output_nodes_names(self, approx: bool=True):
        """
        Returns dictionary identifying output node with readable string indicating its meaning

        args:
            approx_dict: bool, return dict for approximated variance propagation or mc sampling
        returns:
            dictionary
        """

        if approx:
            return self.out_names_approx
        else:
            return self.out_names_mc

    def set_mc_mode(self, mc_mode: bool):
        """
        Sets private property _mc_mode
        """
        self._mc_mode = mc_mode

    def _get_var_tensor_name(self, out_node_name: str):
        """
        Returns corresponding variance tensor name or None
        """

        if out_node_name in list(self.var_dict.keys()):
            return self.var_dict[out_node_name].name
        else:
            return None
