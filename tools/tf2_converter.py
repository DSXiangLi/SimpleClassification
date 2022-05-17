"""
    完成Tensorflow 1 到Tensorflow2的相关转换
"""

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.compat.v1 import get_variable as variables
from tensorflow.python.ops import variable_scope


def layer_norm(input_tensor, name=None):
    if tf.__version__ >= '2':
        return layer_norm_v2(input_tensor, name)
    else:
        return layer_norm_v1(input_tensor, name)


def layer_norm_v1(input_tensor, name):
    """Run layer normalization on the last dimension of the tensor."""
    from tensorflow.contrib import layers as contrib_layers
    return contrib_layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_v2(input_tensor, name):
    """Run layer normalization on the last dimension of the tensor."""
    inputs = input_tensor
    begin_norm_axis = -1
    begin_params_axis = -1
    scope = name
    center = True
    scale = True
    activation_fn = None
    reuse = None
    trainable = True
    with variable_scope.variable_scope(
            scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            # beta_collections = utils.get_variable_collections(variables_collections,
            #                                                  'beta')
            beta = variables(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.zeros_initializer(),
                collections=None,
                trainable=trainable)
        if scale:
            # gamma_collections = utils.get_variable_collections(
            #    variables_collections, 'gamma')
            gamma = variables(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=None,
                trainable=trainable)
        # By default, compute the moments across all the dimensions except the one with index 0.
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        # Note that epsilon must be increased for float16 due to the limited
        # representable range.
        variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
        outputs = nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
