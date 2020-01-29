<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="haste_tf.LSTM" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="output_size"/>
<meta itemprop="property" content="state_size"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# haste_tf.LSTM

<!-- Insert buttons and diff -->


## Class `LSTM`

Long Short-Term Memory layer.



<!-- Placeholder for "Used in" -->

This LSTM layer offers a fused, GPU-accelerated TensorFlow op for inference
and training. Its weights and variables are compatible with `BasicLSTMCell`,
`LSTMCell`, and `LSTMBlockCell` by default, and is able to load weights
from `tf.contrib.cudnn_rnn.CudnnLSTM` when `cudnn_compat=True` is specified.

Although this implementation is comparable in performance to cuDNN's LSTM,
it offers additional options not typically found in other high-performance
implementations. DropConnect and Zoneout regularization are built-in, and
this layer allows setting a non-zero initial forget gate bias.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_units,
    direction='unidirectional',
    **kwargs
)
```

Initialize the parameters of the LSTM layer.


#### Arguments:


* <b>`num_units`</b>: int, the number of units in the LSTM cell.
* <b>`direction`</b>: string, 'unidirectional' or 'bidirectional'.
* <b>`**kwargs`</b>: Dict, keyword arguments (see below).


#### Keyword Arguments:


* <b>`kernel_initializer`</b>: (optional) the initializer to use for the input
  matrix weights. Defaults to `glorot_uniform`.
* <b>`recurrent_initializer`</b>: (optional) the initializer to use for the
  recurrent matrix weights. Defaults to `orthogonal`.
* <b>`bias_initializer`</b>: (optional) the initializer to use for both input and
  recurrent bias vectors. Defaults to `zeros` unless `forget_bias` is
  non-zero (see below).
* <b>`forget_bias`</b>: (optional) float, sets the initial weights for the forget
  gates. Defaults to 1 and overrides the `bias_initializer` unless this
  argument is set to 0.
* <b>`dropout`</b>: (optional) float, sets the dropout rate for DropConnect
  regularization on the recurrent matrix. Defaults to 0.
* <b>`zoneout`</b>: (optional) float, sets the zoneout rate for Zoneout
  regularization. Defaults to 0.
* <b>`dtype`</b>: (optional) the data type for this layer. Defaults to `tf.float32`.
* <b>`name`</b>: (optional) string, the name for this layer.
* <b>`cudnn_compat`</b>: (optional) bool, if `True`, the variables created by this
  layer are compatible with `tf.contrib.cudnn_rnn.CudnnLSTM`. Note that
  this should only be set if you're restoring variables from a cuDNN
  model. It's currently not possible to train a model with
  `cudnn_compat=True` and restore it with CudnnLSTM. Defaults to `False`.



## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="output_size"><code>output_size</code></h3>




<h3 id="state_size"><code>state_size</code></h3>




<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.


<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).


<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).




## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    inputs,
    training,
    sequence_length=None,
    time_major=False
)
```

Runs the LSTM layer.


#### Arguments:


* <b>`inputs`</b>: Tensor, a rank 3 input tensor with shape [N,T,C] if `time_major`
  is `False`, or with shape [T,N,C] if `time_major` is `True`.
* <b>`training`</b>: bool, `True` if running in training mode, `False` if running
  in inference mode.
* <b>`sequence_length`</b>: (optional) Tensor, a rank 1 tensor with shape [N] and
  dtype of `tf.int32` or `tf.int64`. This tensor specifies the unpadded
  length of each example in the input minibatch.
* <b>`time_major`</b>: (optional) bool, specifies whether `input` has shape [N,T,C]
  (`time_major=False`) or shape [T,N,C] (`time_major=True`).


#### Returns:

A pair, `(output, state)` for unidirectional layers, or a pair
`([output_fwd, output_bwd], [state_fwd, state_bwd])` for bidirectional
layers. Each state object will be an instance of `LSTMStateTuple`.


<h3 id="build"><code>build</code></h3>

``` python
build(shape)
```

Creates the variables of the layer.

Calling this method is optional for users of the LSTM class. It is called
internally with the correct shape when `__call__` is invoked.

#### Arguments:


* <b>`shape`</b>: instance of `TensorShape`.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
@classmethod
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




