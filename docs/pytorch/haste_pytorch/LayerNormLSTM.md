<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="haste_pytorch.LayerNormLSTM" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_module"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="buffers"/>
<meta itemprop="property" content="children"/>
<meta itemprop="property" content="cpu"/>
<meta itemprop="property" content="cuda"/>
<meta itemprop="property" content="double"/>
<meta itemprop="property" content="eval"/>
<meta itemprop="property" content="extra_repr"/>
<meta itemprop="property" content="float"/>
<meta itemprop="property" content="forward"/>
<meta itemprop="property" content="half"/>
<meta itemprop="property" content="load_state_dict"/>
<meta itemprop="property" content="modules"/>
<meta itemprop="property" content="named_buffers"/>
<meta itemprop="property" content="named_children"/>
<meta itemprop="property" content="named_modules"/>
<meta itemprop="property" content="named_parameters"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="register_backward_hook"/>
<meta itemprop="property" content="register_buffer"/>
<meta itemprop="property" content="register_forward_hook"/>
<meta itemprop="property" content="register_forward_pre_hook"/>
<meta itemprop="property" content="register_parameter"/>
<meta itemprop="property" content="requires_grad_"/>
<meta itemprop="property" content="reset_parameters"/>
<meta itemprop="property" content="share_memory"/>
<meta itemprop="property" content="state_dict"/>
<meta itemprop="property" content="to"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="type"/>
<meta itemprop="property" content="zero_grad"/>
</div>

# haste_pytorch.LayerNormLSTM

<!-- Insert buttons and diff -->


## Class `LayerNormLSTM`

Layer Normalized Long Short-Term Memory layer.



<!-- Placeholder for "Used in" -->

This LSTM layer applies layer normalization to the input, recurrent, and
output activations of a standard LSTM. The implementation is fused and
GPU-accelerated. DropConnect and Zoneout regularization are built-in, and
this layer allows setting a non-zero initial forget gate bias.

Details about the exact function this layer implements can be found at
https://github.com/lmnt-com/haste/issues/1.

See [\_\_init\_\_](#__init__) and [forward](#forward) for usage.

<h2 id="__init__"><code><a name="__init__">__init__</a></code></h2>

``` python
__init__(
    input_size,
    hidden_size,
    batch_first=False,
    forget_bias=1.0,
    dropout=0.0,
    zoneout=0.0
)
```

Initialize the parameters of the LSTM layer.


#### Arguments:


* <b>`input_size`</b>: int, the feature dimension of the input.
* <b>`hidden_size`</b>: int, the feature dimension of the output.
* <b>`batch_first`</b>: (optional) bool, if `True`, then the input and output
  tensors are provided as `(batch, seq, feature)`.
* <b>`forget_bias`</b>: (optional) float, sets the initial bias of the forget gate
  for this LSTM cell.
* <b>`dropout`</b>: (optional) float, sets the dropout rate for DropConnect
  regularization on the recurrent matrix.
* <b>`zoneout`</b>: (optional) float, sets the zoneout rate for Zoneout
  regularization.


#### Variables:


* <b>`kernel`</b>: the input projection weight matrix. Dimensions
  (input_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
  with Xavier uniform initialization.
* <b>`recurrent_kernel`</b>: the recurrent projection weight matrix. Dimensions
  (hidden_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
  with orthogonal initialization.
* <b>`bias`</b>: the projection bias vector. Dimensions (hidden_size * 4) with
  `i,g,f,o` gate layout. The forget gate biases are initialized to
  `forget_bias` and the rest are zeros.
* <b>`gamma`</b>: the input and recurrent normalization gain. Dimensions
  (2, hidden_size * 4) with `gamma[0]` specifying the input gain and
  `gamma[1]` specifying the recurrent gain. Initialized to ones.
* <b>`gamma_h`</b>: the output normalization gain. Dimensions (hidden_size).
  Initialized to ones.
* <b>`beta_h`</b>: the output normalization bias. Dimensions (hidden_size).
  Initialized to zeros.



## Methods

<h3 id="__call__"><code><a name="__call__">__call__</a></code></h3>

``` python
__call__(
    *input,
    **kwargs
)
```

Call self as a function.


<h3 id="add_module"><code><a name="add_module">add_module</a></code></h3>

``` python
add_module(
    name,
    module
)
```

Adds a child module to the current module.

The module can be accessed as an attribute using the given name.

#### Args:

name (string): name of the child module. The child module can be
    accessed from this module using the given name
module (Module): child module to be added to the module.


<h3 id="apply"><code><a name="apply">apply</a></code></h3>

``` python
apply(fn)
```

Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
as well as self. Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

#### Args:

fn (:class:`Module` -> None): function to be applied to each submodule



#### Returns:


* <b>`Module`</b>: self

Example::

    ```
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.data.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    ```

<h3 id="buffers"><code><a name="buffers">buffers</a></code></h3>

``` python
buffers(recurse=True)
```

Returns an iterator over module buffers.


#### Args:

recurse (bool): if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



#### Yields:


* <b>`torch.Tensor`</b>: module buffer

Example::

    ```
    >>> for buf in model.buffers():
    >>>     print(type(buf.data), buf.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
    ```

<h3 id="children"><code><a name="children">children</a></code></h3>

``` python
children()
```

Returns an iterator over immediate children modules.


#### Yields:


* <b>`Module`</b>: a child module

<h3 id="cpu"><code><a name="cpu">cpu</a></code></h3>

``` python
cpu()
```

Moves all model parameters and buffers to the CPU.


#### Returns:


* <b>`Module`</b>: self

<h3 id="cuda"><code><a name="cuda">cuda</a></code></h3>

``` python
cuda(device=None)
```

Moves all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on GPU while being optimized.

#### Arguments:

device (int, optional): if specified, all parameters will be
    copied to that device



#### Returns:


* <b>`Module`</b>: self

<h3 id="double"><code><a name="double">double</a></code></h3>

``` python
double()
```

Casts all floating point parameters and buffers to ``double`` datatype.


#### Returns:


* <b>`Module`</b>: self

<h3 id="eval"><code><a name="eval">eval</a></code></h3>

``` python
eval()
```

Sets the module in evaluation mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

#### Returns:


* <b>`Module`</b>: self

<h3 id="extra_repr"><code><a name="extra_repr">extra_repr</a></code></h3>

``` python
extra_repr()
```

Set the extra representation of the module

To print customized extra information, you should reimplement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

<h3 id="float"><code><a name="float">float</a></code></h3>

``` python
float()
```

Casts all floating point parameters and buffers to float datatype.


#### Returns:


* <b>`Module`</b>: self

<h3 id="forward"><code><a name="forward">forward</a></code></h3>

``` python
forward(
    input,
    state=None,
    lengths=None
)
```

Runs a forward pass of the LSTM layer.


#### Arguments:


* <b>`input`</b>: Tensor, a batch of input sequences to pass through the LSTM.
  Dimensions (seq_len, batch_size, input_size) if `batch_first` is
  `False`, otherwise (batch_size, seq_len, input_size).
* <b>`lengths`</b>: (optional) Tensor, list of sequence lengths for each batch
  element. Dimension (batch_size). This argument may be omitted if
  all batch elements are unpadded and have the same sequence length.


#### Returns:


* <b>`output`</b>: Tensor, the output of the LSTM layer. Dimensions
  (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
  or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
  that if `lengths` was specified, the `output` tensor will not be
  masked. It's the caller's responsibility to either not use the invalid
  entries or to mask them out before using them.
* <b>`(h_n, c_n)`</b>: the hidden and cell states, respectively, for the last
  sequence item. Dimensions (1, batch_size, hidden_size).

<h3 id="half"><code><a name="half">half</a></code></h3>

``` python
half()
```

Casts all floating point parameters and buffers to ``half`` datatype.


#### Returns:


* <b>`Module`</b>: self

<h3 id="load_state_dict"><code><a name="load_state_dict">load_state_dict</a></code></h3>

``` python
load_state_dict(
    state_dict,
    strict=True
)
```

Copies parameters and buffers from :attr:`state_dict` into
this module and its descendants. If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

#### Arguments:

state_dict (dict): a dict containing parameters and
    persistent buffers.
strict (bool, optional): whether to strictly enforce that the keys
    in :attr:`state_dict` match the keys returned by this module's
    :meth:`~torch.nn.Module.state_dict` function. Default: ``True``



#### Returns:

``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
    * **missing_keys** is a list of str containing the missing keys
    * **unexpected_keys** is a list of str containing the unexpected keys


<h3 id="modules"><code><a name="modules">modules</a></code></h3>

``` python
modules()
```

Returns an iterator over all modules in the network.


#### Yields:


* <b>`Module`</b>: a module in the network


#### Note:

Duplicate modules are returned only once. In the following
example, ``l`` will be returned only once.


Example::

    ```
    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
            print(idx, '->', m)
    ```

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

<h3 id="named_buffers"><code><a name="named_buffers">named_buffers</a></code></h3>

``` python
named_buffers(
    prefix='',
    recurse=True
)
```

Returns an iterator over module buffers, yielding both the
name of the buffer as well as the buffer itself.

#### Args:

prefix (str): prefix to prepend to all buffer names.
recurse (bool): if True, then yields buffers of this module
    and all submodules. Otherwise, yields only buffers that
    are direct members of this module.



#### Yields:


* <b>`(string, torch.Tensor)`</b>: Tuple containing the name and buffer

Example::

    ```
    >>> for name, buf in self.named_buffers():
    >>>    if name in ['running_var']:
    >>>        print(buf.size())
    ```

<h3 id="named_children"><code><a name="named_children">named_children</a></code></h3>

``` python
named_children()
```

Returns an iterator over immediate children modules, yielding both
the name of the module as well as the module itself.

#### Yields:


* <b>`(string, Module)`</b>: Tuple containing a name and child module

Example::

    ```
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)
    ```

<h3 id="named_modules"><code><a name="named_modules">named_modules</a></code></h3>

``` python
named_modules(
    memo=None,
    prefix=''
)
```

Returns an iterator over all modules in the network, yielding
both the name of the module as well as the module itself.

#### Yields:


* <b>`(string, Module)`</b>: Tuple of name and module


#### Note:

Duplicate modules are returned only once. In the following
example, ``l`` will be returned only once.


Example::

    ```
    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
            print(idx, '->', m)
    ```

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

<h3 id="named_parameters"><code><a name="named_parameters">named_parameters</a></code></h3>

``` python
named_parameters(
    prefix='',
    recurse=True
)
```

Returns an iterator over module parameters, yielding both the
name of the parameter as well as the parameter itself.

#### Args:

prefix (str): prefix to prepend to all parameter names.
recurse (bool): if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



#### Yields:


* <b>`(string, Parameter)`</b>: Tuple containing the name and parameter

Example::

    ```
    >>> for name, param in self.named_parameters():
    >>>    if name in ['bias']:
    >>>        print(param.size())
    ```

<h3 id="parameters"><code><a name="parameters">parameters</a></code></h3>

``` python
parameters(recurse=True)
```

Returns an iterator over module parameters.

This is typically passed to an optimizer.

#### Args:

recurse (bool): if True, then yields parameters of this module
    and all submodules. Otherwise, yields only parameters that
    are direct members of this module.



#### Yields:


* <b>`Parameter`</b>: module parameter

Example::

    ```
    >>> for param in model.parameters():
    >>>     print(type(param.data), param.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
    ```

<h3 id="register_backward_hook"><code><a name="register_backward_hook">register_backward_hook</a></code></h3>

``` python
register_backward_hook(hook)
```

Registers a backward hook on the module.

The hook will be called every time the gradients with respect to module
inputs are computed. The hook should have the following signature::

    hook(module, grad_input, grad_output) -> Tensor or None

The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
module has multiple inputs or outputs. The hook should not modify its
arguments, but it can optionally return a new gradient with respect to
input that will be used in place of :attr:`grad_input` in subsequent
computations.

#### Returns:

:class:`torch.utils.hooks.RemovableHandle`:
    a handle that can be used to remove the added hook by calling
    ``handle.remove()``


.. warning ::

    The current implementation will not have the presented behavior
    for complex :class:`Module` that perform many operations.
    In some failure cases, :attr:`grad_input` and :attr:`grad_output` will only
    contain the gradients for a subset of the inputs and outputs.
    For such :class:`Module`, you should use :func:`torch.Tensor.register_hook`
    directly on a specific input or output to get the required gradients.

<h3 id="register_buffer"><code><a name="register_buffer">register_buffer</a></code></h3>

``` python
register_buffer(
    name,
    tensor
)
```

Adds a persistent buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the persistent state.

Buffers can be accessed as attributes using given names.

#### Args:

name (string): name of the buffer. The buffer can be accessed
    from this module using the given name
tensor (Tensor): buffer to be registered.


Example::

    ```
    >>> self.register_buffer('running_mean', torch.zeros(num_features))
    ```

<h3 id="register_forward_hook"><code><a name="register_forward_hook">register_forward_hook</a></code></h3>

``` python
register_forward_hook(hook)
```

Registers a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.
It should have the following signature::

    hook(module, input, output) -> None or modified output

The hook can modify the output. It can modify the input inplace but
it will not have effect on forward since this is called after
:func:`forward` is called.

#### Returns:

:class:`torch.utils.hooks.RemovableHandle`:
    a handle that can be used to remove the added hook by calling
    ``handle.remove()``


<h3 id="register_forward_pre_hook"><code><a name="register_forward_pre_hook">register_forward_pre_hook</a></code></h3>

``` python
register_forward_pre_hook(hook)
```

Registers a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.
It should have the following signature::

    hook(module, input) -> None or modified input

The hook can modify the input. User can either return a tuple or a
single modified value in the hook. We will wrap the value into a tuple
if a single value is returned(unless that value is already a tuple).

#### Returns:

:class:`torch.utils.hooks.RemovableHandle`:
    a handle that can be used to remove the added hook by calling
    ``handle.remove()``


<h3 id="register_parameter"><code><a name="register_parameter">register_parameter</a></code></h3>

``` python
register_parameter(
    name,
    param
)
```

Adds a parameter to the module.

The parameter can be accessed as an attribute using given name.

#### Args:

name (string): name of the parameter. The parameter can be accessed
    from this module using the given name
param (Parameter): parameter to be added to the module.


<h3 id="requires_grad_"><code><a name="requires_grad_">requires_grad_</a></code></h3>

``` python
requires_grad_(requires_grad=True)
```

Change if autograd should record operations on parameters in this
module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

#### Args:

requires_grad (bool): whether autograd should record operations on
                      parameters in this module. Default: ``True``.



#### Returns:


* <b>`Module`</b>: self

<h3 id="reset_parameters"><code><a name="reset_parameters">reset_parameters</a></code></h3>

``` python
reset_parameters()
```

Resets this layer's parameters to their initial values.


<h3 id="share_memory"><code><a name="share_memory">share_memory</a></code></h3>

``` python
share_memory()
```




<h3 id="state_dict"><code><a name="state_dict">state_dict</a></code></h3>

``` python
state_dict(
    destination=None,
    prefix='',
    keep_vars=False
)
```

Returns a dictionary containing a whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.

#### Returns:


* <b>`dict`</b>:     a dictionary containing a whole state of the module

Example::

    ```
    >>> module.state_dict().keys()
    ['bias', 'weight']
    ```

<h3 id="to"><code><a name="to">to</a></code></h3>

``` python
to(
    *args,
    **kwargs
)
```

Moves and/or casts the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)

.. function:: to(dtype, non_blocking=False)

.. function:: to(tensor, non_blocking=False)

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point desired :attr:`dtype` s. In addition, this method will
only cast the floating point parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

#### Args:

device (:class:`torch.device`): the desired device of the parameters
    and buffers in this module
dtype (:class:`torch.dtype`): the desired floating point type of
    the floating point parameters and buffers in this module
tensor (torch.Tensor): Tensor whose dtype and device are the desired
    dtype and device for all parameters and buffers in this module



#### Returns:


* <b>`Module`</b>: self

Example::

    ```
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)
    ```

<h3 id="train"><code><a name="train">train</a></code></h3>

``` python
train(mode=True)
```

Sets the module in training mode.

This has any effect only on certain modules. See documentations of
particular modules for details of their behaviors in training/evaluation
mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

#### Args:

mode (bool): whether to set training mode (``True``) or evaluation
             mode (``False``). Default: ``True``.



#### Returns:


* <b>`Module`</b>: self

<h3 id="type"><code><a name="type">type</a></code></h3>

``` python
type(dst_type)
```

Casts all parameters and buffers to :attr:`dst_type`.


#### Arguments:

dst_type (type or string): the desired type



#### Returns:


* <b>`Module`</b>: self

<h3 id="zero_grad"><code><a name="zero_grad">zero_grad</a></code></h3>

``` python
zero_grad()
```

Sets gradients of all model parameters to zero.




