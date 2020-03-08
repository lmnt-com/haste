<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="haste_tf.LayerNorm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# haste_tf.LayerNorm

<!-- Insert buttons and diff -->


## Class `LayerNorm`

Layer normalization layer.



<!-- Placeholder for "Used in" -->

This class exposes a fused and GPU-accelerated implementation of layer
normalization as described by [Ba et al.](https://arxiv.org/abs/1607.06450)

<h2 id="__init__"><code><a name="__init__">__init__</a></code></h2>

``` python
__init__(name=None)
```

Initialize the parameters of the layer normalization layer.


#### Arguments:


* <b>`name`</b>: (optional) string, the name for this layer.



## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


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

<h3 id="__call__"><code><a name="__call__">__call__</a></code></h3>

``` python
__call__(x)
```

Runs the layer.


#### Arguments:


* <b>`x`</b>: Tensor, a rank R tensor.


#### Returns:


* <b>`y`</b>: Tensor, a rank R tensor with the last dimension normalized.

<h3 id="build"><code><a name="build">build</a></code></h3>

``` python
build(shape)
```

Creates the variables of the layer.

Calling this method is optional for users of the LayerNorm class. It is
called internally with the correct shape when `__call__` is invoked.

#### Arguments:


* <b>`shape`</b>: instance of `TensorShape`.

<h3 id="with_name_scope"><code><a name="with_name_scope">with_name_scope</a></code></h3>

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




