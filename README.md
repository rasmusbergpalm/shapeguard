# ShapeGuard
ShapeGuard allows you to very succinctly assert the expected shapes of tensors in a dynamic, einsum inspired way

It’s easy to make bugs in ml. One particular rich source of bugs is due to the flexibility of the operators: `a*b` works whether a and b are vectors, scalar vector, vector vector, etc. Similarly `.sum()` will work regardless of the shape of your tensor. Since we're doing optimization whatever computation we end up performing, we can probably optimize it to work reasonably, even if it's not doing what we intended. So our algorithm might "work" even if we have bugs (just less well). This makes bugs super hard to discover.

The best way I’ve found to avoid bugs is to religiously check the shapes of all my tensors, all the time, so I end up spending a lot of time debugging and writing comments like `#(bs, n_samples, z_size)` all over the place.

So why not algorithmically check the shapes then? Well it gets ugly fast.

You have to add assert `foo.shape == (bs, n_samples, x_size)` everywhere, which essentially doubles your linecount and
you have to define all your dimensional sizes (bs, etc.), which might vary across train/test, batches, etc.
So I made a small helper that makes it much nicer. I call it ShapeGuard. When you import it, It adds the `sg` method to the `torch.Tensor` and `torch.distributions.Distribution`, and exposes a static `ShapeGuard` class.

You use the `sg` method like an assert:

```python
def forward(self, x, y):
    x.sg("bchw")
    y.sg("by")
```

This will verify that x has 4 dimensions, y has 2 dimensions and that x and y have the same size in the first dimension 'b'. If the assert passes, the tensor is returned. This means you can also use it inline on results of operations: 

```python
z = f(x).sg("bnz")
```

If the assert fails it produces a nice error message.

It works in the following way: the first time sg is called for an unseen shape, the size of the tensor for that shape is saved in the `ShapeGuard.shapes` global dict. Subsequent calls sees this shape in the shapes dict and asserts that the tensor is the same shape for that dimension. If e.g. your batch size changes between train and test you can call `ShapeGuard.reset("b")` to reset the "b" shape. 

I've found it works well to reset all shapes at the start of my main `nn.Module.forward` by calling `ShapeGuard.reset()`. If you want to verify an exact dimension you can pass an int as the shape e.g.

```python
def forward(self, x, y):
    x.sg(("b", 1, "h", "w"))
    y.sg("by")
```

The special shape '\*' is reserved for shapes that should not be asserted, e.g. `x.sg("*chw")` will assert all shapes except the first.
