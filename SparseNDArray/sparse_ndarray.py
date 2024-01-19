"""
This file is the main entry point for the sparse backend. We define a
SparseBackendDevice class that wraps the main implementation backend modules
(i.e. python, cpu, cuda). Note that the SparseNDArray class is now a glorified
wrapper for the SparseBackendDevice, because we no longer deal with shape,
strides, offsets, etc. when working with sparse arrays. Instead, we only need to
keep track of the device and the underlying sparse array handle.
"""

import operator
import math
from functools import reduce
import numpy as np
from . import sparse_ndarray_backend_py
from . import sparse_ndarray_backend_cpu
import time

# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class SparseBackendDevice:
    """
    SparseBackendDevice wraps the underlying implementation modules in python,
    cpu, and cuda. Most functions remain the same, but some functions are no
    longer supported for sparse arrays (e.g. randn, rand).
    """

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # TODO: Support sparse random arrays
        raise NotImplementedError("randn not supported for sparse arrays")

    def rand(self, *shape, dtype="float32"):
        # TODO: Support sparse random arrays
        raise NotImplementedError("rand not supported for sparse arrays")

    def one_hot(self, n, i, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return SparseNDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return SparseNDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return SparseNDArray.make(shape, default_val=fill_value, device=self)

    # This method helps support interoperability with the dense arrays
    def ops(self, op: str):
        """Return the elementwise and scalar versions of an operation"""
        assert self.enabled()
        match op:
            case "+": return (self.ewise_add, self.scalar_add)
            case "-": return (self.ewise_sub, self.scalar_sub)
            case "*": return (self.ewise_mul, self.scalar_mul)
            case "/": return (self.ewise_div, self.scalar_div)
            case "max": return (self.ewise_maximum, self.scalar_maximum)
            case "==": return (self.ewise_eq, self.scalar_eq)
            case "!=": return (self.ewise_ne, self.scalar_ne)
            case ">=": return (self.ewise_ge, self.scalar_ge)
            case "<=": return (self.ewise_le, self.scalar_le)
            case ">": return (self.ewise_gt, self.scalar_gt)
            case "<": return (self.ewise_lt, self.scalar_lt)
            case _:
                raise ValueError(f"Unknown operation {op}")


def sparse_cpu():
    """Return cpu device, currently incomplete"""
    return SparseBackendDevice("cpu", sparse_ndarray_backend_cpu)


def sparse_py():
    """Return py device"""
    return SparseBackendDevice("python", sparse_ndarray_backend_py)


def sparse_cuda():
    """Return cuda device"""
    try:
        from . import sparse_ndarray_backend_cuda
        return SparseBackendDevice("cuda", sparse_ndarray_backend_cuda)
    except ImportError:
        return SparseBackendDevice("cuda", None)


def sparse_default_device():
    """Currently we have a fully working python backend"""
    return sparse_py()


def sparse_all_devices():
    """return a list of all available devices"""
    return [sparse_py(), sparse_cpu()]


class SparseNDArray:
    """
    A generic sparse ND array class that may be backed by any of the available
    sparse backends. i.e. a Numpy backend, a native CPU backend, or a GPU
    backend.

    Unlike dense arrays, where we keep track of shape, strides, offsets, for
    much high level logic (e.g. slicing, broadcasting, etc.), we only need to
    keep track of the device and the underlying sparse array handle.

    This means SparseNDArray is mostly a wrapper for the backend
    implementations, as it no longer contains much of the logic for working with
    contiguous arrays. It also supports interoperability with the dense arrays,
    as well as a toDense() method. Note that any operations with dense arrays
    will automatically promote the sparse array to a dense array.
    """

    def __init__(self, other, device=None):
        """
        We can initialize by copying from another sparse array, converting from
        a dense numpy array, or from a dictionary containing the shape, and
        optionally the default value, indices, and values.
        """
        if isinstance(other, SparseNDArray):
            # copy constructor
            if device is None:
                device = other._device
            self._init(other.to(device) + 0.0)

        elif isinstance(other, dict):
            # initialize from dictionary of shape, default_val, indices, values
            # i.e. some other library's sparse array format
            assert "shape" in other
            device = device if device is not None else sparse_default_device()
            array = self.make(other["shape"], device=device)
            array.device.from_numpy(
                other["shape"],
                other.get("default_val", 0.0),
                other.get("indices", []),
                other.get("values", []),
                array._handle,
            )
            self._init(array)

        else:
            # initialize from dense array by converting to numpy first
            if not isinstance(other, np.ndarray):
                other = np.array(other).astype(np.float32)
            else:
                other = other.astype(np.float32)

            device = device if device is not None else sparse_default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(other, array._handle)
            self._init(array)

    def _init(self, other):
        """Initialize from another SparseNDArray"""
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(*args, **kwargs):
        raise NotImplementedError("sparse arrays do not have strides")

    @staticmethod
    def make(shape, *args, default_val=0, device=None, **kwargs):
        """Create a new sparse array with the given shape and default value."""
        if device is None:
            device = sparse_default_device()

        array = SparseNDArray.__new__(SparseNDArray)
        array._device = device
        array._handle = device.SparseArray(shape, default_val)
        return array

    @property
    def shape(self):
        # access underlying handle to get shape
        return tuple(self._handle.get_shape())

    @property
    def strides(self):
        raise NotImplementedError("sparse arrays do not have strides")

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only supporting float32 for now
        return "float32"

    @property
    def ndim(self):
        # access underlying handle to get ndim
        return self._handle.ndim

    @property
    def size(self):
        """
        Return the total number of elements in the array. To note here, although
        the backend implementation also has a size property, this is the number
        of non-default values in the array, not the total number of elements,
        which is what is expected from the size property.
        """
        return prod(self._handle.get_shape())

    def __repr__(self) -> str:
        return "SparseNDArray(" + self.numpy().__str__() + f", device={self._device})"

    def __str__(self) -> str:
        return self.numpy().__str__()


    #########################################
    ### ARRAY MANIPULATION AND CONVERSION ###
    #########################################
    def fill(self, value):
        """
        Fill (in place) with a constant value Note that for sparse arrays, this
        will overwrite all non-default values as well
        """
        self._handle.fill(value)

    def to(self, device):
        """
        Convert between devices, using to/from numpy calls as the unifying
        bridge.
        """
        if device == self._device:
            return self
        else:
            return SparseNDArray(self.numpy(), device=device)

    def numpy(self):
        """Convert to a numpy array"""
        return self._device.to_numpy(self._handle)

    def toDense(self):
        """Convert to a dense numpy array"""
        return self._device.to_numpy(self._handle)

    def is_compact(self):
        """SparseNDArray is always compact (no strides)"""
        return True

    def compact(self):
        """SparseNDArray is always compact (no strides)"""
        return self

    def as_strided(self, shape, strides):
        raise NotImplementedError("sparse arrays do not have strides")

    @property
    def flat(self):
        return self.reshape((self.size,))


    ###############################
    ### ARRAY DIMS MANIPULATION ###
    ###############################

    # Note that in all of the following manipulation methods, we originally were
    # able to perform the operations without copying memory by simply using
    # stride tricks. For sparse arrays, this is not possible, so all of these
    # involve copying memory.

    def reshape(self, new_shape):
        """
        Reshape the array into the new shape. Unlike dense arrays, we cannot
        simply change the strides here, so we move the main logic to the
        backend.

        Raises:
            ValueError: If the new shape is not compatible with the current
            shape.

        Args:
            new_shape (tuple): The new shape of the array.

        Returns:
            SparseNDArray: The reshaped array.
        """
        if prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape {self.shape} array into shape {new_shape}")
        out = SparseNDArray.make(new_shape, device=self._device)
        self._device.reshape(self._handle, new_shape, out._handle)
        return out

    def permute(self, new_axes):
        """
        Permute the order of the dimensions. Again we move the main logic to the
        backend.

        Args:
            new_axes (tuple): The new order of the dimensions.

        Returns:
            SparseNDArray: The permuted array.
        """
        assert(len(new_axes) == self.ndim)
        new_shape = tuple((self.shape[i] for i in new_axes))
        out = SparseNDArray.make(new_shape, device=self._device)
        self._device.permute(self._handle, new_axes, out._handle)
        return out

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape. Again we move the main logic to the
        backend.
        """
        assert len(new_shape) == self.ndim
        out = SparseNDArray.make(new_shape, device=self._device)
        self._device.broadcast_to(self._handle, new_shape, out._handle)
        return out

    ############################
    ### GET AND SET ELEMENTS ###
    ############################
    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)


    def __getitem__(self, idxs):
        """
        Access elements of sparse array at given indices/slices. Although the
        main logic is in the backend (and we need to copy memory again), we do
        some pre-processing to convert everything to slices with explicit
        start/stop/step.
        """
        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )

        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = tuple((math.ceil((s.stop - s.start)/s.step) for s in idxs))
        out = SparseNDArray.make(new_shape, device=self._device)
        idxs = tuple([[s.start, s.stop, s.step] for s in idxs])
        self._device.getitem(self._handle, idxs, out._handle)
        return out


    def __setitem__(self, idxs, other):
        """
        Set elements of sparse array at given indices/slices. The main logic is
        in the backend and we do the same preprocessing of indices/slices as in
        __getitem__.

        Note, if other is a dense array, this function returns a dense array.
        """
        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        if isinstance(other, SparseNDArray):
            idxs = tuple([[s.start, s.stop, s.step] for s in idxs])
            self._device.ewise_setitem(self._handle, idxs, other._handle)
        elif isinstance(other, np.ndarray):
            out = self.toDense()
            out[idxs] = other
            return out
        else:
            idxs = tuple([[s.start, s.stop, s.step] for s in idxs])
            self._device.scalar_setitem(self._handle, idxs, other)

    #############################
    ### ELEMENTWISE FUNCTIONS ###
    #############################
    def __pow__(self, other):
        out = SparseNDArray.make(self.shape, device=self._device)
        self._device.scalar_power(self._handle, other, out._handle)
        return out

    def __neg__(self):
        out = SparseNDArray.make(self.shape, device=self._device)
        self._device.ewise_neg(self._handle, out._handle)
        return out

    def log(self):
        out = SparseNDArray.make(self.shape, device=self._device)
        self._device.ewise_log(self._handle, out._handle)
        return out

    def exp(self):
        out = SparseNDArray.make(self.shape, device=self._device)
        self._device.ewise_exp(self._handle, out._handle)
        return out

    def tanh(self):
        out = SparseNDArray.make(self.shape, device=self._device)
        start = time.time()
        self._device.ewise_tanh(self._handle, out._handle)
        end = time.time()
        print("tanh time: ", end - start)
        return out

    #######################
    ### MATH OPERATIONS ###
    #######################
    def ewise_or_scalar(self, other, op):
        """
        General wrapper for ewise or scalar operations.
        """
        ewise_func, scalar_func = self._device.ops(op)
        out = SparseNDArray.make(self.shape, device=self._device)
        if isinstance(other, SparseNDArray):
            # element wise operation
            assert self.shape == other.shape
            ewise_func(self._handle, other._handle, out._handle)

        elif isinstance(other, np.ndarray):
            # ewise operation with dense array, promote sparse array to dense using toDense
            # assert self.shape == other.shape
            raise NotImplementedError("support for ewise with numpy arrays not implemented")
        else:
            # assume scalar operation
            scalar_func(self._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(other, "+")

    __radd__ = __add__

    def __sub__(self, other):
        return self.ewise_or_scalar(other, "-")

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(other, "*")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(other, "/")

    #############################
    ### COMPARISON OPERATIONS ###
    #############################
    def maximum(self, other):
        return self.ewise_or_scalar(other, "max")

    def __eq__(self, other):
        return self.ewise_or_scalar(other, "==")

    def __ne__(self, other):
        return self.ewise_or_scalar(other, "!=")

    def __ge__(self, other):
        return self.ewise_or_scalar(other, ">=")

    def __le__(self, other):
        return self.ewise_or_scalar(other, "<=")

    def __gt__(self, other):
        return self.ewise_or_scalar(other, ">")

    def __lt__(self, other):
        return self.ewise_or_scalar(other, "<")

    #############################
    ### MATRIX MULTIPLICATION ###
    #############################
    def __matmul__(self, other):
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        if isinstance(other, SparseNDArray):
            # sparse x sparse
            out = SparseNDArray.make((self.shape[0], other.shape[1]), device=self._device)
            self._device.matmul(self._handle, other._handle, out._handle)
            return out

        elif isinstance(other, np.ndarray):
            # sparse x dense
            return self.toDense() @ other

    ############################
    ### REDUCTION OPERATIONS ###
    ############################
    def reduce_out(self, axis, keepdims=False):
        """
        This method is used to create the output array for a reduction operation.
        We can simply create a new array with the appropriate shape.
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            out = SparseNDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            out = SparseNDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return out, axis # return axis to ensure if it is a tuple it is converted to an int

    def sum(self, axis=None, keepdims=False):
        out, axis = self.reduce_out(axis, keepdims) # get output array
        if axis is not None:
            self._device.reduce_sum(self._handle, axis, keepdims, out._handle)
        else:
        # For our sparse array representation, we have to be careful when axis is None
        # due to the presence of non-zero default values. We need to sum over all axes
        # in reverse order in this case due to the implementation of device.reduce_sum.
        # The default value is included in the sum, and updated to account for empty axes.
            for i in range(self.ndim-1, -1, -1):
                self._device.reduce_sum(self._handle, i, False, out._handle)
        return out

    def max(self, axis=None, keepdims=False):
        out, axis = self.reduce_out(axis, keepdims) # get output array
        if axis is not None:
            self._device.reduce_max(self._handle, axis, keepdims, out._handle)
        else:
            # As above, we again have to be careful here when axis is None
            for i in range(self.ndim-1, -1, -1):
                self._device.reduce_max(self._handle, i, False, out._handle)
        return out


    ##############################
    ### OTHER ARRAY OPERATIONS ###
    ##############################
    def flip(self, axes):
        raise NotImplementedError()

    def pad(self, axes):
        raise NotImplementedError()


###########################
### CONVENIENCE METHODS ###
###########################

def array(a, dtype="float32", device=None):
    """Convenience method for creating sparse arrays"""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return SparseNDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else sparse_default_device()
    return device.empty(shape, dtype=dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else sparse_default_device()
    return device.full(shape, fill_value, dtype=dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis, keepdims)


def flip(a, axes):
    return a.flip(axes)
