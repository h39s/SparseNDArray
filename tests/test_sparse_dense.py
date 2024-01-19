import numpy as np
import pytest
from .. import SparseNDArray as nds

SPARSE_DEVICES = [
    nds.sparse_py(),
    nds.sparse_cpu(),
    pytest.param(
        nds.sparse_cuda(), marks=pytest.mark.skipif(not nds.sparse_cuda().enabled(), reason="No GPU")
    ),
]


""" For converting slice notation to slice objects to make some proceeding tests easier to read """


class _ShapeAndSlices(np.ndarray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))


OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "radd": lambda a, b: b + a,
    "subtract": lambda a, b: a - b,
    "rsubtract": lambda a, b: b - a,
    "equal": lambda a, b: a == b,
    "not_equal": lambda a, b: a != b,
    "greater_than": lambda a, b: a > b,
    "less_than": lambda a, b: a < b,
    "greater_than_equal": lambda a, b: a >= b,
    "less_than_equal": lambda a, b: a <= b,
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = [k for k in OPS]

ewise_shapes = [(1, 1, 1), (4, 5, 6)]


@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
def test_ewise_dense_sparse_fn(fn, shape, sparse_device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = np.array(_A)
    B = nds.array(_B, device=sparse_device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
def test_ewise_sparse_dense_fn(fn, shape, sparse_device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nds.array(_A, device=sparse_device)
    B = np.array(_B)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
def test_ewise_dense_sparse_max(shape, sparse_device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = np.array(_A)
    B = nds.array(_B, device=sparse_device)
    np.testing.assert_allclose(
        np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
def test_ewise_sparse_dense_max(shape, sparse_device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nds.array(_A, device=sparse_device)
    B = np.array(_B)
    np.testing.assert_allclose(
        np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5
    )


matmul_dims = [
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 16),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
def test_matmul_sparse_dense(m, n, p, sparse_device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = nds.array(_A, device=sparse_device)
    B = np.array(_B)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, atol=1e-3)


@pytest.mark.parametrize("sparse_device", SPARSE_DEVICES, ids=["sparse_py", "sparse_cpu", "sparse_gpu"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
def test_matmul_dense_sparse(m, n, p, sparse_device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = np.array(_A)
    B = nds.array(_B, device=sparse_device)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, atol=1e-3)



if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")