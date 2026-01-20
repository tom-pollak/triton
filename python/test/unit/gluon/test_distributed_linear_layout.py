import pytest
import triton.experimental.gluon.language as gl


def test_to_ll_roundtrip():
    layout = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0]],
        lane_bases=[[0, 1], [0, 2]],
        warp_bases=[[4, 0]],
        block_bases=[],
        shape=[8, 4],
    )
    ll = layout.to_linear_layout()
    assert ll.get_in_dim_names() == ["register", "lane", "warp"]
    assert ll.get_out_dim_names() == ["dim0", "dim1"]
    out_dims = dict(ll.out_dims)
    assert out_dims["dim0"] == 8
    assert out_dims["dim1"] == 4

    back = gl.DistributedLinearLayout.from_linear_layout(ll)
    assert back.reg_bases == layout.reg_bases
    assert back.lane_bases == layout.lane_bases
    assert back.warp_bases == layout.warp_bases
    assert back.block_bases == layout.block_bases
    assert back.shape == layout.shape


def test_swizzle_tile():
    swizzle_4x4 = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1, 1], [2, 2]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[4, 4],
    )
    swizzle_64x8 = swizzle_4x4.tile([64, 8])
    assert swizzle_64x8.shape == [64, 8]
    assert swizzle_64x8.lane_bases == [[1, 1], [2, 2]]
    assert swizzle_64x8.warp_bases == [[0, 1], [0, 2]]
    assert swizzle_64x8.reg_bases == [[4, 0], [8, 0], [16, 0], [32, 0], [0, 4]]


def test_mul_operator():
    layout_dim0 = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0]],
        lane_bases=[],
        warp_bases=[],
        block_bases=[],
        shape=[16, 1],
    )
    layout_dim1 = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8]],
        warp_bases=[],
        block_bases=[],
        shape=[1, 16],
    )
    product = layout_dim0 * layout_dim1
    assert product.shape == [16, 16]
    assert product.reg_bases == layout_dim0.reg_bases
    assert product.lane_bases == layout_dim1.lane_bases


def test_invert():
    layout = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0]],
        lane_bases=[[0, 1], [0, 2]],
        warp_bases=[],
        block_bases=[],
        shape=[4, 4],
    )
    inverted = layout.invert()
    assert inverted.get_in_dim_names() == ["dim0", "dim1"]
    assert inverted.get_out_dim_names() == ["register", "lane"]
    assert inverted.get_matrix_view() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_3d_layout():
    layout = gl.DistributedLinearLayout(
        reg_bases=[[1, 0, 0], [2, 0, 0]],
        lane_bases=[[0, 1, 0], [0, 2, 0]],
        warp_bases=[[0, 0, 1], [0, 0, 2]],
        block_bases=[],
        shape=[4, 4, 4],
    )
    ll = layout.to_linear_layout()
    assert ll.get_out_dim_names() == ["dim0", "dim1", "dim2"]
    out_dims = dict(ll.out_dims)
    assert out_dims["dim0"] == 4
    assert out_dims["dim1"] == 4
    assert out_dims["dim2"] == 4

    back = gl.DistributedLinearLayout.from_linear_layout(ll)
    assert back.shape == layout.shape
    assert back.reg_bases == layout.reg_bases
    assert back.lane_bases == layout.lane_bases
    assert back.warp_bases == layout.warp_bases
    assert back.block_bases == layout.block_bases

    tiled = layout.tile([8, 8, 8])
    assert tiled.shape == [8, 8, 8]
    assert tiled.reg_bases == [[1, 0, 0], [2, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]]


def test_tile_assertions():
    layout = gl.DistributedLinearLayout(
        reg_bases=[[1, 0]],
        lane_bases=[[0, 1]],
        warp_bases=[],
        block_bases=[],
        shape=[2, 2],
    )
    with pytest.raises(AssertionError, match=r"layout rank=2 must match rank of new_shape=3"):
        layout.tile([4, 4, 4])
    with pytest.raises(AssertionError, match=r"new_shape\[0\]=3 must be >= and divisible by shape\[0\]=2"):
        layout.tile([3, 2])
    with pytest.raises(AssertionError, match=r"new_shape\[0\]=1 must be >= and divisible by shape\[0\]=2"):
        layout.tile([1, 2])
