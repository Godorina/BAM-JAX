from typing import Optional, Union, Tuple, Dict, Callable, Any

import jax
import jax.numpy as jnp
from flax import nnx

import e3nn_jax as e3nn
from e3nn_jax._src.utils.dtype import get_pytree_dtype

from e3nn_jax import (
    FunctionalLinear,
)
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct

def _get_gradient_normalization(
    gradient_normalization: Optional[Union[float, str]]
) -> float:
    """Get the gradient normalization from the config or from the argument."""
    if gradient_normalization is None:
        gradient_normalization = e3nn.config("gradient_normalization")
        # 기본값으로는 e3nn.config("gradient_normalization") = "element" 이다.

    if isinstance(gradient_normalization, str):
        return {"element": 0.0, "path": 1.0}[gradient_normalization]
    return gradient_normalization

def _linear_vanilla(
    input: e3nn.IrrepsArray,
    linear: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
) -> e3nn.IrrepsArray:
    """Vanilla linear layer."""
    w = [
        get_parameter(
            (
                f"b[{ins.i_out}] {linear.irreps_out[ins.i_out]}"
                if ins.i_in == -1
                else f"w[{ins.i_in},{ins.i_out}] {linear.irreps_in[ins.i_in]},{linear.irreps_out[ins.i_out]}"
            ),
            ins.path_shape,
            ins.weight_std,
            input.dtype,
        )
        for ins in linear.instructions
    ]
    f = lambda x: linear(w, x)
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)

    return f(input)


def _linear_indexed(
    input: e3nn.IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
    indices: jax.Array,
    num_indexed_weights: int,
) -> e3nn.IrrepsArray:
    """Linear layer with indexed weights.

    Each input get an index, and the weights are indexed by these indices.
    """
    shape = jnp.broadcast_shapes(input.shape[:-1], indices.shape)
    input = input.broadcast_to(shape + (-1,))
    indices = jnp.broadcast_to(indices, shape)

    w = [
        get_parameter(
            (
                f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}"
                if ins.i_in == -1
                else f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}"
            ),
            (num_indexed_weights,) + ins.path_shape,
            ins.weight_std,
            input.dtype,
        )
        for ins in lin.instructions
    ]  # List of shape (num_weights, *path_shape)
    w = [wi[indices] for wi in w]  # List of shape (..., *path_shape)

    f = lin
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)
    return f(w, input)


class Linear(nnx.Module):
    r"""Equivariant Linear Flax NNX module

    Args:
        irreps_out (`Irreps`): output representations, if allowed bu Schur's lemma.
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.
        irreps_in (`Irreps`): input representations. If not specified,
            the input representations is obtained when calling the module.
        channel_in (optional int): required when using 'mixed_per_channel' linear_type,
            indicating the size of the last axis before the irreps in the input.
        biases (bool): whether to add a bias to the output.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        num_indexed_weights (optional int): number of indexed weights. See example below.
        weights_per_channel (bool): whether to have one set of weights per channel.
        force_irreps_out (bool): whether to force the output irreps to be the one specified in ``irreps_out``.

    Due to how Flax NNX is implemented, the rngs, irreps_in and irreps_out must be supplied at initialization.
    The type of the linear layer must also be supplied at initialization:
    'vanilla', 'indexed', 'mixed', 'mixed_per_channel'
    Also, depending on what type of linear layer is used, additional options
    (eg. 'num_indexed_weights', 'weights_per_channel', 'weights_dim', 'channel_in')
    must be supplied.

    Examples:
        Vanilla::

            >>> import e3nn_jax as e3nn
            >>> from flax import nnx

            >>> x = e3nn.normal("0e + 1o")
            >>> linear = Linear(
                    irreps_out="2x0e + 1o + 2e",
                    irreps_in=x.irreps,
                    rngs=nnx.Rngs(0),
                )
            >>> linear(x).irreps  # Note that the 2e is discarded. Avoid this by setting force_irreps_out=True.
            2x0e+1x1o
            >>> linear(x).shape
            (5,)

        External weights::

            >>> linear = Linear(
                    irreps_out="2x0e + 1o",
                    irreps_in=x.irreps,
                    linear_type="mixed",
                    weights_dim=4,
                    rngs=nnx.Rngs(0),
                )
            >>> e = jnp.array([1., 2., 3., 4.])
            >>> linear(e, x).irreps
                2x0e+1x1o
            >>> linear(e, x).shape
            (5,)

        Indexed weights::

            >>> linear = Linear(
                    irreps_out="2x0e + 1o + 2e",
                    irreps_in=x.irreps,
                    linear_type="indexed",
                    num_indexed_weights=3,
                    rngs=nnx.Rngs(0),
                )
            >>> i = jnp.array(2)
            >>> linear(i, x).irreps
                2x0e+1x1o
            >>> linear(i, x).shape
            (5,)
    """

    def __init__(
        self,
        *,
        irreps_out: e3nn.Irreps,
        irreps_in: e3nn.Irreps,
        channel_out: Optional[int] = None,
        channel_in: Optional[int] = None,
        biases: bool = False,
        path_normalization: Optional[Union[str, float]] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        num_indexed_weights: Optional[int] = None,
        weights_per_channel: bool = False,
        force_irreps_out: bool = False,
        weights_dim: Optional[int] = None,
        input_dtype: jnp.dtype = jnp.float32,
        linear_type: str = "vanilla",
        rngs: nnx.Rngs,
    ):
        irreps_in_regrouped = e3nn.Irreps(irreps_in).regroup()
        irreps_out = e3nn.Irreps(irreps_out)

        # Store configuration as regular attributes (static in Flax NNX)
        self.irreps_in: e3nn.Irreps = irreps_in_regrouped
        self.channel_in: Optional[int] = channel_in
        self.channel_out: Optional[int] = channel_out
        self.biases: bool = biases
        self.path_normalization: Optional[Union[str, float]] = path_normalization
        self.num_indexed_weights: Optional[int] = num_indexed_weights
        self.weights_per_channel: bool = weights_per_channel
        self.force_irreps_out: bool = force_irreps_out
        self.linear_type: str = linear_type
        self.weights_dim: Optional[int] = weights_dim
        self._input_dtype: jnp.dtype = input_dtype

        self.gradient_normalization: float = _get_gradient_normalization(
            gradient_normalization
        )

        channel_irrep_multiplier = 1
        # 모델 초기화때는 건너뜀. 나중에 필요함.
        if self.channel_out is not None:
            assert not self.weights_per_channel
            channel_irrep_multiplier = self.channel_out
        
        # force_irreps_out=True이면 irreps_out을 그대로 사용. 
        # False이면 irreps_in_regrouped를 기반으로 irreps_out을 필터링하고 단순화.
        if not self.force_irreps_out:
            irreps_out = irreps_out.filter(keep=irreps_in_regrouped)
            irreps_out = irreps_out.simplify()
        self.irreps_out: e3nn.Irreps = irreps_out

        self._linear: FunctionalLinear = FunctionalLinear(
            irreps_in_regrouped,
            channel_irrep_multiplier * irreps_out,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )

        # irreps_in="32x0e",   # 입력: 32차원 스칼라
        # irreps_out="1x0e",   # 출력: 1차원 스칼라
        # 일반적이 Linear 변환 이다. 
        # y = W @ x
        # x: (32,)  입력 벡터
        # W: (1, 32) weight 행렬
        # y: (1,)   출력 벡터      


        # Initialize weights using rngs
        key = rngs.params()
        self._init_weights(key)

    def _init_weights(self, key: jax.Array):
        """Constructs the weights for the linear module as nnx.Param."""
        irreps_in = self._linear.irreps_in
        irreps_out = self._linear.irreps_out

        # instructions = [
        #     Instruction(
        #         i_in=0,        # 입력 0번 irrep (64x0e)
        #         i_out=0,       # 출력 0번 irrep (1x0e)
        #         path_shape=(64, 1),  # weight shape
        #         weight_std=1/sqrt(64),      # 초기화 표준편차
        #         ...
        #     ),
        # ]

        # 예: irreps_in="64x0e", irreps_out="1x0e" 면
        # instructions = [Instruction(i_in=0, i_out=0, path_shape=(64,1), ...)]
        # → instruction 1개

        # 예: irreps_in="64x0e + 64x1o", irreps_out="64x0e + 64x1o" 면
        # instructions = [
        #     Instruction(i_in=0, i_out=0, ...),  # 0e → 0e
        #     Instruction(i_in=1, i_out=1, ...),  # 1o → 1o
        # ]
        # → instruction 2개

        for ins in self._linear.instructions:
            weight_key, key = jax.random.split(key)
            if ins.i_in == -1:
                name = f"b[{ins.i_out}] {irreps_out[ins.i_out]}"
            else:
                name = f"w[{ins.i_in},{ins.i_out}] {irreps_in[ins.i_in]},{irreps_out[ins.i_out]}"
                # ins.i_in = 0   # -1이 아님 → bias 아님 → weight
                # else 실행:
                # name = f"w[{ins.i_in},{ins.i_out}] {irreps_in[ins.i_in]},{irreps_out[ins.i_out]}"
                # name = f"w[0,0] 64x0e,1x0e"

            # Sanitize name for use as attribute name
            attr_name = self._sanitize_name(name) # "weight_w_0_0_64x0e_1x0e"
            weight_std = ins.weight_std           # 1/sqrt(64) = 1/8 = 0.125

            #if self.linear_type == "valilla":
            weight_shape = ins.path_shape         # (64, 1)

            # model 초기화때는 indexed로 사용함. 
            if self.linear_type == "indexed":
                if self.num_indexed_weights is None:
                    raise ValueError(
                        "num_indexed_weights must be provided when 'linear_type' is 'indexed'"
                    )

                weight_shape = (self.num_indexed_weights,) + ins.path_shape
                # atoms speices = 89
                # 예시: self.num_indexed_weights = 89
                # ins.path_shape = (64, 1)
                # weight_shape = (89, 64, 1)

            weight_value = weight_std * jax.random.normal(
                weight_key,
                weight_shape,
                self._input_dtype,
            )
                # jax.random.normal → 표준정규분포 N(0, 1)에서 샘플링
                # jax.random.normal(weight_key, (89, 64, 1), jnp.float32)
                # → shape (89, 64, 1) 배열, 각 값이 대략 -3 ~ +3 사이 (표준정규분포 평균0, 표준편차1 의 성질)

                # weight_value = weight_std * jax.random.normal(weight_key, weight_shape, self._input_dtype)
                # weight_value = 0.125 * jax.random.normal(weight_key, (89, 64, 1), jnp.float32)
                # → shape (89, 64, 1) 배열, 각 값이 대략 -0.375 ~ +0.375 사이

            # Store as nnx.Param with sanitized attribute name
            setattr(self, attr_name, nnx.Param(weight_value))
                # self.weight_w_0_0_64x0e_1x0e = nnx.Param(shape=(89, 64, 1))
                # → 학습 가능한 파라미터로 등록됨
                # → 훈련하면서 gradient로 업데이트됨

        # Store mapping from original names to sanitized names
        self._weight_name_mapping: Dict[str, str] = self._build_name_mapping()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert weight name to valid Python attribute name."""
        # Replace problematic characters with underscores
        sanitized = name.replace("[", "_").replace("]", "_")
        sanitized = sanitized.replace(",", "_").replace(" ", "_")
        sanitized = sanitized.replace(".", "_")
        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        # Remove trailing underscores
        sanitized = sanitized.strip("_")
        return f"weight_{sanitized}"


    def _build_name_mapping(self) -> Dict[str, str]:
        """Build mapping from original weight names to sanitized attribute names."""
        irreps_in = self._linear.irreps_in
        irreps_out = self._linear.irreps_out

        mapping = {}
        for ins in self._linear.instructions:
            if ins.i_in == -1:
                name = f"b[{ins.i_out}] {irreps_out[ins.i_out]}"
            else:
                name = f"w[{ins.i_in},{ins.i_out}] {irreps_in[ins.i_in]},{irreps_out[ins.i_out]}"
            mapping[name] = self._sanitize_name(name)
        return mapping

    def _get_weight(self, name: str) -> jax.Array:
        """Get weight array by original name."""
        attr_name = self._weight_name_mapping[name]
        param = getattr(self, attr_name)
        return param.value

    def __call__(self, weights_or_input, input_or_none=None) -> e3nn.IrrepsArray:
        """Apply the linear operator.

        Args:
            weights (optional IrrepsArray or jax.Array): scalar weights that are contracted with free parameters.
                An array of shape ``(..., contracted_axis)``. Broadcasting with `input` is supported.
            input (IrrepsArray): input irreps-array of shape ``(..., [channel_in,] irreps_in.dim)``.
                Broadcasting with `weights` is supported.

        Returns:
            IrrepsArray: output irreps-array of shape ``(..., [channel_out,] irreps_out.dim)``.
                Properly normalized assuming that the weights and input are properly normalized.
        """
        if input_or_none is None:
            weights = None
            input: e3nn.IrrepsArray = weights_or_input
        else:
            weights: jax.Array = weights_or_input
            input: e3nn.IrrepsArray = input_or_none
        del weights_or_input, input_or_none

        input = e3nn.as_irreps_array(input)

        dtype = get_pytree_dtype(weights, input)
        if dtype.kind == "i":
            dtype = jnp.float32
        input = input.astype(dtype)

        if self.irreps_in != input.irreps.regroup():
            raise ValueError(
                f"e3nn.nnx.Linear: The input irreps ({input.irreps}) "
                f"do not match the expected irreps ({self.irreps_in})."
            )

        if self.channel_in is not None:
            if self.channel_in != input.shape[-2]:
                raise ValueError(
                    f"e3nn.nnx.Linear: The input channel ({input.shape[-2]}) "
                    f"does not match the expected channel ({self.channel_in})."
                )

        input = input.remove_zero_chunks().regroup()

        def get_parameter(
            name: str,
            path_shape: Tuple[int, ...],
            weight_std: float,
            dtype: jnp.dtype = jnp.float32,
        ):
            del path_shape, weight_std, dtype
            return self._get_weight(name)

        assertion_message = (
            "Weights cannot be provided when 'linear_type' is 'vanilla'."
            "Otherwise, weights must be provided."
            "If weights are provided, they must be either: \n"
            "* integers and num_indexed_weights must be provided, or \n"
            "* floats and num_indexed_weights must not be provided.\n"
            f"weights.dtype={weights.dtype if weights is not None else None}, "
            f"num_indexed_weights={self.num_indexed_weights}"
        )

        if self.linear_type == "vanilla":
            assert weights is None, assertion_message
            output = _linear_vanilla(input, self._linear, get_parameter)

        elif self.linear_type == "indexed":
            assert weights is not None, assertion_message
            if isinstance(weights, e3nn.IrrepsArray):
                if not weights.irreps.is_scalar():
                    raise ValueError("weights must be scalar")
                weights = weights.array
            output = _linear_indexed(
                input, self._linear, get_parameter, weights, self.num_indexed_weights
            )

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)

        return output.rechunk(self.irreps_out)


class FullyConnectedTensorProduct(nnx.Module):
    r"""Equivariant Fully Connected Tensor Product Flax NNX module.

    This module wraps ``e3nn_jax.legacy.FunctionalFullyConnectedTensorProduct``
    for use with Flax NNX, automatically managing the weights as ``nnx.Param``.

    Args:
        irreps_in1 (`Irreps`): Irreps for the first input.
        irreps_in2 (`Irreps`): Irreps for the second input.
        irreps_out (`Irreps`): Output representations.
        irrep_normalization (str): Normalization of the irreps, ``component`` or ``norm``.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        num_indexed_weights (optional int): Number of indexed weights for per-element tensor products.
        input_dtype (jnp.dtype): Data type for weights initialization.
        rngs (nnx.Rngs): Random number generators for initialization.

    Examples:
        Basic usage::

            >>> import e3nn_jax as e3nn
            >>> from flax import nnx

            >>> x1 = e3nn.normal("1x0e + 1x1o")
            >>> x2 = e3nn.normal("1x0e + 1x1e")
            >>> tp = FullyConnectedTensorProduct(
                    irreps_in1="1x0e + 1x1o",
                    irreps_in2="1x0e + 1x1e",
                    irreps_out="1x0e + 1x1o + 1x1e",
                    rngs=nnx.Rngs(0),
                )
            >>> output = tp(x1, x2)
            >>> output.irreps
            1x0e+1x1o+1x1e

        Indexed weights (per-species)::

            >>> tp = FullyConnectedTensorProduct(
                    irreps_in1="1x0e + 1x1o",
                    irreps_in2="1x0e + 1x1e",
                    irreps_out="1x0e + 1x1o",
                    num_indexed_weights=4,  # e.g., 4 species
                    rngs=nnx.Rngs(0),
                )
            >>> species_idx = jnp.array([0, 1, 2, 0])  # shape (4,)
            >>> x1_batch = e3nn.normal("1x0e + 1x1o", (4,))  # shape (4, irreps_dim)
            >>> x2_batch = e3nn.normal("1x0e + 1x1e", (4,))
            >>> output = tp(species_idx, x1_batch, x2_batch)
    """

    def __init__(
        self,
        *,
        irreps_in1: e3nn.Irreps,
        irreps_in2: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        irrep_normalization: Optional[str] = None,
        path_normalization: Optional[Union[str, float]] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        num_indexed_weights: Optional[int] = None,
        input_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.irreps_in1: e3nn.Irreps = e3nn.Irreps(irreps_in1)
        self.irreps_in2: e3nn.Irreps = e3nn.Irreps(irreps_in2)
        self.irreps_out: e3nn.Irreps = e3nn.Irreps(irreps_out)
        self.num_indexed_weights: Optional[int] = num_indexed_weights
        self._input_dtype: jnp.dtype = input_dtype

        self.gradient_normalization: float = _get_gradient_normalization(
            gradient_normalization
        )

        # Create the functional tensor product
        self._tp: FunctionalFullyConnectedTensorProduct = FunctionalFullyConnectedTensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            gradient_normalization=gradient_normalization,
        )

        # Initialize weights
        key = rngs.params()
        self._init_weights(key)

    def _init_weights(self, key: jax.Array):
        """Initialize weights for the tensor product as nnx.Param."""
        for idx, ins in enumerate(self._tp.instructions):
            weight_key, key = jax.random.split(key)

            # Build descriptive name based on instruction
            name = (
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                f"{self.irreps_in1[ins.i_in1]},"
                f"{self.irreps_in2[ins.i_in2]},"
                f"{self.irreps_out[ins.i_out]}"
            )

            # Sanitize name for attribute
            attr_name = self._sanitize_name(name, idx)

            # Determine weight shape
            weight_shape = ins.path_shape
            if self.num_indexed_weights is not None:
                weight_shape = (self.num_indexed_weights,) + ins.path_shape

            # Initialize with proper scaling
            weight_std = ins.weight_std if hasattr(ins, 'weight_std') else 1.0
            weight_value = weight_std * jax.random.normal(
                weight_key,
                weight_shape,
                self._input_dtype,
            )

            # Store as nnx.Param
            setattr(self, attr_name, nnx.Param(weight_value))

        # Store mapping from indices to attribute names
        self._weight_attr_names: list = self._build_attr_name_list()

    @staticmethod
    def _sanitize_name(name: str, idx: int) -> str:
        """Convert weight name to valid Python attribute name."""
        sanitized = name.replace("[", "_").replace("]", "_")
        sanitized = sanitized.replace(",", "_").replace(" ", "_")
        sanitized = sanitized.replace(".", "_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        sanitized = sanitized.strip("_")
        return f"tp_weight_{idx}_{sanitized}"

    def _build_attr_name_list(self) -> list:
        """Build list of attribute names for weights in instruction order."""
        attr_names = []
        for idx, ins in enumerate(self._tp.instructions):
            name = (
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                f"{self.irreps_in1[ins.i_in1]},"
                f"{self.irreps_in2[ins.i_in2]},"
                f"{self.irreps_out[ins.i_out]}"
            )
            attr_names.append(self._sanitize_name(name, idx))
        return attr_names

    def _get_weights(self) -> list:
        """Get list of weight arrays in instruction order."""
        return [getattr(self, attr_name).value for attr_name in self._weight_attr_names]

    def __call__(
        self,
        input1_or_indices: Union[e3nn.IrrepsArray, jax.Array],
        input2_or_input1: Union[e3nn.IrrepsArray, jax.Array],
        input2_or_none: Optional[e3nn.IrrepsArray] = None,
    ) -> e3nn.IrrepsArray:
        """Apply the tensor product.

        Args:
            For vanilla mode (num_indexed_weights is None):
                input1 (IrrepsArray): First input irreps-array.
                input2 (IrrepsArray): Second input irreps-array.

            For indexed mode (num_indexed_weights is set):
                indices (jax.Array): Integer indices for selecting weights.
                input1 (IrrepsArray): First input irreps-array.
                input2 (IrrepsArray): Second input irreps-array.

        Returns:
            IrrepsArray: Output irreps-array from the tensor product.
        """
        if input2_or_none is None:
            # Vanilla mode: (input1, input2)
            indices = None
            input1 = e3nn.as_irreps_array(input1_or_indices)
            input2 = e3nn.as_irreps_array(input2_or_input1)
        else:
            # Indexed mode: (indices, input1, input2)
            indices = input1_or_indices
            input1 = e3nn.as_irreps_array(input2_or_input1)
            input2 = e3nn.as_irreps_array(input2_or_none)

        # Get weights
        weights = self._get_weights()

        if self.num_indexed_weights is not None:
            if indices is None:
                raise ValueError(
                    "Indices must be provided when num_indexed_weights is set."
                )
            # Index into weights: each weight has shape (num_indexed, *path_shape)
            # After indexing: shape becomes (..., *path_shape)
            weights = [w[indices] for w in weights]

            # Apply tensor product with vmap for batched inputs
            f = self._tp.left_right
            for _ in range(input1.ndim - 1):
                f = e3nn.utils.vmap(f)
            return f(weights, input1, input2)
        else:
            if indices is not None:
                raise ValueError(
                    "Indices should not be provided when num_indexed_weights is None."
                )
            # Vanilla mode: apply tensor product with vmap for batched inputs
            f = lambda x1, x2: self._tp.left_right(weights, x1, x2)
            for _ in range(input1.ndim - 1):
                f = e3nn.utils.vmap(f)
            return f(input1, input2)
