from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from narwhals._utils import Version, isinstance_or_issubclass

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._snowflake.typing import SnowparkColumnT, SnowparkDataTypeT
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType


def native_to_narwhals_dtype(
    snowflake_type: SnowparkDataTypeT, version: Version
) -> DType:
    """Convert Snowflake data type to Narwhals dtype.

    Arguments:
        snowflake_type: Snowflake Snowpark data type.
        version: Narwhals version for dtype compatibility.

    Returns:
        Corresponding Narwhals DType.
    """
    from snowflake.snowpark import types as T

    dtypes = version.dtypes
    type_class = type(snowflake_type)

    # Handle nested data types first
    if isinstance(snowflake_type, T.ArrayType):
        inner = native_to_narwhals_dtype(snowflake_type.element_type, version)
        return dtypes.List(inner)

    if isinstance(snowflake_type, T.MapType):
        # Map to Struct with key-value fields
        key_dtype = native_to_narwhals_dtype(snowflake_type.key_type, version)
        value_dtype = native_to_narwhals_dtype(snowflake_type.value_type, version)
        return dtypes.Struct(
            [
                dtypes.Field("key", key_dtype),
                dtypes.Field("value", value_dtype),
            ]
        )

    if isinstance(snowflake_type, T.StructType):
        fields = [
            dtypes.Field(
                field.name, native_to_narwhals_dtype(field.datatype, version)
            )
            for field in snowflake_type.fields
        ]
        return dtypes.Struct(fields)

    # Handle simple types
    return _non_nested_native_to_narwhals_dtype(type_class, version)


@lru_cache(maxsize=16)
def _non_nested_native_to_narwhals_dtype(
    snowflake_type_class: type, version: Version
) -> DType:
    """Convert non-nested Snowflake types to Narwhals dtypes.

    Arguments:
        snowflake_type_class: The class of the Snowflake type.
        version: Narwhals version for dtype compatibility.

    Returns:
        Corresponding Narwhals DType.
    """
    from snowflake.snowpark import types as T

    dtypes = version.dtypes

    type_map: Mapping[type, DType] = {
        T.LongType: dtypes.Int64(),
        T.IntegerType: dtypes.Int32(),
        T.ShortType: dtypes.Int16(),
        T.ByteType: dtypes.Int8(),
        T.FloatType: dtypes.Float32(),
        T.DoubleType: dtypes.Float64(),
        T.DecimalType: dtypes.Decimal(),
        T.StringType: dtypes.String(),
        T.BooleanType: dtypes.Boolean(),
        T.DateType: dtypes.Date(),
        T.TimestampType: dtypes.Datetime(),
        T.TimeType: dtypes.Time(),
        T.BinaryType: dtypes.Binary(),
        T.VariantType: dtypes.Object(),
    }

    return type_map.get(snowflake_type_class, dtypes.Unknown())


dtypes = Version.MAIN.dtypes
NW_TO_SNOWFLAKE_DTYPES: Mapping[type[DType], Any] = {
    dtypes.Float64: "DoubleType",
    dtypes.Float32: "FloatType",
    dtypes.Binary: "BinaryType",
    dtypes.String: "StringType",
    dtypes.Boolean: "BooleanType",
    dtypes.Date: "DateType",
    dtypes.Time: "TimeType",
    dtypes.Int8: "ByteType",
    dtypes.Int16: "ShortType",
    dtypes.Int32: "IntegerType",
    dtypes.Int64: "LongType",
}
UNSUPPORTED_DTYPES = (
    dtypes.Int128,
    dtypes.UInt8,
    dtypes.UInt16,
    dtypes.UInt32,
    dtypes.UInt64,
    dtypes.UInt128,
    dtypes.Categorical,
    dtypes.Enum,
)


def narwhals_to_native_dtype(dtype: IntoDType, version: Version) -> SnowparkDataTypeT:
    """Convert Narwhals dtype to Snowflake Snowpark data type.

    Arguments:
        dtype: Narwhals dtype to convert.
        version: Narwhals version for dtype compatibility.

    Returns:
        Corresponding Snowflake Snowpark DataType.

    Raises:
        NotImplementedError: If the dtype is not supported for Snowflake.
        ValueError: If the dtype cannot be converted.
    """
    from snowflake.snowpark import types as T

    dtypes = version.dtypes
    base_type = dtype.base_type()

    # Handle simple types
    if snowflake_type_name := NW_TO_SNOWFLAKE_DTYPES.get(base_type):
        return getattr(T, snowflake_type_name)()

    # Handle Datetime with time units
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        # Snowflake TimestampType doesn't have time unit parameter
        # All timestamps are stored with nanosecond precision
        return T.TimestampType()

    # Handle Duration
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        # Snowflake doesn't have a native Duration type
        # Map to LongType (nanoseconds)
        return T.LongType()

    # Handle List
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return T.ArrayType(inner)

    # Handle Struct
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        fields = [
            T.StructField(
                field.name, narwhals_to_native_dtype(field.dtype, version)
            )
            for field in dtype.fields
        ]
        return T.StructType(fields)

    # Handle Array (fixed-size)
    if isinstance_or_issubclass(dtype, dtypes.Array):
        # Snowflake ArrayType is variable-length, so we treat it as List
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return T.ArrayType(inner)

    # Handle Decimal
    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        # Default precision and scale
        return T.DecimalType(38, 0)

    # Unsupported types
    if issubclass(base_type, UNSUPPORTED_DTYPES):
        msg = f"Converting to {base_type.__name__} dtype is not supported for Snowflake."
        raise NotImplementedError(msg)

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def col(name: str) -> SnowparkColumnT:
    """Create a Snowpark Column reference by name.

    Arguments:
        name: Column name.

    Returns:
        Snowpark Column object.
    """
    from snowflake.snowpark import functions as F

    return F.col(name)


def lit(value: Any, dtype: SnowparkDataTypeT | None = None) -> SnowparkColumnT:
    """Create a Snowpark literal Column.

    Arguments:
        value: Literal value.
        dtype: Optional Snowpark data type.

    Returns:
        Snowpark Column object representing the literal.
    """
    from snowflake.snowpark import functions as F

    if dtype is not None:
        return F.lit(value).cast(dtype)
    return F.lit(value)


def when(condition: SnowparkColumnT) -> Any:
    """Create a Snowpark WHEN expression.

    Arguments:
        condition: Boolean condition Column.

    Returns:
        Snowpark CaseExpr object.
    """
    from snowflake.snowpark import functions as F

    return F.when(condition)


def coalesce(*columns: SnowparkColumnT) -> SnowparkColumnT:
    """Create a Snowpark COALESCE expression.

    Arguments:
        columns: Columns to coalesce.

    Returns:
        Snowpark Column object.
    """
    from snowflake.snowpark import functions as F

    return F.coalesce(*columns)


def function(name: str, *args: SnowparkColumnT) -> SnowparkColumnT:
    """Call a Snowpark function by name.

    Arguments:
        name: Function name.
        args: Function arguments.

    Returns:
        Snowpark Column object.
    """
    from snowflake.snowpark import functions as F

    # Handle special cases
    if name == "count_distinct":
        return F.count_distinct(*args)
    if name == "isnull":
        return args[0].is_null()
    if name == "isnotnull":
        return args[0].is_not_null()

    # Try to get the function from Snowpark functions module
    if hasattr(F, name):
        return getattr(F, name)(*args)

    msg = f"Function '{name}' is not available in Snowpark"
    raise NotImplementedError(msg)


def evaluate_exprs_and_aliases(
    df: Any, *exprs: Any
) -> list[tuple[str, SnowparkColumnT]]:
    """Evaluate expressions and return their aliases.
    
    Arguments:
        df: The dataframe to evaluate expressions on.
        exprs: The expressions to evaluate.
        
    Returns:
        A list of tuples containing (alias, evaluated_column).
    """
    result = []
    for expr in exprs:
        columns = expr(df)
        names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            names = expr._alias_output_names(names)
        result.extend(zip(names, columns))
    return result


__all__ = [
    "col",
    "coalesce",
    "evaluate_exprs_and_aliases",
    "function",
    "lit",
    "narwhals_to_native_dtype",
    "native_to_narwhals_dtype",
    "when",
]
