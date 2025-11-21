from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._snowflake.utils import native_to_narwhals_dtype
from narwhals._sql.dataframe import SQLLazyFrame
from narwhals._utils import Implementation, ValidateBackendVersion, Version

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals._snowflake.expr import SnowflakeExpr
    from narwhals._snowflake.namespace import SnowflakeNamespace
    from narwhals._snowflake.typing import SnowparkDataFrameT
    from narwhals._utils import _LimitedContext
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.stable.v1 import DataFrame as DataFrameV1


class SnowflakeLazyFrame(
    SQLLazyFrame[
        "SnowflakeExpr",
        "SnowparkDataFrameT",
        "LazyFrame[SnowparkDataFrameT] | DataFrameV1[SnowparkDataFrameT]",
    ],
    ValidateBackendVersion,
):
    """Narwhals LazyFrame wrapper for Snowpark DataFrame.
    
    This class wraps a Snowpark DataFrame and provides the Narwhals LazyFrame
    interface for lazy query execution on Snowflake.
    """

    _implementation = Implementation.SNOWFLAKE

    def __init__(
        self,
        df: SnowparkDataFrameT,
        *,
        version: Version,
        validate_backend_version: bool = False,
    ) -> None:
        """Initialize a SnowflakeLazyFrame.
        
        Arguments:
            df: The native Snowpark DataFrame to wrap.
            version: The Narwhals version for compatibility.
            validate_backend_version: Whether to validate the backend version.
        """
        self._native_frame: SnowparkDataFrameT = df
        self._version = version
        if validate_backend_version:
            self._validate_backend_version()

    @property
    def _backend_version(self) -> tuple[int, ...]:
        """Get the Snowpark backend version."""
        return self._implementation._backend_version()

    @staticmethod
    def _is_native(obj: SnowparkDataFrameT | Any) -> TypeIs[SnowparkDataFrameT]:
        """Check if an object is a native Snowpark DataFrame.
        
        Arguments:
            obj: The object to check.
            
        Returns:
            True if the object is a Snowpark DataFrame, False otherwise.
        """
        from narwhals.dependencies import is_snowpark_dataframe

        return is_snowpark_dataframe(obj)

    @classmethod
    def from_native(
        cls, data: SnowparkDataFrameT, /, *, context: _LimitedContext
    ) -> Self:
        """Create a SnowflakeLazyFrame from a native Snowpark DataFrame.
        
        Arguments:
            data: The native Snowpark DataFrame.
            context: The limited context containing version information.
            
        Returns:
            A new SnowflakeLazyFrame instance.
        """
        return cls(data, version=context._version)

    def to_narwhals(
        self, *args: Any, **kwds: Any
    ) -> LazyFrame[SnowparkDataFrameT] | DataFrameV1[SnowparkDataFrameT]:
        """Convert to a Narwhals LazyFrame or DataFrame.
        
        Returns:
            A Narwhals LazyFrame or DataFrame depending on the version.
        """
        if self._version is Version.V1:
            from narwhals.stable.v1 import DataFrame as DataFrameV1

            return DataFrameV1(self, level="interchange")  # type: ignore[no-any-return]
        return self._version.lazyframe(self, level="lazy")

    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        """Return self as a dataframe (for V1 compatibility).
        
        Returns:
            Self.
            
        Raises:
            AttributeError: If not using V1 version.
        """
        if self._version is not Version.V1:
            msg = "__narwhals_dataframe__ is not implemented for SnowflakeLazyFrame"
            raise AttributeError(msg)
        return self

    def __narwhals_lazyframe__(self) -> Self:
        """Return self as a lazyframe.
        
        Returns:
            Self.
        """
        return self

    def __native_namespace__(self) -> ModuleType:
        """Get the native Snowpark module.
        
        Returns:
            The snowflake.snowpark module.
        """
        from narwhals.dependencies import get_snowflake

        return get_snowflake()  # type: ignore[no-any-return]

    def __narwhals_namespace__(self) -> SnowflakeNamespace:
        """Get the Snowflake namespace.
        
        Returns:
            A SnowflakeNamespace instance.
        """
        from narwhals._snowflake.namespace import SnowflakeNamespace

        return SnowflakeNamespace(version=self._version)

    def _with_native(self, df: SnowparkDataFrameT) -> Self:
        """Create a new SnowflakeLazyFrame with a different native DataFrame.
        
        Arguments:
            df: The new native Snowpark DataFrame.
            
        Returns:
            A new SnowflakeLazyFrame instance.
        """
        return self.__class__(df, version=self._version)

    @property
    def schema(self) -> dict[str, DType]:
        """Get the schema of the DataFrame.
        
        Returns:
            A dictionary mapping column names to Narwhals dtypes.
        """
        snowpark_schema = self._native_frame.schema
        return {
            field.name: native_to_narwhals_dtype(field.datatype, self._version)
            for field in snowpark_schema.fields
        }

    @property
    def columns(self) -> list[str]:
        """Get the column names of the DataFrame.
        
        Returns:
            A list of column names.
        """
        return self._native_frame.columns

    def collect_schema(self) -> dict[str, DType]:
        """Collect the schema without executing the query.
        
        Returns:
            A dictionary mapping column names to Narwhals dtypes.
        """
        return self.schema

    def select(self, *exprs: SnowflakeExpr) -> Self:
        """Select columns using expressions.
        
        Arguments:
            exprs: The expressions to select.
            
        Returns:
            A new SnowflakeLazyFrame with the selected columns.
        """
        from narwhals._snowflake.utils import evaluate_exprs_and_aliases

        selection = (
            val.alias(name) for name, val in evaluate_exprs_and_aliases(self, *exprs)
        )
        return self._with_native(self._native_frame.select(*selection))

    def simple_select(self, *column_names: str) -> Self:
        """Select columns by name.
        
        Arguments:
            column_names: The names of columns to select.
            
        Returns:
            A new SnowflakeLazyFrame with the selected columns.
        """
        return self._with_native(self._native_frame.select(*column_names))

    def drop(self, columns: list[str], *, strict: bool) -> Self:
        """Drop columns from the DataFrame.
        
        Arguments:
            columns: The names of columns to drop.
            strict: Whether to raise an error if a column doesn't exist.
            
        Returns:
            A new SnowflakeLazyFrame without the dropped columns.
        """
        from narwhals._snowflake.utils import col
        from narwhals._utils import parse_columns_to_drop

        columns_to_drop = parse_columns_to_drop(self, columns, strict=strict)
        selection = [col(name) for name in self.columns if name not in columns_to_drop]
        return self._with_native(self._native_frame.select(*selection))

    def head(self, n: int) -> Self:
        """Get the first n rows.
        
        Arguments:
            n: The number of rows to return.
            
        Returns:
            A new SnowflakeLazyFrame with at most n rows.
        """
        return self._with_native(self._native_frame.limit(n))

    def lazy(self, backend: None = None, **_: None) -> Self:
        """Return self as a lazy frame.
        
        Arguments:
            backend: Unused parameter for compatibility.
            
        Returns:
            Self.
        """
        if backend is not None:  # pragma: no cover
            msg = "`backend` argument is not supported for Snowflake"
            raise ValueError(msg)
        return self
