from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._snowflake.utils import native_to_narwhals_dtype
from narwhals._sql.dataframe import SQLLazyFrame
from narwhals._utils import Implementation, ValidateBackendVersion, Version

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals._snowflake.expr import SnowflakeExpr
    from narwhals._snowflake.group_by import SnowflakeGroupBy
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

    def filter(self, predicate: SnowflakeExpr) -> Self:
        """Filter rows based on a predicate expression.
        
        Arguments:
            predicate: A boolean expression to filter rows.
            
        Returns:
            A new SnowflakeLazyFrame with filtered rows.
            
        Raises:
            RuntimeError: If the filter operation fails.
        """
        try:
            # Evaluate the predicate expression to get the Snowpark Column
            # [0] is safe as the predicate expression returns a single column
            mask = predicate(self)[0]
            # Use Snowpark's filter method (equivalent to where)
            return self._with_native(self._native_frame.filter(mask))
        except Exception as e:
            msg = f"Failed to apply filter on Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def with_columns(self, *exprs: SnowflakeExpr) -> Self:
        """Add or modify columns in the DataFrame.
        
        This method evaluates the given expressions and adds them as new columns
        or modifies existing columns with the same name. The operation uses
        Snowpark's select to reconstruct the DataFrame with the new/modified columns.
        
        Arguments:
            exprs: Expressions that define the columns to add or modify.
            
        Returns:
            A new SnowflakeLazyFrame with the added/modified columns.
            
        Raises:
            RuntimeError: If the with_columns operation fails.
        """
        try:
            from narwhals._snowflake.utils import col, evaluate_exprs_and_aliases
            
            # Evaluate all expressions and get their aliases
            new_columns_map = dict(evaluate_exprs_and_aliases(self, *exprs))
            
            # Build the selection: existing columns (possibly replaced) + new columns
            result = []
            
            # First, add all existing columns (using new values if they exist in new_columns_map)
            for name in self.columns:
                if name in new_columns_map:
                    # Replace with new column value
                    result.append(new_columns_map.pop(name).alias(name))
                else:
                    # Keep existing column
                    result.append(col(name))
            
            # Then, add any remaining new columns that weren't replacements
            result.extend(value.alias(name) for name, value in new_columns_map.items())
            
            # Use Snowpark's select to apply all columns at once
            return self._with_native(self._native_frame.select(*result))
        except Exception as e:
            msg = f"Failed to apply with_columns on Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def drop_nulls(self, subset: list[str] | None) -> Self:
        """Drop rows with null values in specified columns.
        
        Arguments:
            subset: Column names to check for null values. If None, check all columns.
            
        Returns:
            A new SnowflakeLazyFrame with null rows removed.
        """
        from functools import reduce
        from operator import and_

        from narwhals._snowflake.utils import col

        subset_ = subset if subset is not None else self.columns
        keep_condition = reduce(and_, (col(name).is_not_null() for name in subset_))
        return self._with_native(self._native_frame.filter(keep_condition))

    def rename(self, mapping: dict[str, str]) -> Self:
        """Rename columns in the DataFrame.
        
        Arguments:
            mapping: Dictionary mapping old column names to new column names.
            
        Returns:
            A new SnowflakeLazyFrame with renamed columns.
        """
        result = self._native_frame
        for old_name, new_name in mapping.items():
            result = result.with_column_renamed(old_name, new_name)
        return self._with_native(result)

    def group_by(
        self, *keys: str | SnowflakeExpr, drop_null_keys: bool
    ) -> SnowflakeGroupBy:
        """Group the DataFrame by one or more columns.
        
        Arguments:
            keys: Column names or expressions to group by.
            drop_null_keys: Whether to drop rows with null values in the grouping keys.
            
        Returns:
            A SnowflakeGroupBy object for performing aggregations.
        """
        from narwhals._snowflake.group_by import SnowflakeGroupBy

        return SnowflakeGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def join(
        self,
        other: Self,
        *,
        how: str,
        left_on: list[str] | None,
        right_on: list[str] | None,
        suffix: str,
    ) -> Self:
        """Join this DataFrame with another DataFrame.
        
        Arguments:
            other: The right DataFrame to join with.
            how: Join strategy - one of 'inner', 'left', 'full', 'cross', 'semi', 'anti'.
            left_on: Column names from the left DataFrame to join on.
            right_on: Column names from the right DataFrame to join on.
            suffix: Suffix to add to overlapping column names from the right DataFrame.
            
        Returns:
            A new SnowflakeLazyFrame with the joined data.
            
        Raises:
            RuntimeError: If the join operation fails.
        """
        from functools import reduce
        from operator import and_

        from narwhals._snowflake.utils import col

        try:
            # Map Narwhals join types to Snowpark join types
            # Snowpark uses 'outer' instead of 'full'
            native_how = "outer" if how == "full" else how
            
            if native_how == "cross":
                # Cross join doesn't need join conditions
                lhs = self._native_frame.alias("lhs")
                rhs = other._native_frame.alias("rhs")
                joined = lhs.cross_join(rhs)
            else:
                # All other join types need join conditions
                assert left_on is not None  # noqa: S101
                assert right_on is not None  # noqa: S101
                
                # Create join conditions
                lhs = self._native_frame.alias("lhs")
                rhs = other._native_frame.alias("rhs")
                
                # Build join condition by combining all key pairs with AND
                conditions = [
                    col(f'lhs."{left}"') == col(f'rhs."{right}"')
                    for left, right in zip(left_on, right_on)
                ]
                condition = reduce(and_, conditions)
                
                # Perform the join using Snowpark's join method
                # Snowpark join signature: join(right, on, join_type="inner")
                joined = lhs.join(rhs, on=condition, join_type=native_how)
            
            # Handle column selection and suffix application
            if native_how in {"inner", "left", "cross", "outer"}:
                # Select columns from left side
                select = [col(f'lhs."{name}"') for name in self.columns]
                
                # Add columns from right side with appropriate handling
                for name in other.columns:
                    col_in_lhs = name in self.columns
                    
                    if native_how == "outer" and not col_in_lhs:
                        # For outer join, non-overlapping columns from right don't need suffix
                        select.append(col(f'rhs."{name}"'))
                    elif (native_how == "outer") or (
                        col_in_lhs and (right_on is None or name not in right_on)
                    ):
                        # Add suffix for overlapping columns (except join keys in outer join)
                        select.append(col(f'rhs."{name}"').alias(f"{name}{suffix}"))
                    elif right_on is None or name not in right_on:
                        # Non-overlapping columns from right don't need suffix
                        select.append(col(f'rhs."{name}"'))
                
                result = joined.select(*select)
            else:
                # For semi and anti joins, only return left columns
                select = [col(f'lhs."{name}"') for name in self.columns]
                result = joined.select(*select)
            
            return self._with_native(result)
            
        except Exception as e:
            msg = f"Failed to perform join on Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e
