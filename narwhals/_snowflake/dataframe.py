from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._snowflake.utils import native_to_narwhals_dtype
from narwhals._sql.dataframe import SQLLazyFrame
from narwhals._utils import Implementation, ValidateBackendVersion, Version

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals._compliant.typing import CompliantDataFrameAny
    from narwhals._snowflake.expr import SnowflakeExpr
    from narwhals._snowflake.group_by import SnowflakeGroupBy
    from narwhals._snowflake.namespace import SnowflakeNamespace
    from narwhals._snowflake.typing import SnowparkDataFrameT
    from narwhals._typing import _EagerAllowedImpl
    from narwhals._utils import _LimitedContext
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.stable.v1 import DataFrame as DataFrameV1
    from narwhals.typing import UniqueKeepStrategy


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
            
        Raises:
            ImportError: If snowflake-snowpark-python is not installed.
        """
        from narwhals.dependencies import get_snowflake

        snowpark = get_snowflake()
        if snowpark is None:
            msg = (
                "Snowflake Snowpark is required for Snowflake backend support. "
                "Please install it with: pip install 'snowflake-snowpark-python[pandas]'"
            )
            raise ImportError(msg)
        return snowpark  # type: ignore[no-any-return]

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
    
    def _check_columns_exist(self, columns: Sequence[str]) -> ValueError | None:
        """Check if columns exist in the DataFrame.
        
        Arguments:
            columns: Column names to check.
            
        Returns:
            ValueError if any columns don't exist, None otherwise.
        """
        missing_cols = [col for col in columns if col not in self.columns]
        if missing_cols:
            return ValueError(
                f"Columns {missing_cols} do not exist in the DataFrame. "
                f"Available columns: {self.columns}"
            )
        return None

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
            
        Raises:
            ValueError: If no expressions are provided.
            RuntimeError: If the select operation fails.
        """
        if not exprs:
            msg = "At least one expression must be provided to select()"
            raise ValueError(msg)
            
        try:
            from narwhals._snowflake.utils import evaluate_exprs_and_aliases

            selection = (
                val.alias(name) for name, val in evaluate_exprs_and_aliases(self, *exprs)
            )
            return self._with_native(self._native_frame.select(*selection))
        except Exception as e:
            msg = f"Failed to select columns from Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def simple_select(self, *column_names: str) -> Self:
        """Select columns by name.
        
        Arguments:
            column_names: The names of columns to select.
            
        Returns:
            A new SnowflakeLazyFrame with the selected columns.
            
        Raises:
            ValueError: If no column names are provided or if column names don't exist.
            RuntimeError: If the select operation fails.
        """
        if not column_names:
            msg = "At least one column name must be provided to simple_select()"
            raise ValueError(msg)
            
        # Validate that all column names exist
        missing_cols = [col for col in column_names if col not in self.columns]
        if missing_cols:
            msg = f"Columns {missing_cols} do not exist in the DataFrame. Available columns: {self.columns}"
            raise ValueError(msg)
            
        try:
            return self._with_native(self._native_frame.select(*column_names))
        except Exception as e:
            msg = f"Failed to select columns from Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

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
            
        Raises:
            ValueError: If n is negative.
            RuntimeError: If the head operation fails.
        """
        if n < 0:
            msg = f"Number of rows must be non-negative, got {n}"
            raise ValueError(msg)
            
        try:
            return self._with_native(self._native_frame.limit(n))
        except Exception as e:
            msg = f"Failed to limit Snowflake DataFrame to {n} rows: {e}"
            raise RuntimeError(msg) from e

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
            
        Raises:
            ValueError: If subset contains column names that don't exist.
            RuntimeError: If the drop_nulls operation fails.
        """
        from functools import reduce
        from operator import and_

        from narwhals._snowflake.utils import col

        subset_ = subset if subset is not None else self.columns
        
        # Validate that all columns in subset exist
        if subset is not None:
            missing_cols = [col_name for col_name in subset if col_name not in self.columns]
            if missing_cols:
                msg = f"Columns {missing_cols} do not exist in the DataFrame. Available columns: {self.columns}"
                raise ValueError(msg)
        
        # Handle empty subset
        if not subset_:
            return self
            
        try:
            keep_condition = reduce(and_, (col(name).is_not_null() for name in subset_))
            return self._with_native(self._native_frame.filter(keep_condition))
        except Exception as e:
            msg = f"Failed to drop null rows from Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def rename(self, mapping: dict[str, str]) -> Self:
        """Rename columns in the DataFrame.
        
        Arguments:
            mapping: Dictionary mapping old column names to new column names.
            
        Returns:
            A new SnowflakeLazyFrame with renamed columns.
            
        Raises:
            ValueError: If mapping contains column names that don't exist or if new names conflict.
            RuntimeError: If the rename operation fails.
        """
        if not mapping:
            return self
            
        # Validate that all old column names exist
        missing_cols = [old for old in mapping.keys() if old not in self.columns]
        if missing_cols:
            msg = f"Columns {missing_cols} do not exist in the DataFrame. Available columns: {self.columns}"
            raise ValueError(msg)
        
        # Check for duplicate new names
        new_names = list(mapping.values())
        if len(new_names) != len(set(new_names)):
            duplicates = [name for name in new_names if new_names.count(name) > 1]
            msg = f"Duplicate new column names found: {set(duplicates)}"
            raise ValueError(msg)
            
        try:
            result = self._native_frame
            for old_name, new_name in mapping.items():
                result = result.with_column_renamed(old_name, new_name)
            return self._with_native(result)
        except Exception as e:
            msg = f"Failed to rename columns in Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def group_by(
        self, *keys: str | SnowflakeExpr, drop_null_keys: bool
    ) -> SnowflakeGroupBy:
        """Group the DataFrame by one or more columns.
        
        Arguments:
            keys: Column names or expressions to group by.
            drop_null_keys: Whether to drop rows with null values in the grouping keys.
            
        Returns:
            A SnowflakeGroupBy object for performing aggregations.
            
        Raises:
            ValueError: If no keys are provided or if column names don't exist.
        """
        if not keys:
            msg = "At least one grouping key must be provided to group_by()"
            raise ValueError(msg)
            
        # Validate string keys exist
        string_keys = [k for k in keys if isinstance(k, str)]
        if string_keys:
            missing_cols = [k for k in string_keys if k not in self.columns]
            if missing_cols:
                msg = f"Grouping columns {missing_cols} do not exist in the DataFrame. Available columns: {self.columns}"
                raise ValueError(msg)
        
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

    def sort(
        self, *by: str, descending: bool | Sequence[bool], nulls_last: bool
    ) -> Self:
        """Sort the DataFrame by one or more columns.
        
        Arguments:
            by: Column names to sort by.
            descending: Whether to sort in descending order. Can be a single bool
                for all columns or a sequence of bools for each column.
            nulls_last: Whether to place null values last in the sort order.
            
        Returns:
            A new SnowflakeLazyFrame with sorted rows.
            
        Raises:
            RuntimeError: If the sort operation fails.
        """
        from narwhals._snowflake.utils import col
        from narwhals._utils import extend_bool

        try:
            # Extend descending to match the number of sort columns
            descending_list = extend_bool(descending, len(by))
            
            # Build sort columns with appropriate ordering
            sort_cols = []
            for name, desc in zip(by, descending_list):
                sort_col = col(name)
                
                # Apply descending order if needed
                if desc:
                    sort_col = sort_col.desc()
                else:
                    sort_col = sort_col.asc()
                
                # Apply null ordering
                if nulls_last:
                    sort_col = sort_col.nulls_last()
                else:
                    sort_col = sort_col.nulls_first()
                
                sort_cols.append(sort_col)
            
            # Use Snowpark's sort method
            return self._with_native(self._native_frame.sort(*sort_cols))
            
        except Exception as e:
            msg = f"Failed to sort Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: UniqueKeepStrategy,
        order_by: Sequence[str] | None,
    ) -> Self:
        """Get unique rows from the DataFrame.
        
        Arguments:
            subset: Column names to consider for uniqueness. If None, use all columns.
            keep: Which duplicate to keep - 'any', 'first', 'last', or 'none'.
                - 'any': Keep any duplicate (fastest, uses distinct)
                - 'first': Keep the first occurrence
                - 'last': Keep the last occurrence
                - 'none': Drop all duplicates (keep only unique rows)
            order_by: Column names to order by when determining first/last.
            
        Returns:
            A new SnowflakeLazyFrame with unique rows.
            
        Raises:
            RuntimeError: If the unique operation fails.
        """
        from snowflake.snowpark import Window
        from snowflake.snowpark import functions as F

        from narwhals._snowflake.utils import col
        from narwhals._utils import extend_bool, generate_temporary_column_name

        try:
            # Determine which columns to check for uniqueness
            subset_ = subset if subset is not None else self.columns
            
            # Check that subset columns exist
            if error := self._check_columns_exist(subset_):
                raise error
            
            # For 'any' strategy, we can use Snowpark's distinct or drop_duplicates
            if keep == "any":
                if subset is None:
                    # Use distinct for all columns
                    return self._with_native(self._native_frame.distinct())
                else:
                    # Use drop_duplicates for specific columns
                    return self._with_native(self._native_frame.drop_duplicates(subset_))
            
            # For 'first', 'last', or 'none', we need to use window functions
            # Generate a temporary column name for the row number
            tmp_name = generate_temporary_column_name(8, self.columns, prefix="row_index_")
            
            # Build the window specification
            # Partition by the subset columns
            window_spec = Window.partition_by([col(name) for name in subset_])
            
            # Add ordering if specified
            if order_by:
                # For 'last', we need to reverse the order
                if keep == "last":
                    descending_list = extend_bool(True, len(order_by))
                    order_cols = [
                        col(name).desc().nulls_last() if desc else col(name).asc().nulls_first()
                        for name, desc in zip(order_by, descending_list)
                    ]
                else:
                    order_cols = [col(name).asc().nulls_first() for name in order_by]
                window_spec = window_spec.order_by(order_cols)
            
            # Choose the appropriate window function
            if keep == "none":
                # Use count to identify rows that appear only once
                window_expr = F.count("*").over(window_spec)
            else:
                # Use row_number to identify the first/last occurrence
                window_expr = F.row_number().over(window_spec)
            
            # Add the window expression as a temporary column
            df_with_index = self._native_frame.with_column(tmp_name, window_expr)
            
            # Filter to keep only the desired rows
            if keep == "none":
                # Keep only rows where count == 1 (unique rows)
                filtered = df_with_index.filter(col(tmp_name) == 1)
            else:
                # Keep only rows where row_number == 1 (first/last occurrence)
                filtered = df_with_index.filter(col(tmp_name) == 1)
            
            # Drop the temporary column
            result = filtered.drop(tmp_name)
            
            return self._with_native(result)
            
        except Exception as e:
            msg = f"Failed to get unique rows from Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        """Unpivot a DataFrame from wide to long format.
        
        This method transforms columns into rows, converting a wide-format DataFrame
        into a long-format DataFrame. It's the inverse operation of pivot.
        
        Arguments:
            on: Column names to unpivot. If None, unpivot all columns except those in index.
            index: Column names to keep as identifier variables. If None, no columns are kept as identifiers.
            variable_name: Name for the variable column that will contain the unpivoted column names.
            value_name: Name for the value column that will contain the unpivoted values.
            
        Returns:
            A new SnowflakeLazyFrame in long format.
            
        Raises:
            NotImplementedError: If variable_name or value_name is an empty string.
            ValueError: If column names don't exist or if there are no columns to unpivot.
            RuntimeError: If the unpivot operation fails.
        """
        try:
            # Validate that variable_name and value_name are not empty
            if variable_name == "":
                msg = "`variable_name` cannot be empty string for Snowflake backend."
                raise NotImplementedError(msg)
            
            if value_name == "":
                msg = "`value_name` cannot be empty string for Snowflake backend."
                raise NotImplementedError(msg)
            
            # Determine index and on columns
            index_ = [] if index is None else list(index)
            on_ = [c for c in self.columns if c not in index_] if on is None else list(on)
            
            # Validate index columns exist
            if index_ and (error := self._check_columns_exist(index_)):
                raise error
            
            # Validate on columns exist
            if on and (error := self._check_columns_exist(on)):
                raise error
            
            # Check that there are columns to unpivot
            if not on_:
                msg = "No columns to unpivot. Either specify 'on' or ensure 'index' doesn't include all columns."
                raise ValueError(msg)
            
            # Snowpark's unpivot method signature:
            # unpivot(value_column, name_column, column_list)
            # where:
            # - value_column: name for the column that will hold the values
            # - name_column: name for the column that will hold the column names
            # - column_list: list of columns to unpivot
            
            result = self._native_frame.unpivot(value_name, variable_name, on_)
            
            # Select columns in the correct order: index columns first, then variable and value
            final_columns = [*index_, variable_name, value_name]
            result = result.select(*final_columns)
            
            return self._with_native(result)
            
        except Exception as e:
            msg = f"Failed to unpivot Snowflake DataFrame: {e}"
            raise RuntimeError(msg) from e

    def collect(
        self, backend: _EagerAllowedImpl | None, **kwargs: Any
    ) -> CompliantDataFrameAny:
        """Execute the query and collect results into an eager DataFrame.
        
        This method executes the lazy Snowflake query and materializes the results
        into an eager DataFrame using the specified backend.
        
        Arguments:
            backend: The backend to use for the result DataFrame. Options:
                - None or Implementation.PANDAS: Return a pandas DataFrame
                - Implementation.PYARROW: Return a PyArrow Table
            **kwargs: Additional keyword arguments (unused, for compatibility).
            
        Returns:
            A compliant eager DataFrame with the query results.
            
        Raises:
            RuntimeError: If the query execution fails.
            ValueError: If an unsupported backend is specified.
        """
        try:
            if backend is None or backend is Implementation.PANDAS:
                from narwhals._pandas_like.dataframe import PandasLikeDataFrame

                # Use Snowpark's to_pandas() to execute query and get pandas DataFrame
                pandas_df = self._native_frame.to_pandas()
                
                return PandasLikeDataFrame(
                    pandas_df,
                    implementation=Implementation.PANDAS,
                    validate_backend_version=True,
                    version=self._version,
                    validate_column_names=True,
                )

            if backend is Implementation.PYARROW:
                from narwhals._arrow.dataframe import ArrowDataFrame

                # Use Snowpark's to_arrow() to execute query and get PyArrow Table
                arrow_table = self._native_frame.to_arrow()
                
                return ArrowDataFrame(
                    arrow_table,
                    validate_backend_version=True,
                    version=self._version,
                    validate_column_names=True,
                )

            msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
            
        except Exception as e:
            msg = f"Failed to execute Snowflake query and collect results: {e}"
            raise RuntimeError(msg) from e
