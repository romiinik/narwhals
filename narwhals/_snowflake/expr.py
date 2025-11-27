from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from narwhals._sql.expr import SQLExpr
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._snowflake.dataframe import SnowflakeLazyFrame
    from narwhals._snowflake.expr_dt import SnowflakeExprDateTimeNamespace
    from narwhals._snowflake.expr_str import SnowflakeExprStringNamespace
    from narwhals._snowflake.namespace import SnowflakeNamespace
    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals._utils import _LimitedContext, NoDefault
    from narwhals.typing import ClosedInterval, FillNullStrategy, IntoDType

    SnowflakeWindowFunction = WindowFunction[SnowflakeLazyFrame, SnowparkColumnT]


class SnowflakeExpr(SQLExpr["SnowflakeLazyFrame", "SnowparkColumnT"]):
    """Narwhals Expression wrapper for Snowpark Column.
    
    This class wraps Snowpark Column expressions and provides the Narwhals
    expression interface for building queries on Snowflake.
    """

    _implementation = Implementation.SNOWFLAKE

    def __init__(
        self,
        call: EvalSeries[SnowflakeLazyFrame, SnowparkColumnT],
        window_function: SnowflakeWindowFunction | None = None,
        *,
        evaluate_output_names: EvalNames[SnowflakeLazyFrame],
        alias_output_names: AliasNames | None,
        version: Version,
        implementation: Implementation = Implementation.SNOWFLAKE,
    ) -> None:
        """Initialize a SnowflakeExpr.
        
        Arguments:
            call: Function that evaluates the expression on a dataframe.
            window_function: Optional window function for windowed operations.
            evaluate_output_names: Function to evaluate output column names.
            alias_output_names: Optional function to alias output names.
            version: The Narwhals version for compatibility.
            implementation: The backend implementation (default: SNOWFLAKE).
        """
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version
        self._window_function: SnowflakeWindowFunction | None = window_function

    def __narwhals_namespace__(self) -> SnowflakeNamespace:  # pragma: no cover
        """Get the Snowflake namespace.
        
        Returns:
            A SnowflakeNamespace instance.
        """
        from narwhals._snowflake.namespace import SnowflakeNamespace

        return SnowflakeNamespace(version=self._version)

    def broadcast(self) -> Self:
        """Broadcast the expression to all rows.
        
        For Snowflake, broadcasting is achieved by using a window function
        over a constant partition.
        
        Returns:
            A new SnowflakeExpr with broadcasting applied.
        """
        from narwhals._snowflake.utils import lit

        return self.over([lit(1)], [])

    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[SnowflakeLazyFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        """Create an expression from column names.
        
        Arguments:
            evaluate_column_names: Function that returns column names from a dataframe.
            context: Limited context containing version information.
            
        Returns:
            A new SnowflakeExpr that selects the specified columns.
        """
        from narwhals._snowflake.utils import col

        def func(df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            return [col(name) for name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(
        cls, *column_indices: int, context: _LimitedContext
    ) -> Self:
        """Create an expression from column indices.
        
        Arguments:
            column_indices: The indices of columns to select.
            context: Limited context containing version information.
            
        Returns:
            A new SnowflakeExpr that selects columns by index.
        """
        from narwhals._snowflake.utils import col

        def func(df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            columns = df.columns
            return [col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def _alias_native(cls, expr: SnowparkColumnT, name: str) -> SnowparkColumnT:
        """Alias a native Snowpark Column.
        
        Arguments:
            expr: The Snowpark Column to alias.
            name: The new name for the column.
            
        Returns:
            The aliased Snowpark Column.
        """
        return expr.alias(name)

    # Arithmetic operations
    def __neg__(self) -> Self:
        """Negate the expression.
        
        Returns:
            A new SnowflakeExpr with negation applied.
        """
        return self._with_elementwise(lambda expr: -expr)

    def __invert__(self) -> Self:
        """Invert the expression (bitwise NOT for integers, logical NOT for booleans).
        
        Returns:
            A new SnowflakeExpr with inversion applied.
        """
        return self._with_elementwise(lambda expr: ~expr)

    def __radd__(self, other: Self) -> Self:
        """Right-hand addition (other + self).
        
        Arguments:
            other: The left-hand operand.
            
        Returns:
            A new SnowflakeExpr with addition applied.
        """
        return (self + other).alias("literal")  # type: ignore[return-value]

    def __rmul__(self, other: Self) -> Self:
        """Right-hand multiplication (other * self).
        
        Arguments:
            other: The left-hand operand.
            
        Returns:
            A new SnowflakeExpr with multiplication applied.
        """
        return (self * other).alias("literal")  # type: ignore[return-value]

    # Comparison operations
    # The following comparison operations are inherited from SQLExpr:
    # __eq__, __ne__, __lt__, __le__, __gt__, __ge__
    # These work with Snowpark Column objects which support Python's comparison operators natively.

    # Logical operations
    # The following logical operations are inherited from SQLExpr:
    # __and__, __or__
    # Combined with __invert__ (implemented above), these provide full boolean logic support.
    # Snowpark Column objects support these operators natively for boolean operations.

    # Aggregation functions
    # Most aggregation functions (sum, mean, min, max, count, std, var, median) are inherited
    # from SQLExpr and work with Snowpark's function API.
    
    def quantile(
        self, quantile: float, interpolation: str
    ) -> Self:
        """Calculate the quantile of the expression.
        
        Arguments:
            quantile: The quantile to calculate (between 0 and 1).
            interpolation: Interpolation method (only 'linear' is supported).
            
        Returns:
            A new SnowflakeExpr with the quantile aggregation.
            
        Raises:
            ValueError: If quantile is not between 0 and 1.
            NotImplementedError: If interpolation method is not 'linear'.
        """
        # Validate quantile value
        if not 0 <= quantile <= 1:
            msg = f"Quantile must be between 0 and 1, got {quantile}"
            raise ValueError(msg)
            
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            if interpolation == "linear":
                # Snowflake uses PERCENTILE_CONT for linear interpolation
                # percentile_cont takes the percentile value as first argument
                from snowflake.snowpark import functions as F
                return F.percentile_cont(quantile).within_group(expr)
            msg = "Only linear interpolation methods are supported for Snowflake quantile."
            raise NotImplementedError(msg)

        return self._with_callable(func)

    def null_count(self) -> Self:
        """Count the number of null values in the expression.
        
        Returns:
            A new SnowflakeExpr with the null count aggregation.
        """
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            # Count nulls by summing is_null() cast to int
            from snowflake.snowpark import functions as F
            return F.sum(expr.is_null().cast("int"))

        return self._with_callable(func)

    def _count_star(self) -> SnowparkColumnT:
        """Create a COUNT(*) expression.
        
        Returns:
            A Snowpark Column representing COUNT(*).
        """
        from snowflake.snowpark import functions as F
        return F.count("*")

    def _first(self, expr: SnowparkColumnT, *order_by: str) -> SnowparkColumnT:
        """Get the first value in a group, optionally ordered.
        
        This method is used by the base class's first() method which handles
        the window function logic. This should return just the aggregation
        function that will be wrapped in a window expression.
        
        Arguments:
            expr: The expression to get the first value from.
            order_by: Optional column names to order by (handled by caller).
            
        Returns:
            A Snowpark Column representing the first value aggregation.
        """
        # For Snowflake, we can use FIRST_VALUE or ANY_VALUE
        # The ordering is handled by the caller via _window_expression
        from snowflake.snowpark import functions as F
        
        if order_by:
            # When order_by is specified, use first_value
            # The actual ordering will be applied by _window_expression
            return F.first_value(expr)
        else:
            # Without ordering, use any_value
            return F.any_value(expr)

    def _last(self, expr: SnowparkColumnT, *order_by: str) -> SnowparkColumnT:
        """Get the last value in a group, optionally ordered.
        
        This method is used by the base class's last() method which handles
        the window function logic. This should return just the aggregation
        function that will be wrapped in a window expression.
        
        Arguments:
            expr: The expression to get the last value from.
            order_by: Optional column names to order by (handled by caller).
            
        Returns:
            A Snowpark Column representing the last value aggregation.
        """
        # For Snowflake, we can use LAST_VALUE or ANY_VALUE
        # The ordering is handled by the caller via _window_expression
        from snowflake.snowpark import functions as F
        
        if order_by:
            # When order_by is specified, use last_value
            # The actual ordering will be applied by _window_expression
            return F.last_value(expr)
        else:
            # Without ordering, use any_value
            return F.any_value(expr)

    # Fill and replace operations
    def fill_null(
        self, value: Self | None, strategy: FillNullStrategy | None, limit: int | None
    ) -> Self:
        """Fill null values with a value or strategy.
        
        Arguments:
            value: Value to fill nulls with (if strategy is None).
            strategy: Strategy to use for filling ('forward' or 'backward').
            limit: Maximum number of consecutive nulls to fill.
            
        Returns:
            A new SnowflakeExpr with nulls filled.
            
        Raises:
            ValueError: If both value and strategy are None, or if limit is negative.
        """
        from narwhals._compliant.window import WindowInputs
        
        # Validate parameters
        if value is None and strategy is None:
            msg = "Either 'value' or 'strategy' must be provided to fill_null()"
            raise ValueError(msg)
            
        if limit is not None and limit < 0:
            msg = f"Limit must be non-negative, got {limit}"
            raise ValueError(msg)
        
        if strategy is not None:
            # Validate strategy value
            valid_strategies = {"forward", "backward"}
            if strategy not in valid_strategies:
                msg = f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
                raise ValueError(msg)
                
            def _fill_with_strategy(
                df: SnowflakeLazyFrame, inputs: WindowInputs[SnowparkColumnT]
            ) -> Sequence[SnowparkColumnT]:
                from snowflake.snowpark import functions as F
                
                # Snowflake uses FIRST_VALUE for forward fill and LAST_VALUE for backward fill
                # with IGNORE NULLS option
                fill_func_name = "last_value" if strategy == "forward" else "first_value"
                fill_func = getattr(F, fill_func_name)
                
                rows_start, rows_end = (
                    (-limit if limit is not None else None, 0)
                    if strategy == "forward"
                    else (0, limit)
                )
                
                return [
                    self._window_expression(
                        fill_func(expr, ignore_nulls=True),
                        inputs.partition_by,
                        inputs.order_by,
                        rows_start=rows_start,
                        rows_end=rows_end,
                    )
                    for expr in self(df)
                ]
            
            return self._with_window_function(_fill_with_strategy)
        
        def _fill_constant(expr: SnowparkColumnT, value: SnowparkColumnT) -> SnowparkColumnT:
            return self._coalesce(expr, value)
        
        assert value is not None  # noqa: S101
        return self._with_elementwise(_fill_constant, value=value)

    def replace_strict(
        self,
        default: Self | NoDefault,
        old: Sequence[Any],
        new: Sequence[Any],
        *,
        return_dtype: IntoDType | None,
    ) -> Self:
        """Replace values strictly, requiring all old values to be present.
        
        Arguments:
            default: Default value to use when old value is not in the mapping.
            old: Sequence of values to replace.
            new: Sequence of replacement values.
            return_dtype: Optional dtype to cast the result to.
            
        Returns:
            A new SnowflakeExpr with values replaced.
            
        Raises:
            ValueError: If default is not provided.
        """
        from narwhals._utils import no_default
        
        if default is no_default:
            msg = "`replace_strict` requires an explicit value for `default` for Snowflake backend."
            raise ValueError(msg)
        
        def func(df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            from snowflake.snowpark import functions as F
            
            default_col = df._evaluate_single_output_expr(default)
            
            # Build a CASE WHEN expression for each old->new mapping
            results = []
            for expr in self(df):
                # Start with the default
                result = default_col
                
                # Build the CASE WHEN chain in reverse order
                for old_val, new_val in zip(reversed(list(old)), reversed(list(new))):
                    result = self._when(
                        expr == self._lit(old_val),
                        self._lit(new_val),
                        result
                    )
                
                results.append(result)
            
            if return_dtype:
                from narwhals._snowflake.utils import narwhals_to_native_dtype
                native_dtype = narwhals_to_native_dtype(return_dtype, self._version)
                return [res.cast(native_dtype) for res in results]
            
            return results
        
        return self.__class__(
            func,
            None,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
            implementation=self._implementation,
        )

    # Conditional operations
    def is_in(self, other: Sequence[Any]) -> Self:
        """Check if values are in a given sequence.
        
        Arguments:
            other: Sequence of values to check membership against.
            
        Returns:
            A new SnowflakeExpr with boolean values indicating membership.
        """
        from snowflake.snowpark import functions as F
        
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            # Snowflake uses isin() method on Column
            return expr.isin(list(other))
        
        return self._with_elementwise(func)

    def is_between(
        self, lower_bound: Self, upper_bound: Self, closed: ClosedInterval
    ) -> Self:
        """Check if values are between bounds.
        
        Arguments:
            lower_bound: Lower bound expression.
            upper_bound: Upper bound expression.
            closed: Which bounds are inclusive ('left', 'right', 'both', 'none').
            
        Returns:
            A new SnowflakeExpr with boolean values indicating if values are in range.
            
        Raises:
            ValueError: If closed parameter has an invalid value.
        """
        # Validate closed parameter early
        valid_closed = {"left", "right", "both", "none"}
        if closed not in valid_closed:
            msg = f"Invalid value for `closed`: {closed}. Must be one of {valid_closed}"
            raise ValueError(msg)
            
        def func(
            expr: SnowparkColumnT,
            lower_bound: SnowparkColumnT,
            upper_bound: SnowparkColumnT,
        ) -> SnowparkColumnT:
            if closed == "left":
                return (expr >= lower_bound) & (expr < upper_bound)
            elif closed == "right":
                return (expr > lower_bound) & (expr <= upper_bound)
            elif closed == "none":
                return (expr > lower_bound) & (expr < upper_bound)
            else:  # closed == "both"
                return (expr >= lower_bound) & (expr <= upper_bound)
        
        return self._with_elementwise(
            func, lower_bound=lower_bound, upper_bound=upper_bound
        )

    # Type checking and casting
    def cast(self, dtype: IntoDType) -> Self:
        """Cast the expression to a different data type.
        
        Arguments:
            dtype: The target data type to cast to.
            
        Returns:
            A new SnowflakeExpr with the cast applied.
            
        Raises:
            NotImplementedError: If the dtype is not supported for Snowflake.
            RuntimeError: If the cast operation fails.
        """
        from narwhals._snowflake.utils import narwhals_to_native_dtype
        
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            try:
                native_dtype = narwhals_to_native_dtype(dtype, self._version)
                return expr.cast(native_dtype)
            except NotImplementedError:
                raise
            except Exception as e:
                msg = f"Failed to cast expression to {dtype}: {e}"
                raise RuntimeError(msg) from e
        
        return self._with_elementwise(func)

    def is_not_null(self) -> Self:
        """Check if the expression is not null.
        
        Returns:
            A new SnowflakeExpr with boolean values indicating non-null elements.
        """
        return self._with_elementwise(lambda expr: expr.is_not_null())

    def is_nan(self) -> Self:
        """Check if the expression contains NaN values.
        
        Returns:
            A new SnowflakeExpr with boolean values indicating NaN elements.
        """
        from snowflake.snowpark import functions as F
        
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            # For Snowflake, we need to check if the value is not null first
            # then check if it's NaN
            return self._when(expr.is_not_null(), F.is_nan(expr), self._lit(False))
        
        return self._with_elementwise(func)

    def is_finite(self) -> Self:
        """Check if the expression contains finite values.
        
        Returns:
            A new SnowflakeExpr with boolean values indicating finite elements.
        """
        from snowflake.snowpark import functions as F
        
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            # A value is finite if it's not null, not NaN, and not infinite
            # In Snowflake, we check: not null AND not NaN AND not infinite
            is_not_nan = ~F.is_nan(expr)
            is_not_inf = ~F.is_infinite(expr)
            return self._when(
                expr.is_not_null(),
                is_not_nan & is_not_inf,
                self._lit(False)
            )
        
        return self._with_elementwise(func)

    # Namespaces
    @property
    def str(self) -> SnowflakeExprStringNamespace:
        """Access string operations namespace.
        
        Returns:
            A SnowflakeExprStringNamespace for string operations.
        """
        from narwhals._snowflake.expr_str import SnowflakeExprStringNamespace
        
        return SnowflakeExprStringNamespace(self)

    @property
    def dt(self) -> SnowflakeExprDateTimeNamespace:
        """Access datetime operations namespace.
        
        Returns:
            A SnowflakeExprDateTimeNamespace for datetime operations.
        """
        from narwhals._snowflake.expr_dt import SnowflakeExprDateTimeNamespace
        
        return SnowflakeExprDateTimeNamespace(self)
