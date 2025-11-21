from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from narwhals._snowflake.utils import col, lit
from narwhals._sql.expr import SQLExpr
from narwhals._utils import Implementation, Version, extend_bool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._compliant.window import WindowInputs
    from narwhals._snowflake.dataframe import SnowflakeLazyFrame
    from narwhals._snowflake.namespace import SnowflakeNamespace
    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals._utils import _LimitedContext

    SnowflakeWindowFunction = WindowFunction[SnowflakeLazyFrame, SnowparkColumnT]
    SnowflakeWindowInputs = WindowInputs[SnowparkColumnT]


class SnowflakeExpr(SQLExpr["SnowflakeLazyFrame", "SnowparkColumnT"]):
    """Narwhals expression wrapper for Snowpark Column.
    
    This class wraps Snowpark Column operations and provides the Narwhals
    expression interface for lazy query building on Snowflake.
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
            implementation: The implementation type (SNOWFLAKE).
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
        
        Returns:
            A new expression that broadcasts to all rows.
        """
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
            evaluate_column_names: Function that returns column names.
            context: The limited context containing version information.
            
        Returns:
            A new SnowflakeExpr instance.
        """
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
            context: The limited context containing version information.
            
        Returns:
            A new SnowflakeExpr instance.
        """
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

    def _count_star(self) -> SnowparkColumnT:
        """Create a COUNT(*) expression.
        
        Returns:
            A Snowpark Column representing COUNT(*).
        """
        from snowflake.snowpark import functions as F

        return F.count(lit(1))

    def _window_expression(
        self,
        expr: SnowparkColumnT,
        partition_by: Sequence[str | SnowparkColumnT] = (),
        order_by: Sequence[str | SnowparkColumnT] = (),
        rows_start: int | None = None,
        rows_end: int | None = None,
        *,
        descending: Sequence[bool] | None = None,
        nulls_last: Sequence[bool] | None = None,
    ) -> SnowparkColumnT:
        """Create a window expression.
        
        Arguments:
            expr: The expression to apply the window to.
            partition_by: Columns to partition by.
            order_by: Columns to order by.
            rows_start: Starting row offset for the window frame.
            rows_end: Ending row offset for the window frame.
            descending: Whether to sort in descending order.
            nulls_last: Whether to put nulls last in ordering.
            
        Returns:
            A Snowpark Column with window specification.
        """
        from snowflake.snowpark import Window

        # Convert string column names to Column objects
        partition_cols = [
            col(c) if isinstance(c, str) else c for c in partition_by
        ]
        order_cols = [
            col(c) if isinstance(c, str) else c for c in order_by
        ]

        # Create window specification
        window_spec = Window.partition_by(*partition_cols) if partition_cols else Window

        # Add ordering if specified
        if order_cols:
            descending = descending or extend_bool(False, len(order_cols))
            nulls_last = nulls_last or extend_bool(False, len(order_cols))
            
            for order_col, desc, null_last in zip(order_cols, descending, nulls_last):
                if desc:
                    order_col = order_col.desc()
                else:
                    order_col = order_col.asc()
                
                if null_last:
                    order_col = order_col.nulls_last()
                else:
                    order_col = order_col.nulls_first()
                
                window_spec = window_spec.order_by(order_col)

        # Add row frame if specified
        if rows_start is not None or rows_end is not None:
            start = rows_start if rows_start is not None else Window.UNBOUNDED_PRECEDING
            end = rows_end if rows_end is not None else Window.CURRENT_ROW
            window_spec = window_spec.rows_between(start, end)

        return expr.over(window_spec)

    def _first(self, expr: SnowparkColumnT, *order_by: str) -> SnowparkColumnT:
        """Get the first value ordered by specified columns.
        
        Arguments:
            expr: The expression to get the first value from.
            order_by: Columns to order by.
            
        Returns:
            A Snowpark Column representing the first value.
        """
        from snowflake.snowpark import functions as F

        if order_by:
            # Use first_value with ordering
            return F.first_value(expr)
        return F.first_value(expr)

    def _last(self, expr: SnowparkColumnT, *order_by: str) -> SnowparkColumnT:
        """Get the last value ordered by specified columns.
        
        Arguments:
            expr: The expression to get the last value from.
            order_by: Columns to order by.
            
        Returns:
            A Snowpark Column representing the last value.
        """
        from snowflake.snowpark import functions as F

        if order_by:
            # Use last_value with ordering
            return F.last_value(expr)
        return F.last_value(expr)

    def __invert__(self) -> Self:
        """Invert a boolean expression (NOT operation).
        
        Returns:
            A new expression with the inverted boolean value.
        """
        import operator

        invert = operator.invert
        return self._with_elementwise(invert)
