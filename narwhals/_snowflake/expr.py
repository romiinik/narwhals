from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from narwhals._sql.expr import SQLExpr
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._snowflake.dataframe import SnowflakeLazyFrame
    from narwhals._snowflake.namespace import SnowflakeNamespace
    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals._utils import _LimitedContext

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
