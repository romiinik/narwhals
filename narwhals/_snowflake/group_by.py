from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._sql.group_by import SQLGroupBy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._snowflake.dataframe import SnowflakeLazyFrame
    from narwhals._snowflake.expr import SnowflakeExpr
    from narwhals._snowflake.typing import SnowparkColumnT


class SnowflakeGroupBy(SQLGroupBy["SnowflakeLazyFrame", "SnowflakeExpr", "SnowparkColumnT"]):
    """GroupBy implementation for Snowflake backend.
    
    This class handles grouped aggregations on Snowflake DataFrames using
    Snowpark's group_by() and agg() methods.
    """

    def __init__(
        self,
        df: SnowflakeLazyFrame,
        keys: Sequence[SnowflakeExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        """Initialize a SnowflakeGroupBy.
        
        Arguments:
            df: The SnowflakeLazyFrame to group.
            keys: Column names or expressions to group by.
            drop_null_keys: Whether to drop rows with null values in the grouping keys.
        """
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._compliant_frame = frame.drop_nulls(self._keys) if drop_null_keys else frame

    def agg(self, *exprs: SnowflakeExpr) -> SnowflakeLazyFrame:
        """Perform aggregations on the grouped data.
        
        Arguments:
            exprs: Aggregation expressions to apply.
            
        Returns:
            A new SnowflakeLazyFrame with the aggregated results.
        """
        # Evaluate aggregation expressions
        agg_columns = list(self._evaluate_exprs(exprs))
        
        # Use Snowpark's group_by().agg() pattern
        result = self.compliant.native.group_by(self._keys).agg(*agg_columns)
        
        # Rename the grouping keys to their output names if needed
        return self.compliant._with_native(result).rename(
            dict(zip(self._keys, self._output_key_names))
        )
