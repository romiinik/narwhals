from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any

from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._snowflake.dataframe import SnowflakeLazyFrame
from narwhals._snowflake.expr import SnowflakeExpr
from narwhals._snowflake.utils import coalesce, function, lit, narwhals_to_native_dtype, when
from narwhals._sql.namespace import SQLNamespace
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals._utils import Version
    from narwhals.typing import ConcatMethod, IntoDType, NonNestedLiteral, PythonLiteral


class SnowflakeNamespace(
    SQLNamespace[SnowflakeLazyFrame, SnowflakeExpr, Any, "SnowparkColumnT"]
):
    _implementation: Implementation = Implementation.SNOWFLAKE

    def __init__(self, *, version: Version) -> None:
        self._version = version

    @property
    def _expr(self) -> type[SnowflakeExpr]:
        return SnowflakeExpr

    @property
    def _lazyframe(self) -> type[SnowflakeLazyFrame]:
        return SnowflakeLazyFrame

    def _function(self, name: str, *args: SnowparkColumnT | PythonLiteral) -> SnowparkColumnT:  # type: ignore[override]
        return function(name, *args)

    def _lit(self, value: Any) -> SnowparkColumnT:
        return lit(value)

    def _when(
        self,
        condition: SnowparkColumnT,
        value: SnowparkColumnT,
        otherwise: SnowparkColumnT | None = None,
    ) -> SnowparkColumnT:
        if otherwise is None:
            return when(condition, value)
        return when(condition, value).otherwise(otherwise)

    def _coalesce(self, *exprs: SnowparkColumnT) -> SnowparkColumnT:
        return coalesce(*exprs)

    def concat(
        self, items: Iterable[SnowflakeLazyFrame], *, how: ConcatMethod
    ) -> SnowflakeLazyFrame:
        if how == "diagonal":
            msg = "diagonal concat not supported for Snowflake. Please join instead."
            raise NotImplementedError(msg)

        items = list(items)
        native_items = [item._native_frame for item in items]
        schema = items[0].schema
        if not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)

        # Use Snowpark's union_all for vertical concatenation
        result = native_items[0]
        for native_item in native_items[1:]:
            result = result.union_all(native_item)

        return self._lazyframe(result)

    def concat_str(
        self, *exprs: SnowflakeExpr, separator: str, ignore_nulls: bool
    ) -> SnowflakeExpr:
        def func(df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            from snowflake.snowpark import functions as F

            cols: Iterable[SnowparkColumnT] = chain.from_iterable(e(df) for e in exprs)
            cols_list = list(cols)

            if ignore_nulls:
                # Use concat_ws which ignores nulls
                return [F.concat_ws(lit(separator), *cols_list)]
            else:
                # Check for nulls and return null if any column is null
                null_mask = reduce(operator.or_, (c.is_null() for c in cols_list))
                # Cast all columns to string and concatenate
                cols_str = [c.cast("string") for c in cols_list]
                result = cols_str[0]
                for col_str in cols_str[1:]:
                    result = F.concat(result, lit(separator), col_str)
                return [when(null_mask, lit(None)).otherwise(result)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            version=self._version,
        )

    def mean_horizontal(self, *exprs: SnowflakeExpr) -> SnowflakeExpr:
        def func(cols: Iterable[SnowparkColumnT]) -> SnowparkColumnT:
            cols = tuple(cols)
            total = reduce(operator.add, (coalesce(col, lit(0)) for col in cols))
            count = reduce(
                operator.add, (when(col.is_not_null(), lit(1)).otherwise(lit(0)) for col in cols)
            )
            return total / count

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def lit(self, value: NonNestedLiteral, dtype: IntoDType | None) -> SnowflakeExpr:
        def func(df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            if dtype is not None:
                target = narwhals_to_native_dtype(dtype, self._version)
                return [lit(value).cast(target)]
            return [lit(value)]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            version=self._version,
        )

    def len(self) -> SnowflakeExpr:
        def func(_df: SnowflakeLazyFrame) -> list[SnowparkColumnT]:
            from snowflake.snowpark import functions as F

            return [F.count(lit(1))]

        return self._expr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            version=self._version,
        )
