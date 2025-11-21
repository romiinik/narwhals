from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._sql.expr import SQLExpr
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        WindowFunction,
    )
    from narwhals._snowflake.dataframe import SnowflakeLazyFrame
    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals._utils import Version

    SnowflakeWindowFunction = WindowFunction[SnowflakeLazyFrame, SnowparkColumnT]


class SnowflakeExpr(SQLExpr["SnowflakeLazyFrame", "SnowparkColumnT"]):
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
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version
        self._window_function: SnowflakeWindowFunction | None = window_function

    def __narwhals_namespace__(self) -> Self:  # pragma: no cover
        from narwhals._snowflake.namespace import SnowflakeNamespace

        return SnowflakeNamespace(version=self._version)  # type: ignore[return-value]
