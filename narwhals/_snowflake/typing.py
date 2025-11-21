from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from snowflake.snowpark import Column as SnowparkColumn
    from snowflake.snowpark import DataFrame as SnowparkDataFrame
    from snowflake.snowpark import Session as SnowparkSession
    from snowflake.snowpark.types import DataType as SnowparkDataType

    SnowparkColumnT = SnowparkColumn
    SnowparkDataFrameT = SnowparkDataFrame
    SnowparkSessionT = SnowparkSession
    SnowparkDataTypeT = SnowparkDataType
else:
    SnowparkColumnT = Any
    SnowparkDataFrameT = Any
    SnowparkSessionT = Any
    SnowparkDataTypeT = Any
    ModuleType = Any
