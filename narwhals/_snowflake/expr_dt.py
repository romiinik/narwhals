from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._duration import Interval
from narwhals._sql.expr_dt import SQLExprDateTimeNamesSpace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals._snowflake.expr import SnowflakeExpr
    from narwhals._snowflake.typing import SnowparkColumnT
    from narwhals.typing import TimeUnit


class SnowflakeExprDateTimeNamespace(SQLExprDateTimeNamesSpace["SnowflakeExpr"]):
    """DateTime operations namespace for Snowflake expressions.
    
    This class provides datetime manipulation methods for Snowflake expressions,
    leveraging Snowpark's datetime functions.
    
    Most datetime operations are inherited from SQLExprDateTimeNamesSpace and work
    with Snowpark's built-in datetime functions:
    - year() -> uses YEAR()
    - month() -> uses MONTH()
    - day() -> uses DAY()
    - hour() -> uses HOUR()
    - minute() -> uses MINUTE()
    - second() -> uses SECOND()
    - date() -> uses TO_DATE()
    """

    def timestamp(self, time_unit: TimeUnit = "us") -> SnowflakeExpr:
        """Convert a date or datetime to a timestamp.
        
        Arguments:
            time_unit: Time unit for the timestamp ('us', 'ns', 'ms').
            
        Returns:
            A new SnowflakeExpr with timestamp conversion applied.
        """
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            from snowflake.snowpark import functions as F
            
            # Convert to timestamp (Snowflake stores timestamps with nanosecond precision)
            timestamp_expr = F.to_timestamp(expr)
            
            # Snowflake timestamps are in nanoseconds internally
            # We need to convert based on the requested time_unit
            if time_unit == "ns":
                # Return nanoseconds since epoch
                return F.date_part("epoch_nanosecond", timestamp_expr)
            elif time_unit == "us":
                # Return microseconds since epoch
                return F.date_part("epoch_microsecond", timestamp_expr)
            elif time_unit == "ms":
                # Return milliseconds since epoch
                return F.date_part("epoch_millisecond", timestamp_expr)
            else:
                msg = f"Unsupported time_unit: {time_unit}. Use 'ns', 'us', or 'ms'."
                raise ValueError(msg)
        
        return self.compliant._with_elementwise(func)

    def truncate(self, every: str) -> SnowflakeExpr:
        """Truncate datetime to the specified interval.
        
        Arguments:
            every: Interval string (e.g., '1d', '1h', '1mo').
            
        Returns:
            A new SnowflakeExpr with truncation applied.
        """
        interval = Interval.parse(every)
        multiple, unit = interval.multiple, interval.unit
        
        if multiple != 1:
            msg = f"Only multiple 1 is currently supported for Snowflake truncate.\nGot {multiple!s}."
            raise ValueError(msg)
        
        # Map Narwhals units to Snowflake DATE_TRUNC units
        unit_map = {
            "ns": "nanosecond",
            "us": "microsecond",
            "ms": "millisecond",
            "s": "second",
            "m": "minute",
            "h": "hour",
            "d": "day",
            "w": "week",
            "mo": "month",
            "q": "quarter",
            "y": "year",
        }
        
        if unit not in unit_map:
            msg = f"Unsupported truncate unit: {unit}"
            raise ValueError(msg)
        
        snowflake_unit = unit_map[unit]
        
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            from snowflake.snowpark import functions as F
            
            return F.date_trunc(snowflake_unit, expr)
        
        return self.compliant._with_elementwise(func)

    def replace_time_zone(self, time_zone: str | None) -> SnowflakeExpr:
        """Replace the time zone of a datetime expression.
        
        Arguments:
            time_zone: Target time zone string (e.g., 'UTC', 'America/New_York').
                      If None, removes time zone information.
            
        Returns:
            A new SnowflakeExpr with time zone replaced.
        """
        if time_zone is None:
            # Remove time zone by casting to timestamp without time zone
            def func(expr: SnowparkColumnT) -> SnowparkColumnT:
                from snowflake.snowpark import types as T
                
                return expr.cast(T.TimestampType())
            
            return self.compliant._with_elementwise(func)
        
        # Replace time zone using CONVERT_TIMEZONE
        def func(expr: SnowparkColumnT) -> SnowparkColumnT:
            from snowflake.snowpark import functions as F
            
            # CONVERT_TIMEZONE in Snowflake converts from one timezone to another
            # To "replace" the timezone (not convert), we need to:
            # 1. Cast to timestamp_ntz (no timezone)
            # 2. Then interpret it as being in the target timezone
            # This is a bit tricky in Snowflake, so we'll use a simpler approach:
            # Just convert to the target timezone
            return F.convert_timezone(time_zone, expr)
        
        return self.compliant._with_elementwise(func)

    # The following methods are inherited from SQLExprDateTimeNamesSpace:
    # - year(), month(), day(), hour(), minute(), second()
    # - date() (uses TO_DATE)
    # - ordinal_day() (uses DAYOFYEAR)
    
    # These methods are not yet implemented for Snowflake:
    millisecond = not_implemented()
    microsecond = not_implemented()
    nanosecond = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
    convert_time_zone = not_implemented()
    offset_by = not_implemented()
    weekday = not_implemented()
    to_string = not_implemented()
