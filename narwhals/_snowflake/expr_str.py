from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._sql.expr_str import SQLExprStringNamespace

if TYPE_CHECKING:
    from narwhals._snowflake.expr import SnowflakeExpr


class SnowflakeExprStringNamespace(SQLExprStringNamespace["SnowflakeExpr"]):
    """String operations namespace for Snowflake expressions.
    
    This class provides string manipulation methods for Snowflake expressions,
    leveraging Snowpark's string functions.
    
    Most string operations are inherited from SQLExprStringNamespace and work
    with Snowpark's built-in string functions:
    - len_chars() -> uses LENGTH()
    - to_uppercase() -> uses UPPER()
    - to_lowercase() -> uses LOWER()
    - strip_chars() -> uses TRIM()
    - starts_with() -> uses STARTSWITH()
    - ends_with() -> uses ENDSWITH()
    - contains() -> uses CONTAINS() or REGEXP_MATCHES()
    - replace() -> uses REPLACE() or REGEXP_REPLACE()
    - slice() -> uses SUBSTR()
    - split() -> uses SPLIT()
    """
