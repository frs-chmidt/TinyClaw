"""
Plugin: time
Gets the current time
"""

import datetime


@mcp.tool()
def get_current_date_time() -> str:
    """
    Get the current system time and date (datetime) in isoformat

    Returns:
    - string timestamp in isoformat
    """
    return str(datetime.datetime.now().isoformat())
