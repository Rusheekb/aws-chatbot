"""Tools the model can ask us to run: the Python functions, their specs, and a dispatcher."""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rag import search


def get_current_time(timezone="UTC"):
    """Return the current date and time in the given IANA timezone."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        raise ValueError(f"Unknown timezone: {timezone}")
    return {"timezone": timezone, "datetime": now.strftime("%A, %B %d, %Y %I:%M %p")}


def calculate(a, b, operation):
    """Do exact arithmetic on two numbers."""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
    elif operation == "power":
        result = a ** b
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return {"result": result}


def search_documents(query):
    """Search the user's documents and return the most relevant passages."""
    return {"results": search(query)}


# Which Python function runs for each tool name the model can ask for
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_documents": search_documents,
}

# What we tell the model about each tool (sent with every request)
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_current_time",
                "description": "Get the current date and time. Use this whenever the user asks about the current time, date, or day.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "IANA timezone name, e.g. 'America/New_York' or 'Asia/Tokyo'. Defaults to UTC.",
                            }
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "calculate",
                "description": "Do exact arithmetic on two numbers. Use this for any math instead of calculating in your head.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "The first number"},
                            "b": {"type": "number", "description": "The second number"},
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide", "power"],
                            },
                        },
                        "required": ["a", "b", "operation"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_documents",
                "description": (
                    "Search the user's own documents (notes, handbooks, policies) for passages "
                    "related to a question. Use this whenever the answer might be in the user's "
                    "documents rather than general knowledge. Base your answer on the passages "
                    "returned, mention the source file, and say so if they don't contain the answer."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for, written as a short question or phrase.",
                            }
                        },
                        "required": ["query"],
                    }
                },
            }
        },
    ]
}


def run_tool(tool_call):
    """Run one tool the model asked for and return a toolResult block."""
    name = tool_call["name"]
    function = TOOL_FUNCTIONS.get(name)
    try:
        if function is None:
            raise ValueError(f"No tool named {name}")
        result = function(**tool_call["input"])
        content = [{"json": result}]
        status = "success"
    except (TypeError, ValueError) as error:
        content = [{"text": str(error)}]
        status = "error"
    return {
        "toolResult": {
            "toolUseId": tool_call["toolUseId"],
            "content": content,
            "status": status,
        }
    }
