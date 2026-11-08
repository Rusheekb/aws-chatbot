"""Tools the model can ask us to run: the Python functions, their specs, and a dispatcher."""
from rag import search


def search_documents(query):
    """Search Learn2Lead's documents and return the most relevant passages."""
    return {"results": search(query)}


# Which Python function runs for each tool name the model can ask for
TOOL_FUNCTIONS = {
    "search_documents": search_documents,
}

# What we tell the model about each tool (sent with every request)
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "search_documents",
                "description": (
                    "Search Learn2Lead's website and Help Center: what Learn2Lead offers, subjects, "
                    "pricing and hours, how tutoring works, and step-by-step guides for students, "
                    "parents and tutors. Use it for any question about Learn2Lead. Answer only from "
                    "the passages returned, and say so if they don't contain the answer."
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
