# Function
# Defines a function in your own code the model can choose to call. Learn more about function calling.

# name: string
# The name of the function to call.

# parameters: map[unknown]
# A JSON schema object describing the parameters of the function.

# strict: boolean
# Whether to enforce strict parameter validation. Default true.

# type: "function"
# The type of the function tool. Always function.

# defer_loading: optional boolean
# Whether this function is deferred and loaded via tool search.

# description: optional string
# A description of the function. Used by the model to determine whether or not to call the function.

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OPEN_PROJECT_FILE_TOOL = {
    "type": "function",
    "name": "open_project_file",
    "description": "Open and read a UTF-8 text file located in the project directory or one of its subdirectories. This tool cannot read binary files, directories, missing files, or files outside the project.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from the project root, for example main.py or data/session.json. Parent-directory paths are not allowed.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


TOOLS = [
    OPEN_PROJECT_FILE_TOOL,
]


def open_project_file(path):
    requested_path = Path(path)
    if requested_path.is_absolute():
        return {
            "error": "Use a relative path inside the project directory.",
        }

    file_path = (PROJECT_ROOT / requested_path).resolve()

    try:
        file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "error": "That path is outside the project directory.",
        }

    if not file_path.exists():
        return {
            "error": f"File not found: {path}",
        }

    if not file_path.is_file():
        return {
            "error": f"Path is not a file: {path}",
        }

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "error": f"File is not valid UTF-8 text: {path}",
        }
    except OSError as error:
        return {
            "error": str(error),
        }

    return {
        "path": str(file_path.relative_to(PROJECT_ROOT)),
        "content": content,
    }


AVAILABLE_FUNCTIONS = {
    "open_project_file": open_project_file,
}
