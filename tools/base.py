from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, Dict

from anthropic.types.beta import BetaToolUnionParam
from google.generativeai.types import FunctionDeclaration


class BaseAnthropicTool(metaclass=ABCMeta):
    """Abstract base class for Anthropic and Gemini-defined tools."""

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        ...

    @abstractmethod
    def to_params(
        self,
    ) -> BetaToolUnionParam:
        raise NotImplementedError

    def to_gemini_tool(self) -> FunctionDeclaration:
        """Convert the tool definition to Gemini's FunctionDeclaration format."""
        anthropic_params = self.to_params()
        
        if not isinstance(anthropic_params, dict):
            raise ValueError("Tool params must be a dictionary")
        
        function_name = anthropic_params.get("function", {}).get("name", "")
        description = anthropic_params.get("function", {}).get("description", "")
        parameters = anthropic_params.get("function", {}).get("parameters", {})
        
        # Convert JSON Schema type parameters to Gemini's format
        def convert_params(params: Dict) -> Dict:
            if "type" not in params:
                return params
            
            # Handle required fields
            required = params.get("required", [])
            properties = params.get("properties", {})
            
            # Mark required fields in properties
            for prop_name, prop_details in properties.items():
                if prop_name in required:
                    prop_details["required"] = True
            
            # Convert type names if needed
            type_mapping = {
                "integer": "number",
                "array": "list",
                # Add more mappings if needed
            }
            
            params_type = params["type"]
            if params_type in type_mapping:
                params["type"] = type_mapping[params_type]
            
            if "properties" in params:
                for prop in params["properties"].values():
                    if isinstance(prop, dict):
                        convert_params(prop)
            
            return params
        
        converted_parameters = convert_params(parameters)
        
        return FunctionDeclaration(
            name=function_name,
            description=description,
            parameters=converted_parameters
        )


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message
