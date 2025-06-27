"""Comprehensive exception hierarchy for LightMCP framework.

This module provides specific exception types for different error scenarios
in MCP operations, with detailed context and recovery information.
"""

from typing import Any, Dict, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for different exception types."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LightMCPError(Exception):
    """Base exception for LightMCP framework with enhanced context.
    
    Provides detailed error information including severity, context,
    and suggested recovery actions.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "original_error": str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        """Enhanced string representation with context."""
        base = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base += f" (Context: {context_str})"
        return base


# =============================================================================
# CONFIGURATION AND SETUP ERRORS
# =============================================================================

class ConfigurationError(LightMCPError):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, **kwargs):
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check configuration parameters",
                "Verify environment variables",
                "Review application setup"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ToolRegistrationError(LightMCPError):
    """Raised when there's an error registering a tool."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if tool_name:
            context['tool_name'] = tool_name
        
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check function signature and annotations",
                "Verify Pydantic model definitions",
                "Ensure function is async if required",
                "Check for naming conflicts"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class ResourceRegistrationError(LightMCPError):
    """Raised when there's an error registering a resource."""
    
    def __init__(self, message: str, resource_uri: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if resource_uri:
            context['resource_uri'] = resource_uri
        
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check URI format and scheme",
                "Verify resource function signature",
                "Ensure URI is unique",
                "Check MIME type specification"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class PromptRegistrationError(LightMCPError):
    """Raised when there's an error registering a prompt."""
    
    def __init__(self, message: str, prompt_name: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if prompt_name:
            context['prompt_name'] = prompt_name
        
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check prompt function signature",
                "Verify argument definitions",
                "Ensure prompt name is unique",
                "Check return type format"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


# =============================================================================
# RUNTIME EXECUTION ERRORS
# =============================================================================

class ToolExecutionError(LightMCPError):
    """Raised when a tool fails to execute properly."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if tool_name:
            context['tool_name'] = tool_name
        if input_data:
            context['input_data'] = input_data
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=[
                "Check input parameters and types",
                "Verify tool function implementation",
                "Check for required dependencies",
                "Review error logs for details"
            ],
            **kwargs
        )


class ResourceAccessError(LightMCPError):
    """Raised when a resource cannot be accessed or read."""
    
    def __init__(
        self, 
        message: str, 
        resource_uri: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if resource_uri:
            context['resource_uri'] = resource_uri
        
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Verify resource URI exists",
                "Check resource function implementation",
                "Ensure resource data is available",
                "Check permissions and access rights"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class PromptExecutionError(LightMCPError):
    """Raised when a prompt fails to execute or generate content."""
    
    def __init__(
        self, 
        message: str, 
        prompt_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if prompt_name:
            context['prompt_name'] = prompt_name
        if arguments:
            context['arguments'] = arguments
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=[
                "Check prompt arguments and types",
                "Verify prompt function implementation",
                "Ensure required parameters are provided",
                "Check prompt template format"
            ],
            **kwargs
        )


# =============================================================================
# VALIDATION AND INPUT ERRORS
# =============================================================================

class ValidationError(LightMCPError):
    """Raised when input validation fails with detailed context."""
    
    def __init__(
        self, 
        message: str, 
        field_errors: Optional[List[Dict[str, Any]]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if field_errors:
            context['field_errors'] = field_errors
        if input_data:
            context['input_data'] = input_data
        
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            context=context,
            recovery_suggestions=[
                "Check input data types and formats",
                "Verify required fields are provided",
                "Review field validation rules",
                "Check for missing or invalid values"
            ],
            **kwargs
        )


class TypeValidationError(LightMCPError):
    """Raised when type validation fails for function parameters."""
    
    def __init__(
        self, 
        message: str, 
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        parameter_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if expected_type:
            context['expected_type'] = expected_type
        if actual_type:
            context['actual_type'] = actual_type
        if parameter_name:
            context['parameter_name'] = parameter_name
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=[
                "Check parameter type annotations",
                "Verify input data matches expected types",
                "Review Pydantic model definitions",
                "Ensure proper type conversion"
            ],
            **kwargs
        )


# =============================================================================
# TRANSPORT AND PROTOCOL ERRORS
# =============================================================================

class TransportError(LightMCPError):
    """Raised when there's a transport-related error."""
    
    def __init__(self, message: str, **kwargs):
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check network connectivity",
                "Verify transport configuration",
                "Review protocol settings",
                "Check for firewall or proxy issues"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ProtocolError(LightMCPError):
    """Raised when there's an MCP protocol-related error."""
    
    def __init__(self, message: str, protocol_version: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if protocol_version:
            context['protocol_version'] = protocol_version
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=[
                "Check MCP protocol version compatibility",
                "Verify message format and structure",
                "Review protocol specifications",
                "Check for implementation updates"
            ],
            **kwargs
        )


class ConnectionError(LightMCPError):
    """Raised when connection to MCP client/server fails."""
    
    def __init__(self, message: str, **kwargs):
        # Don't override recovery_suggestions if provided in kwargs
        if 'recovery_suggestions' not in kwargs:
            kwargs['recovery_suggestions'] = [
                "Check connection parameters",
                "Verify client/server is running",
                "Review network configuration",
                "Check for authentication issues"
            ]
        
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_validation_error_from_pydantic(pydantic_error: Exception, input_data: Optional[Dict[str, Any]] = None) -> ValidationError:
    """Convert Pydantic validation error to LightMCP ValidationError."""
    field_errors = []
    
    if hasattr(pydantic_error, 'errors'):
        for error in pydantic_error.errors():
            field_errors.append({
                'field': '.'.join(str(loc) for loc in error.get('loc', [])),
                'message': error.get('msg', 'Validation failed'),
                'type': error.get('type', 'unknown'),
                'input': error.get('input')
            })
    
    return ValidationError(
        message=f"Input validation failed: {str(pydantic_error)}",
        field_errors=field_errors,
        input_data=input_data,
        original_exception=pydantic_error
    )


def wrap_async_execution_error(func_name: str, original_error: Exception, input_data: Optional[Dict[str, Any]] = None) -> LightMCPError:
    """Wrap async function execution errors with appropriate LightMCP exception."""
    if isinstance(original_error, LightMCPError):
        return original_error
    
    # Determine error type based on function name pattern
    if 'tool' in func_name.lower():
        return ToolExecutionError(
            message=f"Tool '{func_name}' execution failed: {str(original_error)}",
            tool_name=func_name,
            input_data=input_data,
            original_exception=original_error
        )
    elif 'resource' in func_name.lower():
        return ResourceAccessError(
            message=f"Resource '{func_name}' access failed: {str(original_error)}",
            resource_uri=func_name,
            original_exception=original_error
        )
    elif 'prompt' in func_name.lower():
        return PromptExecutionError(
            message=f"Prompt '{func_name}' execution failed: {str(original_error)}",
            prompt_name=func_name,
            arguments=input_data,
            original_exception=original_error
        )
    else:
        return LightMCPError(
            message=f"Function '{func_name}' execution failed: {str(original_error)}",
            context={'function_name': func_name},
            original_exception=original_error
        )