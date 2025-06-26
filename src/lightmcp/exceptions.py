"""Exception classes for LightMCP."""


class LightMCPError(Exception):
    """Base exception for LightMCP framework."""

    pass


class ConfigurationError(LightMCPError):
    """Raised when there's a configuration error."""

    pass


class ToolRegistrationError(LightMCPError):
    """Raised when there's an error registering a tool."""

    pass


class TransportError(LightMCPError):
    """Raised when there's a transport-related error."""

    pass