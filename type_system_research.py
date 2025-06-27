#!/usr/bin/env python3
"""
Comprehensive Research on Python's Typing System for Enhanced Type Detection

This script explores advanced type introspection capabilities, edge cases, and best practices
for building a robust type detection system for the LightMCP framework.

Focus areas:
1. Advanced type introspection with typing.get_origin(), get_args(), get_type_hints()
2. inspect.signature() and Parameter analysis for complex function signatures
3. Handling Union, Optional, Generic, and nested types
4. Forward references and string annotations
5. Edge cases and challenges
6. Performance considerations and best practices
"""

import asyncio
import inspect
import sys
import typing
from typing import (
    Any, Dict, List, Optional, Union, Generic, TypeVar, ForwardRef,
    get_origin, get_args, get_type_hints
)
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import warnings

# ============================================================================
# SETUP: Example Models and Types for Testing
# ============================================================================

class UserModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

class ProductModel(BaseModel):
    id: int
    name: str
    price: float

class OrderModel(BaseModel):
    user: UserModel
    products: List[ProductModel]
    total: float

# Generic type variable
T = TypeVar('T', bound=BaseModel)

class GenericContainer(BaseModel, Generic[T]):
    data: T
    metadata: Dict[str, Any]

# Forward reference scenario
class NodeModel(BaseModel):
    value: int
    children: Optional[List['NodeModel']] = None  # Forward reference

# Complex nested types
ComplexType = Dict[str, List[Union[UserModel, ProductModel]]]
DeepNested = List[Dict[str, Optional[Union[UserModel, List[ProductModel]]]]]

# ============================================================================
# 1. ADVANCED TYPE INTROSPECTION CAPABILITIES
# ============================================================================

def demonstrate_type_introspection():
    """Demonstrate typing.get_origin(), get_args(), get_type_hints() capabilities."""
    print("=" * 80)
    print("1. ADVANCED TYPE INTROSPECTION CAPABILITIES")
    print("=" * 80)
    
    # Basic type introspection
    test_types = [
        Optional[str],
        Union[str, int, None],
        List[UserModel],
        Dict[str, UserModel],
        Union[UserModel, ProductModel],
        List[Dict[str, Optional[UserModel]]],
        GenericContainer[UserModel],
    ]
    
    for test_type in test_types:
        print(f"\nType: {test_type}")
        print(f"  Origin: {get_origin(test_type)}")
        print(f"  Args: {get_args(test_type)}")
        print(f"  Is Union: {get_origin(test_type) is Union}")
        print(f"  Is Generic: {hasattr(test_type, '__origin__')}")
        
        # Deep introspection for nested types
        if get_args(test_type):
            for i, arg in enumerate(get_args(test_type)):
                print(f"    Arg[{i}]: {arg}")
                print(f"      Origin: {get_origin(arg)}")
                print(f"      Args: {get_args(arg)}")
                
                # Check if arg is Pydantic model
                if hasattr(arg, 'model_fields') or hasattr(arg, '__fields__'):
                    print(f"      Is Pydantic Model: True")
                elif get_origin(arg) in (list, List):
                    nested_args = get_args(arg)
                    if nested_args:
                        for nested_arg in nested_args:
                            if hasattr(nested_arg, 'model_fields'):
                                print(f"      Contains Pydantic Model: {nested_arg}")


def demonstrate_function_signature_analysis():
    """Demonstrate inspect.signature() and Parameter analysis."""
    print("\n" + "=" * 80)
    print("2. FUNCTION SIGNATURE ANALYSIS")
    print("=" * 80)
    
    # Example functions with different signature patterns
    
    async def simple_func(user: UserModel) -> dict:
        """Simple function with Pydantic model parameter."""
        return {"user": user.name}
    
    async def optional_func(user: Optional[UserModel] = None) -> dict:
        """Function with optional Pydantic model."""
        return {"user": user.name if user else "anonymous"}
    
    async def union_func(data: Union[UserModel, ProductModel, str]) -> dict:
        """Function with Union type including Pydantic models."""
        if isinstance(data, str):
            return {"message": data}
        return {"model_data": True}
    
    async def complex_func(
        users: List[UserModel],
        metadata: Dict[str, Any],
        config: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[dict]:
        """Function with complex nested types."""
        return [{"user": u.name} for u in users]
    
    async def generic_func(container: GenericContainer[UserModel]) -> dict:
        """Function with generic type."""
        return {"data": container.data.name}
    
    # Analyze each function
    test_functions = [
        simple_func, optional_func, union_func, complex_func, generic_func
    ]
    
    for func in test_functions:
        print(f"\nFunction: {func.__name__}")
        sig = inspect.signature(func)
        print(f"  Signature: {sig}")
        
        # Get type hints (handles forward references)
        try:
            type_hints = get_type_hints(func)
            print(f"  Type Hints: {type_hints}")
        except NameError as e:
            print(f"  Type Hints Error: {e}")
            # Fallback to raw annotations
            type_hints = getattr(func, '__annotations__', {})
            print(f"  Raw Annotations: {type_hints}")
        
        # Analyze each parameter
        for param_name, param in sig.parameters.items():
            print(f"    Parameter: {param_name}")
            print(f"      Annotation: {param.annotation}")
            print(f"      Default: {param.default}")
            print(f"      Kind: {param.kind}")
            
            # Extract Pydantic models from parameter type
            models = extract_pydantic_models_from_type(param.annotation)
            if models:
                print(f"      Pydantic Models: {models}")


def extract_pydantic_models_from_type(type_annotation) -> List[type]:
    """Extract all Pydantic models from a type annotation."""
    models = []
    
    def _extract_recursive(t):
        # Direct Pydantic model check
        if hasattr(t, 'model_fields') or hasattr(t, '__fields__'):
            models.append(t)
            return
        
        # Handle Union types
        if get_origin(t) is Union:
            for arg in get_args(t):
                _extract_recursive(arg)
            return
        
        # Handle generic types (List, Dict, etc.)
        origin = get_origin(t)
        if origin in (list, List, dict, Dict, tuple, set):
            for arg in get_args(t):
                _extract_recursive(arg)
            return
        
        # Handle other generic types
        if hasattr(t, '__origin__') and hasattr(t, '__args__'):
            for arg in get_args(t):
                _extract_recursive(arg)
    
    _extract_recursive(type_annotation)
    return models


# ============================================================================
# 3. COMPLEX TYPE PATTERNS HANDLING
# ============================================================================

def demonstrate_complex_type_patterns():
    """Demonstrate handling of complex type patterns."""
    print("\n" + "=" * 80)
    print("3. COMPLEX TYPE PATTERNS HANDLING")
    print("=" * 80)
    
    # Test patterns that the framework needs to support
    complex_patterns = [
        ("Optional[BaseModel]", Optional[UserModel]),
        ("List[BaseModel]", List[UserModel]),
        ("Dict[str, BaseModel]", Dict[str, UserModel]),
        ("Union[BaseModel1, BaseModel2, str]", Union[UserModel, ProductModel, str]),
        ("Generic[T] where T: BaseModel", GenericContainer[UserModel]),
        ("Nested List[Dict[str, BaseModel]]", List[Dict[str, UserModel]]),
        ("Complex Union", Union[List[UserModel], Dict[str, ProductModel], str]),
        ("Deep Nesting", Dict[str, List[Optional[Union[UserModel, ProductModel]]]]),
    ]
    
    for pattern_name, pattern_type in complex_patterns:
        print(f"\nPattern: {pattern_name}")
        print(f"  Type: {pattern_type}")
        
        # Analyze the pattern
        analysis = analyze_type_pattern(pattern_type)
        print(f"  Analysis: {analysis}")
        
        # Test model extraction
        models = extract_pydantic_models_from_type(pattern_type)
        print(f"  Extracted Models: {[m.__name__ for m in models]}")


def analyze_type_pattern(type_annotation) -> dict:
    """Analyze a type pattern and return structured information."""
    analysis = {
        'is_optional': False,
        'is_union': False,
        'is_generic': False,
        'is_collection': False,
        'base_types': [],
        'pydantic_models': [],
        'complexity_score': 0
    }
    
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    
    # Check if Optional (Union with None)
    if origin is Union and type(None) in args:
        analysis['is_optional'] = True
        analysis['complexity_score'] += 1
    
    # Check if Union
    if origin is Union:
        analysis['is_union'] = True
        analysis['complexity_score'] += len(args)
    
    # Check if Generic/Collection
    if origin in (list, List, dict, Dict, tuple, set):
        analysis['is_collection'] = True
        analysis['complexity_score'] += 2
    
    # Check if custom Generic
    if hasattr(type_annotation, '__origin__') and hasattr(type_annotation, '__args__'):
        analysis['is_generic'] = True
        analysis['complexity_score'] += 1
    
    # Extract base types and Pydantic models recursively
    def _extract_info(t):
        if hasattr(t, 'model_fields') or hasattr(t, '__fields__'):
            analysis['pydantic_models'].append(t)
        elif hasattr(t, '__name__'):
            analysis['base_types'].append(t)
        
        # Recurse into args
        for arg in get_args(t):
            _extract_info(arg)
    
    _extract_info(type_annotation)
    
    return analysis


# ============================================================================
# 4. EDGE CASES AND CHALLENGES
# ============================================================================

def demonstrate_edge_cases():
    """Demonstrate edge cases and challenges in type introspection."""
    print("\n" + "=" * 80)
    print("4. EDGE CASES AND CHALLENGES")
    print("=" * 80)
    
    # Forward references
    print("\n4.1 Forward References:")
    try:
        # This will work in Python 3.11+ with proper handling
        node_hints = get_type_hints(NodeModel)
        print(f"  NodeModel type hints: {node_hints}")
    except NameError as e:
        print(f"  Forward reference error: {e}")
        print("  Raw annotations:", NodeModel.__annotations__)
    
    # String annotations (from __future__ import annotations)
    print("\n4.2 String Annotations:")
    def string_annotated_func(user: "UserModel") -> "Dict[str, Any]":
        return {}
    
    try:
        string_hints = get_type_hints(string_annotated_func)
        print(f"  String annotation resolved: {string_hints}")
    except NameError as e:
        print(f"  String annotation error: {e}")
        print(f"  Raw annotations: {string_annotated_func.__annotations__}")
    
    # Circular dependencies simulation
    print("\n4.3 Circular Dependencies:")
    # This would typically cause issues in real scenarios
    try:
        # Simulate resolving a type that might have circular refs
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            print("  Type checking context available")
        else:
            print("  Runtime context - circular refs may cause issues")
    except Exception as e:
        print(f"  Circular dependency error: {e}")
    
    # Runtime vs compile-time type checking
    print("\n4.4 Runtime vs Compile-time:")
    def runtime_check_example(data: Any) -> bool:
        """Example of runtime type checking."""
        # Check if data matches UserModel structure
        if hasattr(data, 'model_fields'):
            print(f"  Runtime: Detected Pydantic model: {type(data)}")
            return True
        
        # Check if data is dict that could be validated as UserModel
        if isinstance(data, dict):
            try:
                UserModel(**data)
                print(f"  Runtime: Dict can be validated as UserModel")
                return True
            except ValidationError as e:
                print(f"  Runtime: Dict validation failed: {e}")
                return False
        
        return False
    
    # Test runtime checking
    test_data = [
        UserModel(name="John", age=30),
        {"name": "Jane", "age": 25},
        {"invalid": "data"},
        "not a model"
    ]
    
    for data in test_data:
        print(f"  Testing: {type(data)} - {runtime_check_example(data)}")


# ============================================================================
# 5. PERFORMANCE CONSIDERATIONS AND BEST PRACTICES
# ============================================================================

def demonstrate_performance_considerations():
    """Demonstrate performance considerations for type introspection."""
    print("\n" + "=" * 80)
    print("5. PERFORMANCE CONSIDERATIONS AND BEST PRACTICES")
    print("=" * 80)
    
    import time
    from functools import lru_cache
    
    # Performance test: Type introspection with and without caching
    complex_type = Dict[str, List[Union[UserModel, ProductModel, Optional[OrderModel]]]]
    
    def slow_type_analysis(type_annotation):
        """Non-cached type analysis."""
        return {
            'origin': get_origin(type_annotation),
            'args': get_args(type_annotation),
            'models': extract_pydantic_models_from_type(type_annotation),
            'analysis': analyze_type_pattern(type_annotation)
        }
    
    @lru_cache(maxsize=128)
    def cached_type_analysis(type_annotation):
        """Cached type analysis."""
        return slow_type_analysis(type_annotation)
    
    # Time the operations
    iterations = 1000
    
    # Non-cached version
    start_time = time.time()
    for _ in range(iterations):
        slow_type_analysis(complex_type)
    slow_time = time.time() - start_time
    
    # Cached version (first call to populate cache)
    cached_type_analysis(complex_type)
    start_time = time.time()
    for _ in range(iterations):
        cached_type_analysis(complex_type)
    cached_time = time.time() - start_time
    
    print(f"\nPerformance Test Results ({iterations} iterations):")
    print(f"  Non-cached: {slow_time:.4f}s ({slow_time/iterations*1000:.2f}ms per call)")
    print(f"  Cached: {cached_time:.4f}s ({cached_time/iterations*1000:.2f}ms per call)")
    print(f"  Speedup: {slow_time/cached_time:.1f}x")
    
    # Memory usage considerations
    print(f"\nMemory Considerations:")
    print(f"  Cache info: {cached_type_analysis.cache_info()}")
    
    # Best practices recommendations
    print(f"\nBest Practices:")
    print(f"  1. Cache type analysis results using @lru_cache")
    print(f"  2. Limit cache size to prevent memory leaks")
    print(f"  3. Handle forward references with try/except")
    print(f"  4. Use get_type_hints() for proper resolution")
    print(f"  5. Validate at registration time, not runtime")


# ============================================================================
# 6. ENHANCED TYPE DETECTION SYSTEM DESIGN
# ============================================================================

class EnhancedTypeDetector:
    """
    Enhanced type detection system with comprehensive type introspection.
    
    Features:
    - Advanced type pattern matching
    - Pydantic model extraction
    - Forward reference resolution
    - Performance optimization with caching
    - Comprehensive error handling
    """
    
    def __init__(self):
        self._cache = {}
        self._resolution_cache = {}
    
    @lru_cache(maxsize=256)
    def analyze_function_signature(self, func: callable) -> dict:
        """
        Analyze function signature and extract type information.
        
        Returns comprehensive analysis including:
        - Parameter types and models
        - Return type information
        - Validation requirements
        - Error handling suggestions
        """
        try:
            sig = inspect.signature(func)
            
            # Try to get resolved type hints
            try:
                type_hints = get_type_hints(func)
            except (NameError, AttributeError) as e:
                # Fallback to raw annotations
                type_hints = getattr(func, '__annotations__', {})
                warnings.warn(f"Could not resolve type hints for {func.__name__}: {e}")
            
            analysis = {
                'function_name': func.__name__,
                'parameters': {},
                'return_type': None,
                'primary_input_model': None,
                'all_models': [],
                'complexity_score': 0,
                'validation_strategy': 'none',
                'errors': []
            }
            
            # Analyze each parameter
            for param_name, param in sig.parameters.items():
                if param_name == 'return':
                    continue
                
                param_info = self._analyze_parameter(param, type_hints.get(param_name))
                analysis['parameters'][param_name] = param_info
                analysis['all_models'].extend(param_info['pydantic_models'])
                analysis['complexity_score'] += param_info['complexity_score']
            
            # Determine primary input model (first Pydantic model parameter)
            for param_info in analysis['parameters'].values():
                if param_info['pydantic_models']:
                    analysis['primary_input_model'] = param_info['pydantic_models'][0]
                    analysis['validation_strategy'] = 'pydantic_model'
                    break
            
            # If no primary model, check for dict-like parameters
            if not analysis['primary_input_model']:
                for param_info in analysis['parameters'].values():
                    if param_info['is_dict_like']:
                        analysis['validation_strategy'] = 'dict_validation'
                        break
            
            # Analyze return type
            return_type = type_hints.get('return', sig.return_annotation)
            if return_type != inspect.Parameter.empty:
                analysis['return_type'] = self._analyze_type(return_type)
            
            return analysis
            
        except Exception as e:
            return {
                'function_name': getattr(func, '__name__', 'unknown'),
                'error': str(e),
                'parameters': {},
                'validation_strategy': 'error'
            }
    
    def _analyze_parameter(self, param: inspect.Parameter, type_hint: Any) -> dict:
        """Analyze a single function parameter."""
        param_info = {
            'name': param.name,
            'annotation': param.annotation,
            'type_hint': type_hint,
            'default': param.default,
            'required': param.default == inspect.Parameter.empty,
            'pydantic_models': [],
            'base_types': [],
            'is_optional': False,
            'is_union': False,
            'is_collection': False,
            'is_dict_like': False,
            'complexity_score': 0
        }
        
        # Use type hint if available, otherwise use annotation
        analysis_type = type_hint if type_hint is not None else param.annotation
        
        if analysis_type != inspect.Parameter.empty:
            type_analysis = self._analyze_type(analysis_type)
            param_info.update(type_analysis)
        
        return param_info
    
    @lru_cache(maxsize=512)
    def _analyze_type(self, type_annotation: Any) -> dict:
        """Analyze a type annotation and return structured information."""
        analysis = {
            'pydantic_models': [],
            'base_types': [],
            'is_optional': False,
            'is_union': False,
            'is_collection': False,
            'is_dict_like': False,
            'complexity_score': 0
        }
        
        if type_annotation is None or type_annotation == inspect.Parameter.empty:
            return analysis
        
        # Handle string annotations
        if isinstance(type_annotation, str):
            try:
                # Attempt to resolve string annotation
                # This is simplified - real implementation would need proper context
                analysis['complexity_score'] += 1
                return analysis
            except:
                analysis['complexity_score'] += 2
                return analysis
        
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)
        
        # Check for Union types (including Optional)
        if origin is Union:
            analysis['is_union'] = True
            analysis['complexity_score'] += len(args)
            
            # Check if it's Optional (Union with None)
            if type(None) in args:
                analysis['is_optional'] = True
            
            # Analyze each union member
            for arg in args:
                if arg != type(None):
                    sub_analysis = self._analyze_type(arg)
                    analysis['pydantic_models'].extend(sub_analysis['pydantic_models'])
                    analysis['base_types'].extend(sub_analysis['base_types'])
        
        # Check for collection types
        elif origin in (list, List, dict, Dict, tuple, set):
            analysis['is_collection'] = True
            analysis['complexity_score'] += 2
            
            if origin in (dict, Dict):
                analysis['is_dict_like'] = True
            
            # Analyze collection element types
            for arg in args:
                sub_analysis = self._analyze_type(arg)
                analysis['pydantic_models'].extend(sub_analysis['pydantic_models'])
                analysis['base_types'].extend(sub_analysis['base_types'])
                analysis['complexity_score'] += sub_analysis['complexity_score']
        
        # Check for Pydantic models
        elif hasattr(type_annotation, 'model_fields') or hasattr(type_annotation, '__fields__'):
            analysis['pydantic_models'].append(type_annotation)
            analysis['complexity_score'] += 1
        
        # Check for other types
        else:
            analysis['base_types'].append(type_annotation)
        
        return analysis
    
    def generate_validation_strategy(self, func: callable) -> dict:
        """Generate a validation strategy for a function."""
        analysis = self.analyze_function_signature(func)
        
        strategy = {
            'type': analysis['validation_strategy'],
            'input_model': analysis['primary_input_model'],
            'parameter_validation': {},
            'error_handling': [],
            'performance_notes': []
        }
        
        # Generate parameter-specific validation
        for param_name, param_info in analysis['parameters'].items():
            if param_info['pydantic_models']:
                strategy['parameter_validation'][param_name] = {
                    'method': 'pydantic_validation',
                    'model': param_info['pydantic_models'][0],
                    'required': param_info['required']
                }
            elif param_info['is_dict_like']:
                strategy['parameter_validation'][param_name] = {
                    'method': 'dict_validation',
                    'required': param_info['required']
                }
        
        # Generate error handling recommendations
        if analysis['complexity_score'] > 5:
            strategy['error_handling'].append('Use comprehensive try/catch blocks')
            strategy['performance_notes'].append('Consider caching validation results')
        
        if any(p['is_union'] for p in analysis['parameters'].values()):
            strategy['error_handling'].append('Handle Union type validation carefully')
        
        return strategy


# ============================================================================
# 7. DEMONSTRATION AND TESTING
# ============================================================================

def run_enhanced_type_detection_demo():
    """Demonstrate the enhanced type detection system."""
    print("\n" + "=" * 80)
    print("6. ENHANCED TYPE DETECTION SYSTEM DEMO")
    print("=" * 80)
    
    detector = EnhancedTypeDetector()
    
    # Test functions with various complexity levels
    async def simple_func(user: UserModel) -> dict:
        return {"name": user.name}
    
    async def complex_func(
        users: List[UserModel],
        products: Optional[Dict[str, ProductModel]] = None,
        metadata: Union[str, Dict[str, Any]] = "default"
    ) -> List[Dict[str, Any]]:
        return []
    
    async def generic_func(container: GenericContainer[UserModel]) -> dict:
        return {"data": container.data}
    
    test_functions = [simple_func, complex_func, generic_func]
    
    for func in test_functions:
        print(f"\n--- Analysis for {func.__name__} ---")
        
        # Analyze function signature
        analysis = detector.analyze_function_signature(func)
        print(f"Function: {analysis['function_name']}")
        print(f"Complexity Score: {analysis['complexity_score']}")
        print(f"Primary Input Model: {analysis['primary_input_model']}")
        print(f"Validation Strategy: {analysis['validation_strategy']}")
        
        # Show parameter analysis
        for param_name, param_info in analysis['parameters'].items():
            print(f"  Parameter '{param_name}':")
            print(f"    Required: {param_info['required']}")
            print(f"    Pydantic Models: {[m.__name__ for m in param_info['pydantic_models']]}")
            print(f"    Is Optional: {param_info['is_optional']}")
            print(f"    Is Union: {param_info['is_union']}")
            print(f"    Is Collection: {param_info['is_collection']}")
        
        # Generate validation strategy
        strategy = detector.generate_validation_strategy(func)
        print(f"  Validation Strategy: {strategy}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations and research."""
    print("COMPREHENSIVE PYTHON TYPING SYSTEM RESEARCH")
    print("For Enhanced Type Detection in FastAPI-inspired MCP Framework")
    print("=" * 80)
    
    try:
        demonstrate_type_introspection()
        demonstrate_function_signature_analysis()
        demonstrate_complex_type_patterns()
        demonstrate_edge_cases()
        demonstrate_performance_considerations()
        run_enhanced_type_detection_demo()
        
        print("\n" + "=" * 80)
        print("RESEARCH COMPLETE")
        print("=" * 80)
        
        print("\nKEY FINDINGS AND RECOMMENDATIONS:")
        print("1. Use get_type_hints() for proper forward reference resolution")
        print("2. Implement caching with @lru_cache for performance")
        print("3. Handle Union types and Optional carefully")
        print("4. Extract Pydantic models recursively from complex types")
        print("5. Provide fallbacks for unresolvable type annotations")
        print("6. Validate at registration time, not runtime when possible")
        print("7. Use comprehensive error handling for edge cases")
        print("\nImplementation ready for integration into LightMCP framework!")
        
    except Exception as e:
        print(f"Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()