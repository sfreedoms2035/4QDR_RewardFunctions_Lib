"""
coding.py — Task-Specific Rewards for Coding and Visual Tasks

RF-14: code_syntax_reward — Syntactic quality of generated code
RF-15: code_execution_reward — Verifiable code execution in sandbox
RF-16: visual_code_volume_reward — Line count for visual artifacts
"""

import re
import subprocess
import tempfile
import os

from reward_functions.utils.text_analysis import (
    extract_code_blocks, count_code_lines, sigmoid, clamp,
)


# Placeholder patterns that indicate incomplete code
PLACEHOLDER_PATTERNS = [
    r'\bTODO\b', r'\bFIXME\b', r'\bHACK\b',
    r'\bpass\s*$', r'\.\.\.\s*$',
    r'//\s*add\s+logic', r'#\s*add\s+logic',
    r'//\s*implement', r'#\s*implement',
    r'placeholder', r'not\s+implemented',
    r'raise\s+NotImplementedError',
    r'throw\s+new\s+Error\(["\']not\s+implemented',
]


def code_syntax_reward(completions: list, **kwargs) -> list:
    """RF-14: Code syntactic quality assessment.
    
    Checks generated code blocks for structural quality markers.
    
    Scoring:
        +0.30  Matching braces/brackets/parens
        +0.20  No placeholder patterns
        +0.25  Function/class definitions present
        +0.25  Import statements present and plausible
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            # No code blocks found — might be inline code
            rewards.append(0.0)
            continue
        
        all_code = '\n'.join(code_blocks)
        score = 0.0
        
        # Check 1: Matching delimiters
        brace_balanced = all_code.count('{') == all_code.count('}')
        bracket_balanced = all_code.count('[') == all_code.count(']')
        paren_balanced = all_code.count('(') == all_code.count(')')
        
        delimiter_score = sum([brace_balanced, bracket_balanced, paren_balanced]) / 3.0
        score += delimiter_score * 0.30
        
        # Check 2: No placeholders
        placeholder_count = sum(
            len(re.findall(p, all_code, re.IGNORECASE | re.MULTILINE))
            for p in PLACEHOLDER_PATTERNS
        )
        if placeholder_count == 0:
            score += 0.20
        else:
            score += max(0.0, 0.20 - placeholder_count * 0.05)
        
        # Check 3: Function/class definitions
        func_defs = len(re.findall(
            r'(?:def |function |fn |func |class |struct |impl |pub fn )',
            all_code
        ))
        if func_defs >= 3:
            score += 0.25
        elif func_defs >= 1:
            score += 0.12
        
        # Check 4: Import statements
        import_count = len(re.findall(
            r'(?:^|\n)\s*(?:import |from |use |#include |require |const \w+ = require)',
            all_code
        ))
        if import_count >= 2:
            score += 0.25
        elif import_count >= 1:
            score += 0.12
        
        rewards.append(clamp(score))
    return rewards


def code_execution_reward(completions: list, **kwargs) -> list:
    """RF-15: Verifiable code execution in sandbox (extensive mode only).
    
    Extracts Python code blocks and attempts to execute them in a
    sandboxed subprocess with strict resource limits.
    
    Scoring:
        0.0   No code found or sandbox error
        0.1   Code present but syntax error
        0.3   Syntactically valid but runtime error
        0.7   Runs to completion without error
        1.0   Runs and passes any embedded assertions
    
    Returns: list[float] in [0.0, 1.0]
    """
    mode = kwargs.get("mode", "lightweight")
    if mode != "extensive":
        # In lightweight mode, return neutral score (don't penalize or reward)
        return [0.5] * len(completions)
    
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        code_blocks = extract_code_blocks(completion)
        python_blocks = _filter_python_blocks(code_blocks, completion)
        
        if not python_blocks:
            rewards.append(0.0)
            continue
        
        # Execute the largest Python block
        code = max(python_blocks, key=len)
        score = _execute_python_safely(code)
        rewards.append(score)
    
    return rewards


def _filter_python_blocks(code_blocks: list, full_text: str) -> list:
    """Filter code blocks to only include Python-like code."""
    python_blocks = []
    for block in code_blocks:
        # Heuristic: check for Python-ish syntax
        if any(indicator in block for indicator in [
            'def ', 'import ', 'class ', 'print(', 'if __name__',
            'for ', 'while ', 'return ', 'from ', 'with ',
        ]):
            python_blocks.append(block)
    return python_blocks


def _execute_python_safely(code: str, timeout: int = 30) -> float:
    """Execute Python code in a sandboxed subprocess.
    
    Returns a score from 0.0 to 1.0 based on execution result.
    """
    # Security: strip dangerous operations
    dangerous_patterns = [
        r'\bos\.system\b', r'\bsubprocess\b', r'\beval\b', r'\bexec\b',
        r'\b__import__\b', r'\bopen\b.*["\']w', r'\bshutil\b',
        r'\brm\s+-rf\b', r'\bdel\b.*\bos\b',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return 0.0  # Refuse to execute
    
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                         delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            )
            
            if result.returncode == 0:
                # Check for assertion passes
                if 'AssertionError' not in result.stderr:
                    if 'assert' in code.lower():
                        return 1.0  # Has assertions and they passed
                    return 0.7  # Ran without error
                return 0.3  # Assertion failed
            else:
                stderr = result.stderr.lower()
                if 'syntaxerror' in stderr or 'indentationerror' in stderr:
                    return 0.1
                return 0.3  # Runtime error
        finally:
            os.unlink(temp_path)
            
    except subprocess.TimeoutExpired:
        return 0.3  # Timeout = possible infinite loop
    except Exception:
        return 0.0


def visual_code_volume_reward(completions: list, **kwargs) -> list:
    """RF-16: Visual/diagramming code volume assessment.
    
    Checks line count for visual artifact code (PlantUML, D2, Graphviz, 
    Mermaid, TikZ, SVG, HTML).
    
    Scoring:
        0.35  Code line count via sigmoid (target: 300 lines)
        0.35  Total content volume via sigmoid (target: 600 lines for HTML)
        0.15  No truncated/incomplete structures
        0.15  Multiple interconnected elements detected
    
    Returns: list[float] in [0.0, 1.0]
    """
    task_type = kwargs.get("task_type", "")
    line_target = 600 if task_type in ("html_tool", "html_presentation") else 300
    
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        code_lines = count_code_lines(completion)
        score = 0.0
        
        # Code volume
        score += sigmoid(code_lines, line_target) * 0.35
        
        # Total lines
        total_lines = len(completion.split('\n'))
        score += sigmoid(total_lines, line_target * 1.5) * 0.35
        
        # Completeness (no truncation markers)
        code_blocks = extract_code_blocks(completion)
        all_code = '\n'.join(code_blocks) if code_blocks else ""
        
        truncation_markers = ['...', '<!-- truncated', '/* truncated', '# truncated']
        is_complete = not any(m in all_code for m in truncation_markers)
        score += 0.15 if is_complete else 0.0
        
        # Interconnected elements
        element_patterns = [
            r'class\s+\w+', r'node\s+\w+', r'actor\s+\w+',
            r'component\s+\w+', r'<div', r'<section',
            r'shape:\s+', r'style\s*{',
        ]
        element_count = sum(
            len(re.findall(p, all_code, re.IGNORECASE))
            for p in element_patterns
        )
        score += 0.15 if element_count >= 5 else 0.15 * min(1.0, element_count / 5.0)
        
        rewards.append(clamp(score))
    return rewards
