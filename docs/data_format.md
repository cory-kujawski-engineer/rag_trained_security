# Data Format Specification

## Overview
This document specifies the data format requirements for training the security-focused RAG model.

## Training Data Format

### JSON Structure
```json
{
    "version": "1.0",
    "metadata": {
        "description": "Security training data",
        "created": "YYYY-MM-DD",
        "source": "source_name"
    },
    "examples": [
        {
            "id": "unique_identifier",
            "input": "Security query or context",
            "output": "Expected response or analysis",
            "metadata": {
                "type": "vulnerability_analysis|threat_detection|security_review",
                "severity": "low|medium|high|critical",
                "category": "web|network|system|application",
                "tags": ["sql_injection", "xss", "authentication", ...]
            }
        }
    ]
}
```

### Field Descriptions

#### Top-level Fields
- `version`: Data format version
- `metadata`: Global metadata
- `examples`: Array of training examples

#### Example Fields
- `id`: Unique identifier for the example
- `input`: The security-related query or context
- `output`: The expected response or analysis
- `metadata`: Example-specific metadata

#### Metadata Fields
- `type`: Type of security analysis
- `severity`: Severity level
- `category`: Security category
- `tags`: Relevant security tags

## Data Types

### Input Types
1. Vulnerability Analysis
   ```json
   {
       "input": "Analyze the following code for SQL injection vulnerabilities:\n[code snippet]"
   }
   ```

2. Threat Detection
   ```json
   {
       "input": "Review this log file for potential security threats:\n[log content]"
   }
   ```

3. Security Review
   ```json
   {
       "input": "Perform a security review of the following system architecture:\n[architecture description]"
   }
   ```

### Output Types
1. Analysis Response
   ```json
   {
       "output": {
           "findings": ["Finding 1", "Finding 2"],
           "recommendations": ["Rec 1", "Rec 2"],
           "severity": "high"
       }
   }
   ```

2. Threat Report
   ```json
   {
       "output": {
           "threats": ["Threat 1", "Threat 2"],
           "mitigations": ["Mit 1", "Mit 2"],
           "urgency": "critical"
       }
   }
   ```

## Validation Rules

### Required Fields
- All examples must have `id`, `input`, and `output`
- `metadata` must contain `type`, `severity`, and `category`
- `input` and `output` must be non-empty strings

### Value Constraints
- `severity` must be one of: low, medium, high, critical
- `category` must be one of: web, network, system, application
- `type` must be one of: vulnerability_analysis, threat_detection, security_review

## Best Practices

### Data Quality
1. Input Format
   - Clear and concise
   - Proper formatting
   - Relevant context

2. Output Format
   - Structured response
   - Actionable items
   - Clear recommendations

3. Metadata
   - Accurate categorization
   - Relevant tags
   - Proper severity

### Security Considerations
1. Data Sanitization
   - Remove sensitive info
   - Anonymize data
   - Sanitize inputs

2. Content Guidelines
   - No PII
   - No credentials
   - No internal IPs

## Example

```json
{
    "version": "1.0",
    "metadata": {
        "description": "Web security training data",
        "created": "2024-01-01",
        "source": "security_audit_logs"
    },
    "examples": [
        {
            "id": "vuln_001",
            "input": "Review this login form for security vulnerabilities:\n[code snippet]",
            "output": {
                "findings": [
                    "Unsanitized user input",
                    "Weak password requirements"
                ],
                "recommendations": [
                    "Implement input validation",
                    "Enforce strong password policy"
                ],
                "severity": "high"
            },
            "metadata": {
                "type": "vulnerability_analysis",
                "severity": "high",
                "category": "web",
                "tags": ["authentication", "input_validation"]
            }
        }
    ]
}
```

## Contributing
1. Follow format specification
2. Validate data structure
3. Include comprehensive metadata
4. Maintain data quality
