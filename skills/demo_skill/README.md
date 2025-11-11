# Demo Skill

A demonstration skill showing Hugo's skill system architecture.

## Purpose

This skill serves as a template and example for creating new Hugo skills. It demonstrates:

- **Parameter handling**: How to accept and use skill parameters
- **Execution logging**: Integration with Hugo's logging system
- **Result formatting**: Proper result structure and metadata
- **Error handling**: Graceful error handling and reporting
- **Validation**: Self-validation and testing capabilities

## Usage

### Manual Execution

```bash
hugo skill --validate demo_skill
```

### In Conversation

When chatting with Hugo, you can trigger this skill by:
- Using the command: "demo"
- Using keywords: "demonstration", "example", "show me"

### Parameters

- `message` (string, optional): Message to display (default: "Hello from Hugo!")
- `repeat` (integer, optional): Number of times to repeat (default: 1)

## Creating New Skills

Use this skill as a template for creating your own:

1. Copy the `demo_skill` folder to a new directory
2. Modify `skill.yaml` with your skill's metadata
3. Implement your logic in `main.py`
4. Add tests in `tests/test_main.py`
5. Validate with `hugo skill --validate <skill_name>`

## File Structure

```
demo_skill/
├── skill.yaml          # Skill metadata and configuration
├── main.py             # Skill implementation
├── README.md           # This file
└── tests/
    ├── __init__.py
    └── test_main.py    # Unit tests
```

## Testing

Run tests using pytest:

```bash
cd skills/demo_skill
pytest tests/
```

## Development Notes

- Always handle errors gracefully
- Log execution events for debugging
- Return consistent result structure
- Implement validation for autonomous testing
- Document parameters and capabilities clearly
