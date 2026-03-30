from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolDisplay


def test_code_tool_call_markdown_uses_code_arg_and_collects_other_args() -> None:
    tool_display = ToolDisplay(ConsoleDisplay())

    markdown, footer_items = tool_display._format_code_tool_call_markdown(
        {
            "code": "def run():\n    return 1\n",
            "limit": 3,
            "raw": True,
        },
        {
            "variant": "code",
            "code_arg": "code",
            "language": "python",
        },
    )

    assert markdown == "```python\ndef run():\n    return 1\n```"
    assert footer_items == ["limit: 3", "raw: true"]
