"""
Tests for Hugo Core-2 Skill Router
-----------------------------------
Tests the minimal skill routing system for CORE mode.

Requirements:
- Detect skill triggers at start of text only
- Route to appropriate handlers
- Format results in machine-readable blocks
- No Ollama calls, no I/O, deterministic
- Inject skill blocks into prompts
"""

import pytest
from core.skills.trigger_detector import detect_skill, SkillTrigger
from core.skills.router import (
    route_skill,
    validate_skill_name,
    is_allowed_skill,
    SkillBlock
)
from core.skills.handlers import (
    handle_search,
    handle_note,
    handle_fetch,
    handle_scrape,
    SkillResult
)
from core.skills.schemas import format_skill_result, format_skill_error
from core.skills.prompt_injection import inject_skill_block


# ============================================================================
# TRIGGER DETECTION TESTS
# ============================================================================

class TestTriggerDetection:
    """Test skill trigger detection"""

    def test_detect_simple_command(self):
        """Test detection of simple skill command"""
        text = "/search python asyncio"
        trigger = detect_skill(text)

        assert trigger is not None
        assert trigger.skill_name == "search"
        assert trigger.args == "python asyncio"
        assert trigger.raw_input == text

    def test_detect_command_no_args(self):
        """Test detection of command without arguments"""
        text = "/note"
        trigger = detect_skill(text)

        assert trigger is not None
        assert trigger.skill_name == "note"
        assert trigger.args == ""
        assert trigger.raw_input == text

    def test_detect_command_with_whitespace(self):
        """Test detection with extra whitespace"""
        text = "/fetch   https://example.com"
        trigger = detect_skill(text)

        assert trigger is not None
        assert trigger.skill_name == "fetch"
        assert trigger.args == "https://example.com"

    def test_no_detection_mid_string(self):
        """Test that commands mid-string are not detected"""
        text = "Can you /search for something?"
        trigger = detect_skill(text)

        assert trigger is None

    def test_no_detection_without_slash(self):
        """Test that text without slash is not detected"""
        text = "search python asyncio"
        trigger = detect_skill(text)

        assert trigger is None

    def test_no_detection_empty_string(self):
        """Test empty string returns None"""
        text = ""
        trigger = detect_skill(text)

        assert trigger is None

    def test_no_detection_whitespace_only(self):
        """Test whitespace-only string returns None"""
        text = "   "
        trigger = detect_skill(text)

        assert trigger is None

    def test_detection_preserves_exact_input(self):
        """Test that raw_input is preserved exactly"""
        text = "/search  python   asyncio  "
        trigger = detect_skill(text)

        assert trigger is not None
        assert trigger.raw_input == text


# ============================================================================
# SKILL NAME VALIDATION TESTS
# ============================================================================

class TestSkillNameValidation:
    """Test skill name validation rules"""

    def test_validate_lowercase_alphabetic(self):
        """Test that lowercase alphabetic names are valid"""
        assert validate_skill_name("search") is True
        assert validate_skill_name("note") is True
        assert validate_skill_name("fetch") is True
        assert validate_skill_name("scrape") is True

    def test_reject_uppercase(self):
        """Test that uppercase names are rejected"""
        assert validate_skill_name("SEARCH") is False
        assert validate_skill_name("Search") is False
        assert validate_skill_name("sEarch") is False

    def test_reject_numbers(self):
        """Test that names with numbers are rejected"""
        assert validate_skill_name("search1") is False
        assert validate_skill_name("1search") is False
        assert validate_skill_name("sea1rch") is False

    def test_reject_underscore(self):
        """Test that names with underscores are rejected"""
        assert validate_skill_name("web_search") is False
        assert validate_skill_name("_search") is False
        assert validate_skill_name("search_") is False

    def test_reject_hyphen(self):
        """Test that names with hyphens are rejected"""
        assert validate_skill_name("web-search") is False
        assert validate_skill_name("-search") is False
        assert validate_skill_name("search-") is False

    def test_reject_special_chars(self):
        """Test that names with special characters are rejected"""
        assert validate_skill_name("search!") is False
        assert validate_skill_name("search@home") is False
        assert validate_skill_name("search$") is False

    def test_reject_unicode(self):
        """Test that unicode characters are rejected"""
        assert validate_skill_name("búsqueda") is False
        assert validate_skill_name("検索") is False

    def test_reject_empty_string(self):
        """Test that empty string is rejected"""
        assert validate_skill_name("") is False

    def test_is_allowed_skill_whitelist(self):
        """Test that only whitelisted skills are allowed"""
        assert is_allowed_skill("search") is True
        assert is_allowed_skill("note") is True
        assert is_allowed_skill("fetch") is True
        assert is_allowed_skill("scrape") is True

    def test_is_allowed_skill_rejects_unknown(self):
        """Test that unknown skills are rejected"""
        assert is_allowed_skill("unknown") is False
        assert is_allowed_skill("delete") is False
        assert is_allowed_skill("execute") is False


# ============================================================================
# ROUTING TESTS
# ============================================================================

class TestSkillRouting:
    """Test skill routing logic"""

    @pytest.mark.asyncio
    async def test_route_search_skill(self):
        """Test routing to search handler"""
        trigger = SkillTrigger(
            skill_name="search",
            args="python asyncio",
            raw_input="/search python asyncio"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillResult]" in block.block
        assert "skill: search" in block.block
        assert "args: \"python asyncio\"" in block.block
        assert "[Placeholder: Search results]" in block.block

    @pytest.mark.asyncio
    async def test_route_note_skill(self):
        """Test routing to note handler"""
        trigger = SkillTrigger(
            skill_name="note",
            args="Remember to buy milk",
            raw_input="/note Remember to buy milk"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillResult]" in block.block
        assert "skill: note" in block.block
        assert "[Placeholder: Note saved]" in block.block

    @pytest.mark.asyncio
    async def test_route_fetch_skill(self):
        """Test routing to fetch handler"""
        trigger = SkillTrigger(
            skill_name="fetch",
            args="https://example.com",
            raw_input="/fetch https://example.com"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillResult]" in block.block
        assert "skill: fetch" in block.block
        assert "[Placeholder: URL content fetched]" in block.block

    @pytest.mark.asyncio
    async def test_route_scrape_skill(self):
        """Test routing to scrape handler"""
        trigger = SkillTrigger(
            skill_name="scrape",
            args="https://example.com",
            raw_input="/scrape https://example.com"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillResult]" in block.block
        assert "skill: scrape" in block.block
        assert "[Placeholder: Page scraped]" in block.block

    @pytest.mark.asyncio
    async def test_route_unknown_skill(self):
        """Test routing unknown skill returns error"""
        trigger = SkillTrigger(
            skill_name="unknown",
            args="test",
            raw_input="/unknown test"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillError]" in block.block
        assert "skill: unknown" in block.block
        assert "error:" in block.block

    @pytest.mark.asyncio
    async def test_route_invalid_skill_name(self):
        """Test routing invalid skill name returns error"""
        trigger = SkillTrigger(
            skill_name="SEARCH",
            args="test",
            raw_input="/SEARCH test"
        )

        block = await route_skill(trigger)

        assert isinstance(block, SkillBlock)
        assert "[SkillError]" in block.block


# ============================================================================
# HANDLER TESTS
# ============================================================================

class TestSkillHandlers:
    """Test individual skill handlers"""

    @pytest.mark.asyncio
    async def test_search_handler_returns_placeholder(self):
        """Test search handler returns placeholder result"""
        result = await handle_search("python asyncio")

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "[Placeholder: Search results]"

    @pytest.mark.asyncio
    async def test_note_handler_returns_placeholder(self):
        """Test note handler returns placeholder result"""
        result = await handle_note("Buy milk")

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "[Placeholder: Note saved]"

    @pytest.mark.asyncio
    async def test_fetch_handler_returns_placeholder(self):
        """Test fetch handler returns placeholder result"""
        result = await handle_fetch("https://example.com")

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "[Placeholder: URL content fetched]"

    @pytest.mark.asyncio
    async def test_scrape_handler_returns_placeholder(self):
        """Test scrape handler returns placeholder result"""
        result = await handle_scrape("https://example.com")

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.output == "[Placeholder: Page scraped]"

    @pytest.mark.asyncio
    async def test_handlers_are_deterministic(self):
        """Test that handlers return consistent results"""
        result1 = await handle_search("test")
        result2 = await handle_search("test")

        assert result1.output == result2.output
        assert result1.success == result2.success


# ============================================================================
# FORMATTING TESTS
# ============================================================================

class TestSkillFormatting:
    """Test skill result and error formatting"""

    def test_format_skill_result(self):
        """Test formatting of successful skill result"""
        result = SkillResult(
            success=True,
            output="[Placeholder: Search results]"
        )

        formatted = format_skill_result("search", "python asyncio", result)

        assert "[SkillResult]" in formatted
        assert "skill: search" in formatted
        assert "args: \"python asyncio\"" in formatted
        assert "output: \"[Placeholder: Search results]\"" in formatted

    def test_format_skill_result_no_args(self):
        """Test formatting with empty args"""
        result = SkillResult(
            success=True,
            output="[Placeholder: Note saved]"
        )

        formatted = format_skill_result("note", "", result)

        assert "[SkillResult]" in formatted
        assert "skill: note" in formatted
        assert "args: \"\"" in formatted

    def test_format_skill_error(self):
        """Test formatting of skill error"""
        formatted = format_skill_error("unknown", "Unknown skill")

        assert "[SkillError]" in formatted
        assert "skill: unknown" in formatted
        assert "error: \"Unknown skill\"" in formatted

    def test_format_skill_error_special_chars(self):
        """Test error formatting with special characters"""
        formatted = format_skill_error("test", "Error: \"Invalid input\"")

        assert "[SkillError]" in formatted
        assert "skill: test" in formatted
        assert "error:" in formatted


# ============================================================================
# PROMPT INJECTION TESTS
# ============================================================================

class TestPromptInjection:
    """Test skill block injection into prompts"""

    def test_inject_skill_block_before_user_text(self):
        """Test that skill block is injected before user text"""
        skill_block = "[SkillResult]\nskill: search\noutput: \"results\""
        user_text = "What did you find?"

        result = inject_skill_block(skill_block, user_text)

        assert result.startswith(skill_block)
        assert result.endswith(user_text)

    def test_inject_preserves_both_parts(self):
        """Test that both skill block and user text are preserved"""
        skill_block = "[SkillResult]\nskill: note"
        user_text = "Did you save it?"

        result = inject_skill_block(skill_block, user_text)

        assert skill_block in result
        assert user_text in result

    def test_inject_no_extra_whitespace_at_ends(self):
        """Test that no extra whitespace is added at start or end"""
        skill_block = "[SkillResult]\nskill: fetch"
        user_text = "Show me the content"

        result = inject_skill_block(skill_block, user_text)

        assert not result.startswith(" ")
        assert not result.startswith("\n")
        assert not result.endswith(" ")
        assert not result.endswith("\n")

    def test_inject_separates_with_newlines(self):
        """Test that skill block and user text are separated"""
        skill_block = "[SkillResult]\nskill: search"
        user_text = "Summarize the results"

        result = inject_skill_block(skill_block, user_text)

        # Should have some separation between block and text
        assert skill_block in result
        assert user_text in result
        # They should not be directly concatenated
        assert f"{skill_block}{user_text}" != result

    def test_inject_empty_user_text(self):
        """Test injection with empty user text"""
        skill_block = "[SkillResult]\nskill: note"
        user_text = ""

        result = inject_skill_block(skill_block, user_text)

        assert skill_block in result

    def test_inject_multiline_skill_block(self):
        """Test injection with multiline skill block"""
        skill_block = "[SkillResult]\nskill: search\nargs: \"test\"\noutput: \"results\""
        user_text = "What did you find?"

        result = inject_skill_block(skill_block, user_text)

        assert result.startswith("[SkillResult]")
        assert user_text in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSkillRouterIntegration:
    """Integration tests for complete skill routing flow"""

    @pytest.mark.asyncio
    async def test_full_flow_detect_route_format(self):
        """Test complete flow: detect → route → format"""
        text = "/search python asyncio"

        # Detect
        trigger = detect_skill(text)
        assert trigger is not None

        # Route
        block = await route_skill(trigger)
        assert isinstance(block, SkillBlock)

        # Verify formatted output
        assert "[SkillResult]" in block.block
        assert "skill: search" in block.block

    @pytest.mark.asyncio
    async def test_full_flow_with_prompt_injection(self):
        """Test complete flow including prompt injection"""
        text = "/note Remember to buy milk"
        user_followup = "Did you save that?"

        # Detect
        trigger = detect_skill(text)
        assert trigger is not None

        # Route
        block = await route_skill(trigger)
        assert isinstance(block, SkillBlock)

        # Inject
        final_prompt = inject_skill_block(block.block, user_followup)
        assert "[SkillResult]" in final_prompt
        assert user_followup in final_prompt
        assert final_prompt.index("[SkillResult]") < final_prompt.index(user_followup)

    @pytest.mark.asyncio
    async def test_full_flow_unknown_skill(self):
        """Test complete flow with unknown skill"""
        text = "/delete something"

        # Detect
        trigger = detect_skill(text)
        assert trigger is not None

        # Route
        block = await route_skill(trigger)
        assert isinstance(block, SkillBlock)

        # Verify error
        assert "[SkillError]" in block.block

    @pytest.mark.asyncio
    async def test_no_skill_detected(self):
        """Test flow when no skill is detected"""
        text = "Just a normal message"

        # Detect
        trigger = detect_skill(text)
        assert trigger is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
