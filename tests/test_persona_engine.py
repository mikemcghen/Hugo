"""
Tests for Hugo Persona Engine
-------------------------------
Validates Hugo's Right Hand persona enforcement and Jarvis-concise style.

Tests cover:
- Domain detection
- Response compression
- Sentence shortening
- Directness increase
- Jarvis style application
- Word count enforcement
- Anticipatory follow-ups
"""

import pytest
from core.persona_engine import (
    HugoPersonaEngine,
    PersonaContext,
    Domain,
    DomainContext
)


@pytest.fixture
def persona_engine():
    """Create persona engine with Jarvis mode enabled"""
    return HugoPersonaEngine(jarvis_mode=True)


@pytest.fixture
def persona_engine_no_jarvis():
    """Create persona engine with Jarvis mode disabled"""
    return HugoPersonaEngine(jarvis_mode=False)


# Domain Detection Tests

def test_detect_metix_domain(persona_engine):
    """Test detection of Metix domain"""
    text = "Help me with Metix widget 41"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.METIX
    assert domain_ctx.confidence > 0.5
    assert "widget" in [k.lower() for k in domain_ctx.keywords] or "41" in domain_ctx.keywords


def test_detect_sql_domain(persona_engine):
    """Test detection of SQL domain"""
    text = "I need to optimize this SQL query with EF Core"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.SQL
    assert domain_ctx.confidence > 0.5


def test_detect_blazor_domain(persona_engine):
    """Test detection of Blazor domain"""
    text = "How do I create a Blazor component with parameters?"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.BLAZOR
    assert domain_ctx.confidence > 0.5


def test_detect_ottrcal_domain(persona_engine):
    """Test detection of OttrCal domain"""
    text = "Update the OttrCal booking flow"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.OTTRCAL
    assert domain_ctx.confidence > 0.5


def test_detect_homelab_domain(persona_engine):
    """Test detection of homelab domain"""
    text = "Deploy a new Proxmox container"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.HOMELAB
    assert domain_ctx.confidence > 0.5


def test_detect_general_domain(persona_engine):
    """Test fallback to general domain"""
    text = "Hello, how are you?"
    domain_ctx = persona_engine.detect_domain(text)

    assert domain_ctx.domain == Domain.GENERAL


# Response Compression Tests

def test_compress_removes_filler(persona_engine):
    """Test filler word removal"""
    text = "Well, basically I think you should probably just do it."
    compressed = persona_engine.compress_response(text)

    assert "basically" not in compressed.lower()
    assert "probably" not in compressed.lower()
    assert "just" not in compressed.lower()
    assert "I think" not in compressed.lower()


def test_compress_removes_verbose_patterns(persona_engine):
    """Test verbose pattern replacement"""
    text = "Let me help you with this. I would recommend that you consider doing it."
    compressed = persona_engine.compress_response(text)

    assert len(compressed) < len(text)
    assert "Let me help you" not in compressed
    assert "I would recommend that you" not in compressed


def test_compress_cleans_whitespace(persona_engine):
    """Test whitespace cleanup"""
    text = "This  has   extra    spaces ."
    compressed = persona_engine.compress_response(text)

    assert "  " not in compressed
    assert compressed.endswith(".")


# Sentence Shortening Tests

def test_shorten_limits_sentences(persona_engine):
    """Test sentence limiting"""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    shortened = persona_engine.shorten_sentences(text, max_sentences=2)

    sentences = [s.strip() for s in shortened.split('.') if s.strip()]
    assert len(sentences) <= 2


def test_shorten_preserves_short_text(persona_engine):
    """Test short text is preserved"""
    text = "One sentence."
    shortened = persona_engine.shorten_sentences(text, max_sentences=2)

    assert text in shortened


# Directness Tests

def test_increase_directness_converts_questions(persona_engine):
    """Test question conversion"""
    text = "Would you like to deploy the server?"
    direct = persona_engine.increase_directness(text)

    assert "Would you like" not in direct


def test_increase_directness_removes_hedging(persona_engine):
    """Test hedging removal"""
    text = "You might want to consider checking the logs."
    direct = persona_engine.increase_directness(text)

    assert "might want to" not in direct.lower()
    assert "should" in direct.lower() or "consider" in direct.lower()


# Jarvis Style Tests

def test_jarvis_removes_greetings(persona_engine):
    """Test greeting removal in Jarvis mode"""
    text = "Hello! How can I help you today?"
    jarvis = persona_engine.apply_jarvis_style(text)

    assert "Hello" not in jarvis
    assert "How can I help" in jarvis


def test_jarvis_removes_closings(persona_engine):
    """Test closing removal in Jarvis mode"""
    text = "Here's the answer. Let me know if you need anything else."
    jarvis = persona_engine.apply_jarvis_style(text)

    assert "Let me know" not in jarvis


def test_jarvis_removes_first_person(persona_engine):
    """Test first-person removal"""
    text = "I will deploy the server for you."
    jarvis = persona_engine.apply_jarvis_style(text)

    # Should remove "I will" but keep core meaning
    assert len(jarvis) < len(text)


# Word Count Enforcement Tests

def test_enforce_word_count_truncates_long_sentences(persona_engine):
    """Test long sentence truncation"""
    text = "This is a very long sentence with way too many words that should be truncated to fit the requirement."
    enforced = persona_engine.enforce_word_count(text, target_words_per_sentence=(4, 10))

    words = enforced.split()
    assert len(words) <= 10


def test_enforce_word_count_multiple_sentences(persona_engine):
    """Test multiple sentences word count"""
    text = "First sentence with many words. Second sentence also with many words."
    enforced = persona_engine.enforce_word_count(text, target_words_per_sentence=(4, 10))

    sentences = [s.strip() for s in enforced.split('.') if s.strip()]
    for sentence in sentences:
        words = sentence.split()
        assert len(words) <= 10


# Anticipatory Follow-up Tests

def test_generate_followup_metix(persona_engine):
    """Test Metix domain follow-up"""
    domain_ctx = DomainContext(
        domain=Domain.METIX,
        confidence=0.9,
        keywords=["widget", "41"],
        suggested_actions=["SQL", "UX notes", "schema"]
    )

    followup = persona_engine.generate_anticipatory_followup(domain_ctx)

    assert followup is not None
    assert "SQL" in followup
    assert "?" in followup


def test_generate_followup_sql(persona_engine):
    """Test SQL domain follow-up"""
    domain_ctx = DomainContext(
        domain=Domain.SQL,
        confidence=0.9,
        keywords=["query"],
        suggested_actions=["optimize", "migrate", "query"]
    )

    followup = persona_engine.generate_anticipatory_followup(domain_ctx)

    assert followup is not None
    assert any(action in followup for action in ["optimize", "migrate", "query"])


def test_no_followup_general_domain(persona_engine):
    """Test no follow-up for general domain"""
    domain_ctx = DomainContext(
        domain=Domain.GENERAL,
        confidence=1.0,
        keywords=[],
        suggested_actions=[]
    )

    followup = persona_engine.generate_anticipatory_followup(domain_ctx)

    assert followup is None


def test_no_followup_without_jarvis_mode(persona_engine_no_jarvis):
    """Test no follow-up when Jarvis mode disabled"""
    domain_ctx = DomainContext(
        domain=Domain.METIX,
        confidence=0.9,
        keywords=["widget"],
        suggested_actions=["SQL", "UX notes"]
    )

    followup = persona_engine_no_jarvis.generate_anticipatory_followup(domain_ctx)

    assert followup is None


# Full Persona Transform Tests

def test_persona_transform_full_pipeline(persona_engine):
    """Test full persona transformation pipeline"""
    original = "Well, basically I think you should probably consider checking the logs and then maybe restarting the server. I would recommend doing this as soon as possible. Let me know if you need any help with that!"

    transformed = persona_engine.persona_transform(original)

    # Should be much shorter
    assert len(transformed) < len(original) * 0.5

    # Should not have filler
    assert "basically" not in transformed.lower()
    assert "probably" not in transformed.lower()

    # Should be direct
    assert "Let me know" not in transformed


def test_persona_transform_with_domain(persona_engine):
    """Test transformation with domain context"""
    original = "Sure, I can help you with widget 41. Let me explain what you need to do."

    domain_ctx = DomainContext(
        domain=Domain.METIX,
        confidence=0.9,
        keywords=["widget", "41"],
        suggested_actions=["SQL", "UX notes", "schema"]
    )

    transformed = persona_engine.persona_transform(original, domain_ctx)

    # Should be shorter
    assert len(transformed) < len(original)

    # Should have follow-up in Jarvis mode
    assert "SQL" in transformed or "UX" in transformed


def test_detect_and_transform(persona_engine):
    """Test convenience method: detect then transform"""
    user_input = "Help me with Metix widget 41"
    response = "Sure, I can help you with that widget. Let me tell you what to do step by step."

    transformed = persona_engine.detect_and_transform(response, user_input)

    # Should detect Metix domain and apply transformation
    assert len(transformed) < len(response)


# Edge Cases

def test_transform_empty_string(persona_engine):
    """Test transformation of empty string"""
    transformed = persona_engine.persona_transform("")

    assert transformed == ""


def test_transform_very_short_response(persona_engine):
    """Test transformation of already-short response"""
    original = "Done."
    transformed = persona_engine.persona_transform(original)

    # Should preserve short responses
    assert len(transformed) >= len("Done")


def test_transform_with_code_blocks(persona_engine):
    """Test transformation preserves code blocks"""
    original = "Here's the SQL query: SELECT * FROM users WHERE active = 1"
    transformed = persona_engine.persona_transform(original)

    # Should keep SQL
    assert "SELECT" in transformed


# Domain-Specific Action Tests

def test_metix_actions_available(persona_engine):
    """Test Metix domain has correct actions"""
    actions = persona_engine.domain_actions[Domain.METIX]

    assert "SQL" in actions
    assert "UX notes" in actions
    assert "schema" in actions


def test_blazor_actions_available(persona_engine):
    """Test Blazor domain has correct actions"""
    actions = persona_engine.domain_actions[Domain.BLAZOR]

    assert "component" in actions
    assert "parameter" in actions


def test_homelab_actions_available(persona_engine):
    """Test homelab domain has correct actions"""
    actions = persona_engine.domain_actions[Domain.HOMELAB]

    assert "deploy" in actions
    assert "monitor" in actions


# Persona Context Tests

def test_persona_context_creation():
    """Test PersonaContext dataclass"""
    context = PersonaContext(
        recent_turns=[{"role": "user", "content": "test"}],
        last_domain=Domain.METIX,
        ongoing_task="widget refactor",
        user_preferences={"style": "concise"}
    )

    assert context.last_domain == Domain.METIX
    assert context.ongoing_task == "widget refactor"
    assert len(context.recent_turns) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
