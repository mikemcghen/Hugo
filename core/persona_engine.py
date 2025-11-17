"""
Hugo Persona Engine
-------------------
Enforces Hugo's identity, speaking style, and behavioral patterns at the cognition layer.

Hugo is Mike's Right Hand / Second-in-Command with Jarvis-concise communication style.

Core Behaviors:
- Short, precise replies (1-2 sentences, 4-10 words each)
- No rambling or unnecessary explanation
- Anticipates what Mike needs next
- Action-first thinking
- Presents options succinctly
- Domain-aware assistance (Metix, SQL, Blazor, OttrCal, etc.)
- Proactive over reactive
- Calm, confident, minimalistic tone
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Domain(Enum):
    """Detected conversation domains"""
    METIX = "metix"
    OTTRCAL = "ottrcal"
    SQL = "sql"
    BLAZOR = "blazor"
    PERSONAL = "personal"
    HOMELAB = "homelab"
    GENERAL = "general"


@dataclass
class DomainContext:
    """Context for detected domain"""
    domain: Domain
    confidence: float
    keywords: List[str]
    suggested_actions: List[str]


@dataclass
class PersonaContext:
    """Context passed to persona transformation"""
    recent_turns: List[Dict[str, str]]
    last_domain: Optional[Domain]
    ongoing_task: Optional[str]
    user_preferences: Dict[str, any]


class HugoPersonaEngine:
    """
    Enforces Hugo's persona and communication style.

    Transforms LLM outputs to match Hugo's identity as Mike's
    Right Hand with Jarvis-concise communication.
    """

    def __init__(self, jarvis_mode: bool = True):
        """
        Initialize persona engine.

        Args:
            jarvis_mode: Enable maximum conciseness and anticipation
        """
        self.jarvis_mode = jarvis_mode

        # Domain detection patterns
        self.domain_patterns = {
            Domain.METIX: [
                r'\bmetix\b',
                r'\bwidget\s*\d+',
                r'\bwidget\s*(id|ID)',
                r'\b(w|W)idget\s+(41|48|45|50)',
                r'\bux\s+notes?\b',
                r'\bschema\b.*\bwidget\b'
            ],
            Domain.OTTRCAL: [
                r'\bottrcal\b',
                r'\bottr\b',
                r'\bcalendar\s+flow',
                r'\bappointment\b',
                r'\bbooking\b'
            ],
            Domain.SQL: [
                r'\bsql\b',
                r'\bef\s+core\b',
                r'\bentity\s+framework\b',
                r'\bquery\b',
                r'\bdatabase\b',
                r'\btable\b',
                r'\bselect\b',
                r'\bjoin\b',
                r'\bmigration\b'
            ],
            Domain.BLAZOR: [
                r'\bblazor\b',
                r'\brazor\b',
                r'\bcomponent\b',
                r'\b\.razor\b',
                r'\b@code\b',
                r'\bparameter\b'
            ],
            Domain.PERSONAL: [
                r'\bschedule\b',
                r'\btask\b',
                r'\btodo\b',
                r'\bremind\b',
                r'\bappointment\b',
                r'\bmeeting\b'
            ],
            Domain.HOMELAB: [
                r'\bproxmox\b',
                r'\bvm\b',
                r'\bcontainer\b',
                r'\bdocker\b',
                r'\bkubernetes\b',
                r'\bhomelab\b',
                r'\bserver\b',
                r'\bhost\b'
            ]
        }

        # Domain-specific action templates
        self.domain_actions = {
            Domain.METIX: ["SQL", "UX notes", "schema", "refactor", "model"],
            Domain.OTTRCAL: ["flow design", "booking logic", "calendar sync", "validation"],
            Domain.SQL: ["optimize", "migrate", "query", "index", "refactor"],
            Domain.BLAZOR: ["component", "parameter", "event", "render", "state"],
            Domain.PERSONAL: ["schedule", "prioritize", "delegate", "track"],
            Domain.HOMELAB: ["deploy", "monitor", "backup", "scale", "debug"]
        }

        # Filler words to remove
        self.filler_words = [
            r'\bbasically\b',
            r'\bactually\b',
            r'\bjust\b',
            r'\breally\b',
            r'\bvery\b',
            r'\bquite\b',
            r'\bsort of\b',
            r'\bkind of\b',
            r'\bperhaps\b',
            r'\bmaybe\b',
            r'\bpossibly\b',
            r'\bprobably\b',
            r'\bI think\b',
            r'\bI believe\b',
            r'\bin my opinion\b'
        ]

        # Verbose patterns to compress
        self.verbose_patterns = [
            (r'Let me (help you|assist you|explain)', ''),
            (r'I can (help|assist) (you )?with', ''),
            (r'Would you like me to', 'Should I'),
            (r'I would recommend that you', 'Recommend:'),
            (r'It would be a good idea to', ''),
            (r'One thing you could do is', ''),
            (r'You might want to consider', 'Consider'),
            (r'In order to', 'To'),
            (r'For the purpose of', 'To'),
            (r'Due to the fact that', 'Because'),
            (r'At this point in time', 'Now'),
            (r'In the event that', 'If'),
            (r"Here's what (I suggest|I recommend|you should do):", ''),
            (r'^(Sure|Okay|Alright)[,.]?\s*', ''),
            (r'^(Yes|No)[,.]?\s*', '')
        ]

    def detect_domain(self, text: str, context: Optional[PersonaContext] = None) -> DomainContext:
        """
        Detect conversation domain from text.

        Args:
            text: User input or conversation text
            context: Optional persona context for better detection

        Returns:
            DomainContext with detected domain and suggestions
        """
        text_lower = text.lower()
        domain_scores = {}

        # Score each domain
        for domain, patterns in self.domain_patterns.items():
            score = 0
            matched_keywords = []

            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    matched_keywords.extend(matches)

            domain_scores[domain] = (score, matched_keywords)

        # Find highest scoring domain
        if not domain_scores or all(score == 0 for score, _ in domain_scores.values()):
            return DomainContext(
                domain=Domain.GENERAL,
                confidence=1.0,
                keywords=[],
                suggested_actions=[]
            )

        best_domain = max(domain_scores, key=lambda d: domain_scores[d][0])
        best_score, keywords = domain_scores[best_domain]

        # Calculate confidence (normalize by keyword count)
        confidence = min(best_score / 3.0, 1.0)

        # Get suggested actions
        actions = self.domain_actions.get(best_domain, [])

        return DomainContext(
            domain=best_domain,
            confidence=confidence,
            keywords=keywords,
            suggested_actions=actions
        )

    def compress_response(self, text: str) -> str:
        """
        Compress response by removing filler and verbose patterns.

        Args:
            text: Response text to compress

        Returns:
            Compressed text
        """
        compressed = text

        # Remove filler words
        for filler in self.filler_words:
            compressed = re.sub(filler, '', compressed, flags=re.IGNORECASE)

        # Replace verbose patterns
        for pattern, replacement in self.verbose_patterns:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

        # Clean up extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = re.sub(r'\s+([.,!?])', r'\1', compressed)
        compressed = compressed.strip()

        return compressed

    def shorten_sentences(self, text: str, max_sentences: int = 2) -> str:
        """
        Limit response to maximum number of sentences.

        Args:
            text: Response text
            max_sentences: Maximum sentences to keep

        Returns:
            Shortened text
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        # Filter out very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        # Keep first max_sentences
        if len(sentences) > max_sentences:
            kept = sentences[:max_sentences]
            shortened = '. '.join(kept)

            # Add period if needed
            if not shortened.endswith(('.', '!', '?')):
                shortened += '.'

            return shortened

        return text

    def increase_directness(self, text: str) -> str:
        """
        Make response more direct and action-oriented.

        Args:
            text: Response text

        Returns:
            More direct text
        """
        # Convert questions to statements when possible
        text = re.sub(r'Would you like to (.+)\?', r'\1?', text)
        text = re.sub(r'Do you want to (.+)\?', r'\1?', text)

        # Remove hedging
        text = re.sub(r'\bmight want to\b', 'should', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcould consider\b', 'consider', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmay want to\b', 'should', text, flags=re.IGNORECASE)

        # Remove qualifying phrases
        text = re.sub(r'\bif you want,?\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bif you\'d like,?\b', '', text, flags=re.IGNORECASE)

        return text

    def generate_anticipatory_followup(
        self,
        domain_context: DomainContext,
        persona_context: Optional[PersonaContext] = None
    ) -> Optional[str]:
        """
        Generate anticipatory follow-up question or action suggestion.

        Args:
            domain_context: Detected domain context
            persona_context: Optional persona context

        Returns:
            Follow-up text or None
        """
        if not self.jarvis_mode:
            return None

        if domain_context.domain == Domain.GENERAL:
            return None

        actions = domain_context.suggested_actions

        if not actions:
            return None

        # Generate concise options
        if len(actions) == 2:
            return f"{actions[0]} or {actions[1]}?"
        elif len(actions) == 3:
            return f"{actions[0]}, {actions[1]}, or {actions[2]}?"
        elif len(actions) > 3:
            # Top 3 most relevant
            return f"{actions[0]}, {actions[1]}, or {actions[2]}?"

        return None

    def apply_jarvis_style(self, text: str) -> str:
        """
        Apply Jarvis-style conciseness rules.

        Jarvis mode:
        - Maximum conciseness
        - No pleasantries
        - Direct statements
        - Action-focused

        Args:
            text: Response text

        Returns:
            Jarvis-styled text
        """
        # Remove greeting pleasantries
        text = re.sub(r'^(Hello|Hi|Hey)[,.]?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(Good (morning|afternoon|evening))[,.]?\s*', '', text, flags=re.IGNORECASE)

        # Remove closing pleasantries
        text = re.sub(r'Let me know if you (need|have|want) .+[.!]?$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Feel free to .+[.!]?$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'I\'m here to help.?$', '', text, flags=re.IGNORECASE)

        # Convert full sentences to fragments where appropriate
        text = re.sub(r'^I will\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^I\'ll\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^I am\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^I\'m\s+', '', text, flags=re.IGNORECASE)

        return text.strip()

    def enforce_word_count(self, text: str, target_words_per_sentence: Tuple[int, int] = (4, 10)) -> str:
        """
        Enforce word count per sentence (4-10 words).

        Args:
            text: Response text
            target_words_per_sentence: (min, max) words per sentence

        Returns:
            Word-count enforced text
        """
        min_words, max_words = target_words_per_sentence

        sentences = re.split(r'([.!?])\s+', text)
        result = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if not sentence:
                continue

            # Get punctuation if available
            punct = sentences[i + 1] if i + 1 < len(sentences) else '.'

            words = sentence.split()
            word_count = len(words)

            # If too long, truncate to max_words
            if word_count > max_words:
                sentence = ' '.join(words[:max_words])

            result.append(sentence + punct)

        return ' '.join(result).strip()

    def persona_transform(
        self,
        response_text: str,
        domain_context: Optional[DomainContext] = None,
        persona_context: Optional[PersonaContext] = None
    ) -> str:
        """
        Main persona transformation method.

        Applies all Hugo persona rules:
        1. Compress (remove filler)
        2. Shorten (limit sentences)
        3. Increase directness
        4. Apply Jarvis style (if enabled)
        5. Enforce word count
        6. Add anticipatory follow-up (if appropriate)

        Args:
            response_text: Original LLM response
            domain_context: Optional detected domain
            persona_context: Optional conversation context

        Returns:
            Transformed response matching Hugo's persona
        """
        # Step 1: Compress
        transformed = self.compress_response(response_text)

        # Step 2: Increase directness
        transformed = self.increase_directness(transformed)

        # Step 3: Apply Jarvis style (if enabled)
        if self.jarvis_mode:
            transformed = self.apply_jarvis_style(transformed)

        # Step 4: Shorten sentences
        transformed = self.shorten_sentences(transformed, max_sentences=2)

        # Step 5: Enforce word count per sentence
        transformed = self.enforce_word_count(transformed, target_words_per_sentence=(4, 10))

        # Step 6: Add anticipatory follow-up (if domain detected)
        if domain_context and self.jarvis_mode:
            followup = self.generate_anticipatory_followup(domain_context, persona_context)
            if followup:
                transformed = f"{transformed} {followup}"

        # Final cleanup
        transformed = re.sub(r'\s+', ' ', transformed).strip()

        return transformed

    def detect_and_transform(
        self,
        response_text: str,
        user_input: str,
        persona_context: Optional[PersonaContext] = None
    ) -> str:
        """
        Convenience method: detect domain then transform.

        Args:
            response_text: Original LLM response
            user_input: User's input (for domain detection)
            persona_context: Optional conversation context

        Returns:
            Transformed response
        """
        # Detect domain from user input
        domain_context = self.detect_domain(user_input, persona_context)

        # Transform response
        return self.persona_transform(response_text, domain_context, persona_context)
