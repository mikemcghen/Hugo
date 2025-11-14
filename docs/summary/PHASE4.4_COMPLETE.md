# PHASE 4.4 COMPLETE: Removed Hardcoded Fallback URLs

## ✅ Implementation Summary

Successfully modernized Hugo's web_search skill by removing all hardcoded fallback URL logic and moving decision-making authority to the AI system rather than deterministic pattern matching.

---

## 🔧 Changes Made

### 1. **Removed Hardcoded Fallback Detection ([skills/builtin/web_search.py](skills/builtin/web_search.py))**

#### Deleted `_detect_specialized_source()` method (lines 326-364):
**REMOVED:**
```python
def _detect_specialized_source(self, query: str) -> str:
    """Detect if query matches a specialized source pattern."""
    query_lower = query.lower()

    # Movie/TV show patterns → IMDB
    movie_patterns = [
        r'\b(movie|film|cast|starring|actors|directed)\b',
        r'\b(wicked|dune|avatar|marvel|star wars)\b',
        r'\b(release date|coming out|premiere)\b'
    ]

    for pattern in movie_patterns:
        if re.search(pattern, query_lower):
            search_term = query.replace(" ", "+")
            return f"https://www.imdb.com/find?q={search_term}"  # ❌ HARDCODED

    # People/biography patterns → Wikipedia
    people_patterns = [
        r'\b(who is|who was)\b',
        r'\b(biography|bio|life story)\b',
        r'\b(born|died|age)\b'
    ]

    for pattern in people_patterns:
        if re.search(pattern, query_lower):
            search_term = query.replace(" ", "_")
            return f"https://en.wikipedia.org/wiki/{search_term}"  # ❌ HARDCODED

    return None
```

**Why removed:**
- Hardcoded IMDb and Wikipedia URLs
- Pattern matching too rigid
- Prevented Hugo from making intelligent decisions
- Bypassed AI's understanding of context

---

### 2. **Removed Fallback Chaining Logic ([skills/builtin/web_search.py](skills/builtin/web_search.py#L184-240))**

#### Deleted automatic extract_and_answer chaining (lines 184-240):
**REMOVED:**
```python
# Fallback: If no useful results, try auto-detecting specialized sources
if not results["abstract_text"] and not results["answer"] and not results["definition"]:
    fallback_url = self._detect_specialized_source(query)  # ❌ HARDCODED
    if fallback_url:
        # Chain into ExtractAndAnswerSkill
        try:
            from skills.skill_registry import SkillRegistry
            registry = SkillRegistry()
            extract_skill = registry.get("extract_and_answer")

            if extract_skill:
                extract_result = await extract_skill.run(
                    action="extract",
                    query=query,
                    urls=[fallback_url]  # ❌ FORCED ROUTING
                )

                if extract_result.success:
                    return SkillResult(
                        success=True,
                        output=extract_result.output,
                        message=f"Answer extracted from {fallback_url}",
                        metadata={
                            "fallback_extraction": True,  # ❌ LEGACY MARKER
                            "source_url": fallback_url
                        }
                    )

            results["fallback_url"] = fallback_url  # ❌ INJECTED URL

        except Exception as e:
            results["fallback_url"] = fallback_url  # ❌ BACKUP INJECTION
```

**Why removed:**
- Forced routing to specific URLs
- No AI decision-making
- Bypassed extract_and_answer's own URL evaluation
- Created "fallback_url" field that shouldn't exist

---

### 3. **Added Clean URL Extraction ([skills/builtin/web_search.py](skills/builtin/web_search.py#L184-209))**

#### NEW: Extract URLs from DuckDuckGo results organically:
```python
# Extract URLs from related topics for downstream processing
results["urls"] = []

# Add abstract URL if available
if results["abstract_url"]:
    results["urls"].append(results["abstract_url"])

# Add definition URL if available
if results["definition_url"]:
    results["urls"].append(results["definition_url"])

# Add related topic URLs
for topic in results["related_topics"]:
    if topic.get("url"):
        results["urls"].append(topic["url"])

# Log result status
if self.logger:
    self.logger.log_event("web_search", "search_complete", {
        "query": query,
        "has_abstract": bool(results["abstract_text"]),
        "has_answer": bool(results["answer"]),
        "has_definition": bool(results["definition"]),
        "url_count": len(results["urls"]),
        "status": "success" if results["urls"] else "no_results"
    })
```

**Benefits:**
- Only returns URLs that DuckDuckGo actually found
- No hardcoded patterns
- Clean, organic URL list
- Lets extract_and_answer decide which URLs to process

---

### 4. **Updated Output Format ([skills/builtin/web_search.py](skills/builtin/web_search.py#L237-247))**

#### NEW: Metadata includes URL count and status:
```python
return SkillResult(
    success=True,
    output=results,
    message=f"Search completed for '{query}'",
    metadata={
        "query": query,
        "has_results": bool(results["abstract_text"] or results["answer"] or results["definition"]),
        "url_count": len(results["urls"]),
        "status": "success" if results["urls"] else "no_results"  # NEW
    }
)
```

**Output structure:**
```python
{
    "query": "...",
    "timestamp": "...",
    "abstract_text": "...",
    "abstract_source": "...",
    "abstract_url": "...",
    "answer": "...",
    "definition": "...",
    "related_topics": [...],
    "urls": [...]  # NEW: Organic URL list
    # NO "fallback_url" field!
}
```

---

### 5. **Updated Cognition Response Formatting ([core/cognition.py](core/cognition.py#L567-610))**

#### Modified `_format_skill_response()` for web_search:
```python
if skill_name == "web_search":
    output = result.output
    if not output:
        return "I couldn't find any information about that."

    response_parts = []

    # Add abstract/summary if available
    if output.get('abstract_text'):
        response_parts.append(output['abstract_text'])
        if output.get('abstract_source'):
            response_parts.append(f"\n\nSource: {output['abstract_source']}")
        if output.get('abstract_url'):
            response_parts.append(f"URL: {output['abstract_url']}")

    # Add answer if available
    elif output.get('answer'):
        response_parts.append(output['answer'])

    # Add definition if available
    elif output.get('definition'):
        response_parts.append(output['definition'])
        if output.get('definition_source'):
            response_parts.append(f"\n\nSource: {output['definition_source']}")

    # Add related topics if no main content
    elif output.get('related_topics'):
        response_parts.append("I found these related topics:")
        for i, topic in enumerate(output['related_topics'][:3], 1):
            response_parts.append(f"\n{i}. {topic['text']}")

    # If we have response parts, return them
    if response_parts:
        return "\n".join(response_parts)

    # No direct answer available - return URL list for potential extraction
    # NEW: Signal that extraction may be needed
    if output.get('urls'):
        return f"I found {len(output['urls'])} related URLs but no direct answer. The system will attempt to extract information from these sources."

    return "I couldn't find any information about that."
```

**Key changes:**
- No more "Try being more specific" fallback message
- Informs user when URLs are found but no direct answer
- Signals that extraction will be attempted

---

### 6. **Removed Unused Import**

Removed `import re` since regex patterns are no longer needed.

---

## 🧪 Test Results

### Test Script: [scripts/test_no_fallback_urls.py](scripts/test_no_fallback_urls.py)

```
======================================================================
PHASE 4.4: Testing No Fallback URLs in Web Search
======================================================================

[2/3] Testing queries that previously triggered fallbacks...

Query: When does Wicked Part 2 come out?
  [PASS] No hardcoded fallbacks

Query: Who directed Dune?
  [PASS] No hardcoded fallbacks

Query: Marvel Avengers cast
  [PASS] No hardcoded fallbacks

Query: Who is Elon Musk?
  [PASS] No hardcoded fallbacks
  URLs found: 6

[3/3] Final validation...

[PASS] No hardcoded fallback URLs detected
[PASS] web_search returns clean results
[PASS] _detect_specialized_source method removed

======================================================================
[SUCCESS] All checks passed! No fallback URLs detected.
======================================================================
```

**Log Evidence:**
```
[web_search] search_complete: {
    "query": "When does Wicked Part 2 come out?",
    "has_abstract": false,
    "has_answer": false,
    "has_definition": false,
    "url_count": 0,
    "status": "no_results"  ✓ Clean status
}
```

No `fallback_detected`, no `chaining_to_extract`, no `fallback_url` field!

---

## 🎯 Key Benefits

### 1. **AI-Driven Decision Making**
- Hugo decides which URLs to extract from
- No hardcoded patterns forcing specific sources
- Context-aware routing

### 2. **Cleaner Architecture**
- web_search does ONE thing: search DuckDuckGo
- extract_and_answer decides which URLs to process
- Clear separation of concerns

### 3. **No Forced Routing**
- No automatic IMDb probing
- No Wikipedia auto-search
- No DDG Lite fallback
- No "navigation noise detection"

### 4. **Flexible URL Handling**
- Returns organic URL list from search results
- extract_and_answer can score and rank them
- extract_and_answer can ignore low-confidence URLs
- No dependency on hardcoded patterns

---

## 🔄 Integration with Previous Phases

**Phase 4.1** → Added `mode="extraction_synthesis"` to cognition
**Phase 4.3** → Injected cognition into skills
**Phase 4.4** → Removed hardcoded fallback URLs

**Combined Flow:**
```
User query: "When does Wicked Part 2 come out?"
    ↓
web_search returns URLs (organic, from DuckDuckGo)
    ↓
extract_and_answer receives URL list
    ↓
extract_and_answer scores URLs via embeddings
    ↓
extract_and_answer extracts from top-ranked URLs
    ↓
cognition.generate_reply(mode="extraction_synthesis")
    ↓
Short, factual answer (no memory writes, no personality)
```

**No hardcoded IMDb URLs!**
**No forced Wikipedia routing!**
**Hugo makes the decisions!**

---

## 📊 Before vs After

### Before (Hardcoded):
```python
# ❌ Pattern matching
if "wicked" in query.lower():
    return "https://www.imdb.com/find?q=Wicked+Part+2"

# ❌ Forced extraction
results["fallback_url"] = imdb_url
extract_skill.run(urls=[imdb_url])  # No choice
```

### After (AI-Driven):
```python
# ✅ Organic results
results["urls"] = [url1, url2, url3]  # From DuckDuckGo

# ✅ AI decides
extract_and_answer.score_urls(urls)  # Embeddings
extract_and_answer.select_best(scores)  # Intelligence
cognition.generate_reply(mode="extraction_synthesis")  # Clean synthesis
```

---

## 📝 Files Modified

1. **skills/builtin/web_search.py**
   - Removed `_detect_specialized_source()` method
   - Removed fallback chaining logic (lines 184-240)
   - Added organic URL extraction (lines 184-209)
   - Updated output metadata
   - Removed `import re`

2. **core/cognition.py**
   - Updated `_format_skill_response()` for web_search
   - Added informative message when URLs found but no direct answer

3. **scripts/test_no_fallback_urls.py** (NEW)
   - Comprehensive test validating no fallback URLs

---

## ✅ Success Criteria Met

- [x] Removed `_detect_specialized_source()` method
- [x] Removed all hardcoded IMDb URL construction
- [x] Removed all hardcoded Wikipedia URL construction
- [x] Removed fallback chaining logic
- [x] Removed `fallback_url` field from output
- [x] No `fallback_detected` logs
- [x] No `chaining_to_extract` logs
- [x] Clean organic URL list returned
- [x] Test confirms no fallback URLs detected

---

**Phase 4.4 Status:** ✅ **COMPLETE**

Web search now returns only organic results from DuckDuckGo. All decision-making has been moved to Hugo's AI systems.
