> **Historical import from DungeonOverMind — 2026-09-24.** This note predates the current GenerationEngine contract and contains API assumptions that may now be obsolete. It is preserved only as migration history. Current authority is `docs/CORE-CONTRACT.md` and current provider tests/code.

# OpenAI Responses API Migration: Learnings & Patterns

**Project:** DungeonMind (GenerationEngine, PlayerCharacterGenerator)  
**Timeline:** December 2025  
**Status:** Production-Ready  
**Purpose:** Document patterns and gotchas when migrating from Chat Completions API to Responses API

---

## 🎯 Executive Summary

The OpenAI Responses API is OpenAI's newer API for text generation. Migrating from Chat Completions API requires handling several key differences:

1. **Different parameter names** - `input` vs `messages`, `instructions` vs `system` role
2. **Different usage object attributes** - `input_tokens`/`output_tokens` vs `prompt_tokens`/`completion_tokens`
3. **No `max_tokens` support** - Responses API doesn't have a token limit parameter
4. **Different streaming event types** - `response.output_text.delta` vs `chunk.choices[0].delta.content`

### Key Takeaways

1. **Always use `getattr()` with defaults** for usage object attributes - avoids runtime errors
2. **Remove `max_tokens`** from all Responses API calls - causes `unexpected keyword argument` error
3. **Document API contracts** - save actual API response examples in contracts folder
4. **Map usage fields correctly** - `input_tokens` (not `prompt_tokens`), `output_tokens` (not `completion_tokens`)

---

## 📐 Core Patterns

### Pattern 1: Usage Object Attribute Mapping

**Problem:** Responses API uses different attribute names for token counts than Chat Completions API.

```python
# ❌ WRONG: Chat Completions attribute names
prompt_tokens = usage.prompt_tokens if usage else 0
completion_tokens = usage.completion_tokens if usage else 0
```

**Symptom:** `'ResponseUsage' object has no attribute 'prompt_tokens'`

**Root Cause:** Responses API uses `input_tokens` and `output_tokens` instead of `prompt_tokens` and `completion_tokens`.

**Solution:**

```python
# ✅ CORRECT: Responses API attribute names with safe getattr
prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
```

**Lesson:** Always use `getattr()` with defaults when accessing API response attributes - provides safety and clarity.

---

### Pattern 2: max_tokens Not Supported

**Problem:** Responses API doesn't support the `max_tokens` parameter.

```python
# ❌ WRONG: Passing max_tokens to Responses API
request_kwargs = {
    "model": "gpt-5.1",
    "input": user_prompt,
    "instructions": system_prompt,
    "temperature": 0.7,
    "max_tokens": 2000,  # ❌ Not supported!
}
response = await client.responses.create(**request_kwargs)
```

**Symptom:** `AsyncResponses.create() got an unexpected keyword argument 'max_tokens'`

**Root Cause:** Responses API doesn't have a token limit parameter. The model uses its default behavior.

**Solution:**

```python
# ✅ CORRECT: Remove max_tokens, add warning
request_kwargs = {
    "model": "gpt-5.1",
    "input": user_prompt,
    "instructions": system_prompt,
    "temperature": 0.7,
    # max_tokens NOT supported in Responses API
}

# Log warning if caller tried to set it
if request.max_tokens:
    logger.warning("max_tokens not supported in Responses API, ignoring")

response = await client.responses.create(**request_kwargs)
```

**Lesson:** Responses API intentionally removes token limiting - adjust expectations and document this for callers.

---

### Pattern 3: Request Parameter Mapping

**Problem:** Chat Completions and Responses APIs use different parameter names.

**Chat Completions API:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.7,
    max_tokens=2000,  # Supported
)
content = response.choices[0].message.content
```

**Responses API:**
```python
response = await client.responses.create(
    model="gpt-5.1",
    input=user_prompt,           # NOT messages array
    instructions=system_prompt,   # NOT system role in messages
    temperature=0.7,
    # NO max_tokens
)
content = response.output_text   # NOT choices[0].message.content
```

**Parameter Mapping Table:**

| Chat Completions | Responses API | Notes |
|------------------|---------------|-------|
| `messages[role=system]` | `instructions` | System prompt |
| `messages[role=user]` | `input` | User prompt |
| `choices[0].message.content` | `output_text` | Response content |
| `usage.prompt_tokens` | `usage.input_tokens` | Input token count |
| `usage.completion_tokens` | `usage.output_tokens` | Output token count |
| `max_tokens` | *(not supported)* | Model uses default |

**Lesson:** Create a mapping table when migrating between API versions - reduces errors.

---

### Pattern 4: Service Migration Pattern

**Problem:** Migrating a service from direct OpenAI client to GenerationEngine requires systematic changes.

**Before (Direct OpenAI):**
```python
from openai import OpenAI

class PlayerCharacterGenerator:
    def __init__(self):
        self.openai_client = OpenAI()
        self.model = "gpt-5.2"
    
    async def _call_openai(self, system_prompt, user_prompt):
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_completion_tokens=2000,
        )
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "totalTokens": response.usage.total_tokens,
        }
```

**After (GenerationEngine):**
```python
from generationengine.services.text_service import TextGenerationService
from generationengine.models.requests import TextGenerationRequest, TextModel

class PlayerCharacterGenerator:
    def __init__(self):
        self.text_service = TextGenerationService()
        self.model = TextModel.GPT_5_1
        self.model_name = "gpt-5.1"  # For logging
    
    async def _call_openai(self, system_prompt, user_prompt):
        ge_request = TextGenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=0.7,
            # max_tokens NOT supported in Responses API
        )
        
        response = await self.text_service.generate(
            ge_request,
            service_name="playercharactergenerator"
        )
        
        if not response.success:
            return {"success": False, "error": response.error.message}
        
        return {
            "success": True,
            "content": response.content,
            "totalTokens": response.metrics.tokens_used if response.metrics else 0,
        }
```

**Migration Checklist:**
- [ ] Replace `OpenAI` import with `TextGenerationService` import
- [ ] Replace `self.openai_client` with `self.text_service`
- [ ] Map model string to `TextModel` enum
- [ ] Build `TextGenerationRequest` instead of kwargs dict
- [ ] Remove `max_tokens` / `max_completion_tokens`
- [ ] Handle `TextGenerationResponse` instead of OpenAI response
- [ ] Update health check to reflect new service
- [ ] Test end-to-end

**Lesson:** Create a migration checklist for API changes - ensures nothing is missed.

---

### Pattern 5: Document API Contracts

**Problem:** API response structures change between versions and can be hard to remember.

**Solution:** Save actual API response examples in a contracts folder.

```json
// specs/001-dm-gen-engine/contracts/responses-api-response.json
{
  "description": "OpenAI Responses API response structure",
  "source": "Actual API response from Responses API",
  "date": "2025-12-23",
  "notes": [
    "Responses API uses input_tokens/output_tokens, NOT prompt_tokens/completion_tokens",
    "Usage object structure is different from Chat Completions API"
  ],
  "example_response": {
    "usage": {
      "input_tokens": 328,
      "output_tokens": 52,
      "total_tokens": 380
    }
  },
  "key_differences_from_chat_completions": {
    "usage": {
      "input_tokens": "Equivalent to prompt_tokens in Chat Completions",
      "output_tokens": "Equivalent to completion_tokens in Chat Completions"
    }
  }
}
```

**Lesson:** When debugging API issues, save the actual response structure - future debugging is much faster.

---

## ⚠️ Anti-Patterns

### Anti-Pattern 1: Assuming API Parity

**Problem:** Assuming Responses API has the same features as Chat Completions.

```python
# ❌ BAD: Assuming max_tokens works
request = TextGenerationRequest(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    model=TextModel.GPT_5_1,
    max_tokens=2000,  # Will be ignored or cause error
)
```

**Why It's Bad:** Responses API intentionally has different parameters. Assumptions lead to runtime errors.

**Better Approach:**
```python
# ✅ GOOD: Acknowledge API differences
request = TextGenerationRequest(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    model=TextModel.GPT_5_1,
    # Note: max_tokens not supported in Responses API
)
```

---

### Anti-Pattern 2: Direct Attribute Access

**Problem:** Accessing response attributes directly without safety checks.

```python
# ❌ BAD: Direct attribute access
total_tokens = response.usage.total_tokens
prompt_tokens = response.usage.prompt_tokens  # Crashes!
```

**Why It's Bad:** Attribute names differ between APIs. Direct access causes `AttributeError`.

**Better Approach:**
```python
# ✅ GOOD: Safe attribute access with getattr
usage = response.usage if hasattr(response, "usage") else None
total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
```

---

## 🔍 Debugging Approaches

### When You See: `unexpected keyword argument`

**Symptom:** `AsyncResponses.create() got an unexpected keyword argument 'max_tokens'`

**Diagnosis Steps:**
1. Check which API you're calling (`responses.create` vs `chat.completions.create`)
2. Review the parameter being passed
3. Check if the parameter is supported in that API version

**Root Cause:** Usually a parameter that exists in Chat Completions but not in Responses API.

**Fix:** Remove unsupported parameters, log warnings for callers.

---

### When You See: `object has no attribute`

**Symptom:** `'ResponseUsage' object has no attribute 'prompt_tokens'`

**Diagnosis Steps:**
1. Log the actual response object: `logger.debug(f"Response: {response}")`
2. Check attribute names in the actual response
3. Compare to API documentation or saved contract

**Root Cause:** Attribute names differ between API versions.

**Fix:** Use `getattr()` with correct attribute names and defaults.

---

## 📊 Quick Reference

### Responses API vs Chat Completions

| Feature | Chat Completions | Responses API |
|---------|------------------|---------------|
| System prompt | `messages[role=system]` | `instructions` |
| User prompt | `messages[role=user]` | `input` |
| Response content | `choices[0].message.content` | `output_text` |
| Input tokens | `usage.prompt_tokens` | `usage.input_tokens` |
| Output tokens | `usage.completion_tokens` | `usage.output_tokens` |
| Token limit | `max_tokens` | Not supported |
| Streaming | `stream=True` | `responses.stream()` |

---

## 📚 Related Documents

- **Contract:** `specs/001-dm-gen-engine/contracts/responses-api-response.json`
- **Migration Handoff:** `specs/001-dm-gen-engine/HANDOFF-Responses-API-Migration.md`
- **PCG Migration:** `specs/001-dm-gen-engine/HANDOFF-PlayerCharacterGenerator-Migration.md`
- **TextGenerationService:** `GenerationEngine/src/generationengine/services/text_service.py`

---

**Last Updated:** 2025-12-23  
**Session:** PCG Migration to GenerationEngine

