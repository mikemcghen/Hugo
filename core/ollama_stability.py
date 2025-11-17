"""
Ollama Stability Module
-----------------------
Enhanced error handling, recovery, and defensive guards for Ollama integration.

Features:
- Automatic retry with reduced context
- Streaming to non-streaming fallback
- Server health detection
- Payload validation
- Comprehensive error recovery
"""

import requests
import json
import time
from typing import Dict, Any, Optional, Generator
from dataclasses import dataclass


@dataclass
class OllamaResponse:
    """Structured Ollama response"""
    success: bool
    content: str
    error: Optional[str] = None
    fallback_used: bool = False
    attempts: int = 1
    duration: float = 0.0
    recovery_action: Optional[str] = None


class OllamaStabilityManager:
    """
    Manages Ollama stability with advanced error recovery.

    Recovery strategies:
    1. Retry with exponential backoff
    2. Reduce context size on 500 errors
    3. Fall back from streaming to non-streaming
    4. Detect server down vs model crash
    5. Automatic model reload detection
    """

    def __init__(self, api_url: str, model_name: str, logger, max_retries: int = 3):
        self.api_url = api_url
        self.model_name = model_name
        self.logger = logger
        self.max_retries = max_retries
        self.retry_backoff = 2
        self.timeout = 120

        # Health tracking
        self.server_available = True
        self.model_loaded = True
        self.consecutive_failures = 0
        self.last_successful_request = time.time()

        # Context reduction on errors
        self.context_reduction_factor = 0.7  # Reduce to 70% on 500 error

    def validate_payload(self, payload: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate Ollama API payload before sending.

        Args:
            payload: Request payload

        Returns:
            (is_valid, error_message)
        """
        # Required fields
        if "model" not in payload:
            return False, "Missing 'model' field"

        if "prompt" not in payload:
            return False, "Missing 'prompt' field"

        # Prompt validation
        prompt = payload["prompt"]
        if not isinstance(prompt, str):
            return False, "Prompt must be string"

        if len(prompt) == 0:
            return False, "Prompt cannot be empty"

        # Warn on very large prompts (may cause 500)
        if len(prompt) > 100000:
            self.logger.log_event("ollama", "large_prompt_warning", {
                "prompt_length": len(prompt),
                "recommendation": "Consider context reduction"
            })

        # Temperature validation
        if "options" in payload and "temperature" in payload["options"]:
            temp = payload["options"]["temperature"]
            if not (0.0 <= temp <= 2.0):
                return False, f"Temperature {temp} out of range [0.0, 2.0]"

        return True, None

    def check_server_health(self) -> bool:
        """
        Check if Ollama server is responsive.

        Returns:
            True if server is healthy
        """
        try:
            # Try to list models (lightweight health check)
            response = requests.get(
                f"{self.api_url.replace('/api/generate', '/api/tags')}",
                timeout=5
            )

            if response.status_code == 200:
                self.server_available = True
                self.logger.log_event("ollama", "server_health_check", {
                    "status": "healthy",
                    "status_code": 200
                })
                return True
            else:
                self.server_available = False
                self.logger.log_event("ollama", "server_health_check", {
                    "status": "unhealthy",
                    "status_code": response.status_code
                })
                return False

        except Exception as e:
            self.server_available = False
            self.logger.log_event("ollama", "server_health_check", {
                "status": "unreachable",
                "error": str(e)
            })
            return False

    def reduce_context(self, prompt: str, reduction_factor: float = 0.7) -> str:
        """
        Reduce prompt context size to recover from 500 errors.

        Args:
            prompt: Original prompt
            reduction_factor: Factor to reduce by (0.7 = keep 70%)

        Returns:
            Reduced prompt
        """
        target_length = int(len(prompt) * reduction_factor)

        # Try to find a natural break point (paragraph, sentence)
        if target_length < len(prompt):
            # Look for last paragraph break
            reduced = prompt[:target_length]
            last_para = reduced.rfind('\n\n')

            if last_para > target_length * 0.8:  # Within 80% of target
                reduced = prompt[:last_para]
            else:
                # Look for last sentence break
                last_sentence = max(
                    reduced.rfind('. '),
                    reduced.rfind('! '),
                    reduced.rfind('? ')
                )
                if last_sentence > target_length * 0.8:
                    reduced = prompt[:last_sentence + 1]

            self.logger.log_event("ollama", "context_reduction", {
                "original_length": len(prompt),
                "reduced_length": len(reduced),
                "reduction_factor": reduction_factor
            })

            return reduced

        return prompt

    def handle_500_error(self, response: requests.Response, attempt: int) -> Optional[str]:
        """
        Handle 500 Server Error from Ollama.

        Common causes:
        - Model overloaded (too much context)
        - Model unloaded mid-generation
        - CUDA out of memory

        Args:
            response: Failed response
            attempt: Current attempt number

        Returns:
            Recovery action suggestion or None
        """
        try:
            error_body = response.json()
            error_msg = error_body.get("error", str(response.text))
        except:
            error_msg = str(response.text)

        self.logger.log_event("ollama", "500_error_analysis", {
            "attempt": attempt,
            "error_message": error_msg,
            "response_headers": dict(response.headers)
        })

        # Detect specific error types
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            return "reduce_context"
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return "model_reload"
        elif "overloaded" in error_msg.lower() or "busy" in error_msg.lower():
            return "retry_with_backoff"
        else:
            return "reduce_context"  # Default recovery

    def soft_fallback_message(self, error_type: str = "general") -> str:
        """
        Generate user-friendly soft fallback message.

        Args:
            error_type: Type of error encountered

        Returns:
            Soft fallback message
        """
        messages = {
            "general": "(My reasoning engine is warming up… let me try that again.)",
            "server_down": "(My reasoning core is restarting… one moment please.)",
            "model_reload": "(Reloading my neural model… just a moment.)",
            "context_too_large": "(That's a lot to process — let me simplify and try again.)",
            "timeout": "(Still thinking… let me approach this differently.)",
            "500_error": "(Adjusting my reasoning parameters… trying again.)"
        }

        return messages.get(error_type, messages["general"])

    def stream_with_recovery(
        self,
        prompt: str,
        temperature: float = 0.7,
        enable_context_reduction: bool = True
    ) -> Generator[str, None, None]:
        """
        Stream with automatic recovery and fallback.

        Strategy:
        1. Try streaming
        2. On 500: reduce context and retry streaming
        3. On persistent failure: fall back to non-streaming
        4. On total failure: yield soft fallback message

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            enable_context_reduction: Whether to reduce context on 500

        Yields:
            Text chunks
        """
        attempt = 0
        current_prompt = prompt
        last_error = None
        used_context_reduction = False

        while attempt < self.max_retries:
            attempt += 1
            start_time = time.time()

            # Build payload
            payload = {
                "model": self.model_name,
                "prompt": current_prompt,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }

            # Validate payload
            is_valid, validation_error = self.validate_payload(payload)
            if not is_valid:
                self.logger.log_event("ollama", "payload_validation_failed", {
                    "error": validation_error,
                    "attempt": attempt
                })
                yield self.soft_fallback_message("general")
                return

            # Log request
            self.logger.log_event("ollama", "ollama_request_payload", {
                "attempt": attempt,
                "prompt_length": len(current_prompt),
                "temperature": temperature,
                "stream": True,
                "context_reduced": used_context_reduction
            })

            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )

                # Log response headers
                self.logger.log_event("ollama", "ollama_response_headers", {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "attempt": attempt
                })

                # Handle 500 error
                if response.status_code == 500:
                    recovery_action = self.handle_500_error(response, attempt)

                    if recovery_action == "reduce_context" and enable_context_reduction and not used_context_reduction:
                        # Try again with reduced context
                        current_prompt = self.reduce_context(current_prompt)
                        used_context_reduction = True

                        self.logger.log_event("ollama", "ollama_server_recovering", {
                            "action": "context_reduction",
                            "attempt": attempt,
                            "retry": True
                        })

                        time.sleep(self.retry_backoff ** attempt)
                        continue

                    elif attempt < self.max_retries:
                        # Retry with backoff
                        self.logger.log_event("ollama", "ollama_server_recovering", {
                            "action": "retry_with_backoff",
                            "attempt": attempt,
                            "backoff": self.retry_backoff ** attempt
                        })

                        time.sleep(self.retry_backoff ** attempt)
                        continue

                response.raise_for_status()

                # Stream chunks
                total_generated = []
                chunk_count = 0

                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))

                            # Log chunk received
                            chunk_count += 1
                            if chunk_count % 50 == 0:  # Log every 50 chunks
                                self.logger.log_event("ollama", "ollama_response_stream_chunk", {
                                    "chunk_number": chunk_count,
                                    "total_length": len("".join(total_generated))
                                })

                            chunk_text = chunk_data.get("response", "")
                            if chunk_text:
                                total_generated.append(chunk_text)
                                yield chunk_text

                            if chunk_data.get("done", False):
                                break

                        except json.JSONDecodeError as e:
                            self.logger.log_event("ollama", "ollama_stream_decode_error", {
                                "error": str(e),
                                "line": line.decode('utf-8', errors='replace')[:100]
                            })
                            continue

                # Success
                duration = time.time() - start_time
                self.logger.log_event("ollama", "ollama_streaming_complete", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "chunks": chunk_count,
                    "response_length": len("".join(total_generated)),
                    "context_reduced": used_context_reduction
                })

                self.consecutive_failures = 0
                self.last_successful_request = time.time()
                return

            except requests.exceptions.HTTPError as e:
                last_error = e
                duration = time.time() - start_time

                self.logger.log_event("ollama", "ollama_streaming_http_error", {
                    "attempt": attempt,
                    "status_code": e.response.status_code if e.response else None,
                    "duration": round(duration, 2),
                    "error": str(e)
                })

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)

            except requests.exceptions.Timeout as e:
                last_error = e
                duration = time.time() - start_time

                self.logger.log_event("ollama", "ollama_streaming_timeout", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "timeout_limit": self.timeout
                })

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)

            except requests.exceptions.ConnectionError as e:
                last_error = e
                duration = time.time() - start_time

                # Check if server is down
                server_healthy = self.check_server_health()

                self.logger.log_event("ollama", "ollama_connection_error", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "server_healthy": server_healthy,
                    "error": str(e)
                })

                if not server_healthy:
                    # Server is down - yield soft fallback immediately
                    yield self.soft_fallback_message("server_down")
                    return

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)

            except Exception as e:
                last_error = e
                duration = time.time() - start_time

                self.logger.log_event("ollama", "ollama_streaming_unexpected_error", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "error_type": type(e).__name__,
                    "error": str(e)
                })

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)

        # All retries exhausted - try non-streaming fallback
        self.logger.log_event("ollama", "fallback_to_nonstreaming", {
            "reason": "streaming_failed_all_attempts",
            "last_error": str(last_error)
        })

        nonstream_result = self.non_stream_fallback(current_prompt, temperature)
        if nonstream_result.success:
            yield nonstream_result.content
        else:
            # Total failure - soft fallback
            self.consecutive_failures += 1
            yield self.soft_fallback_message("500_error")

    def non_stream_fallback(self, prompt: str, temperature: float = 0.7) -> OllamaResponse:
        """
        Non-streaming fallback when streaming fails.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature

        Returns:
            OllamaResponse object
        """
        self.logger.log_event("ollama", "nonstreaming_fallback_attempt", {
            "prompt_length": len(prompt)
        })

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "").strip()

            self.logger.log_event("ollama", "nonstreaming_fallback_success", {
                "response_length": len(content)
            })

            return OllamaResponse(
                success=True,
                content=content,
                recovery_action="nonstreaming_fallback"
            )

        except Exception as e:
            self.logger.log_event("ollama", "nonstreaming_fallback_failed", {
                "error": str(e)
            })

            return OllamaResponse(
                success=False,
                content="",
                error=str(e),
                fallback_used=True
            )
