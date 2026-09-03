"""
Claude client — Anthropic API integration with prompt caching.

Implements two query modes (Explore and Deep Dive) with:
  - Ephemeral prompt caching on the system prompt block (~90% token savings
    on repeated queries)
  - Streaming responses via client.messages.stream()
  - Per-request token usage logging
"""
import logging
from typing import AsyncIterator

from anthropic import Anthropic

from config import settings

logger = logging.getLogger("proxios.claude")

# The system prompt is cached as an ephemeral block. This is the single
# most expensive input, so caching it on the Anthropic side dramatically
# reduces cost for repeated queries (~90% savings on system prompt tokens).
SYSTEM_PROMPT = """\
You are PeroxiOS, an AI research assistant specialized exclusively in peroxisome biology, \
peroxisomal biogenesis disorders (PBDs), and related metabolic pathways. You answer based \
ONLY on retrieved scientific literature provided to you. Always cite paper titles and PMC IDs. \
Never speculate beyond the provided context. If evidence is insufficient, say so explicitly.

Modes:
- EXPLORE: Plain language, patient/family/clinician audience. Summarize key findings clearly. \
  Avoid jargon. End with 1-sentence plain English takeaway.
- DEEP_DIVE: Expert researcher audience. Include methodology notes, conflicting evidence, \
  knowledge gaps, and 3 suggested follow-up research questions. Use scientific terminology.
"""

# Maximum tokens to request per citation in the prompt
MAX_FOLLOW_UP_TOKENS = 512


class ClaudeClient:
    """Wraps the Anthropic SDK with prompt caching and streaming."""

    def __init__(self) -> None:
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    def _build_system_block(self) -> list[dict]:
        """Build the system prompt with ephemeral cache control."""
        return [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_user_message(self, context: str, question: str, mode: str) -> str:
        """Build the user message with retrieved context and the question."""
        mode_label = "EXPLORE" if mode == "explore" else "DEEP_DIVE"
        return (
            f"Mode: {mode_label}\n\n"
            f"Retrieved context from PMC open-access literature:\n"
            f"---\n{context}\n---\n\n"
            f"Question: {question}\n\n"
            f"Answer based ONLY on the context above. Cite paper titles and PMC IDs. "
            f"{'End with a 1-sentence plain English takeaway.' if mode == 'explore' else 'End with 3 suggested follow-up research questions.'}"
        )

    async def query_explore(self, context: str, question: str) -> AsyncIterator[dict]:
        """
        Stream an Explore-mode response.

        Yields dicts with keys:
          - "token": str — a text chunk
          - "usage": dict — token usage stats (on final chunk)
        """
        async for chunk in self._stream(
            context=context,
            question=question,
            mode="explore",
            max_tokens=settings.max_tokens_explore,
        ):
            yield chunk

    async def query_deep_dive(self, context: str, question: str) -> AsyncIterator[dict]:
        """
        Stream a Deep Dive-mode response.

        Yields dicts with keys:
          - "token": str — a text chunk
          - "usage": dict — token usage stats (on final chunk)
        """
        async for chunk in self._stream(
            context=context,
            question=question,
            mode="deep_dive",
            max_tokens=settings.max_tokens_deep_dive,
        ):
            yield chunk

    async def _stream(
        self,
        context: str,
        question: str,
        mode: str,
        max_tokens: int,
    ) -> AsyncIterator[dict]:
        """Internal streaming method shared by both modes."""
        user_message = self._build_user_message(context, question, mode)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self._build_system_block(),
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield {"token": text}

            # Log token usage after streaming completes
            final_message = stream.get_final_message()
            usage = final_message.usage
            logger.info(
                "Claude API usage — mode=%s, input_tokens=%d, output_tokens=%d, "
                "cache_read_tokens=%d, cache_creation_tokens=%d",
                mode,
                usage.input_tokens,
                usage.output_tokens,
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
            )
            yield {"usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}}

    async def generate_follow_ups(self, answer: str, question: str) -> list[str]:
        """Generate 3 follow-up research questions based on the answer."""
        prompt = (
            f"Based on this research answer about '{question}', generate exactly 3 "
            f"specific follow-up research questions that a researcher might ask next. "
            f"Return only the questions, one per line, numbered 1-3.\n\nAnswer:\n{answer[:3000]}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_FOLLOW_UP_TOKENS,
            system=[{
                "type": "text",
                "text": "You are a scientific research assistant. Generate concise follow-up questions.",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        lines = response.content[0].text.strip().split("\n")
        follow_ups = []
        for line in lines:
            # Strip numbering like "1. " or "1) "
            cleaned = line.strip()
            for prefix in ["1. ", "2. ", "3. ", "1) ", "2) ", "3) "]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    break
            if cleaned:
                follow_ups.append(cleaned)
            if len(follow_ups) >= 3:
                break

        return follow_ups[:3]
