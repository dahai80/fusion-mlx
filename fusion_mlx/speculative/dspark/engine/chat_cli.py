#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .api import DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, DSparkGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive DSpark chat on MLX.")
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="MLX target model repo or local path.",
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help="Hugging Face repo or local path for the DSpark draft weights.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculative-tokens", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--verify-mode",
        choices=["full", "lazy-logits"],
        default="full",
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--draft-quant-bits", type=int, default=None)
    parser.add_argument("--draft-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--think",
        action="store_true",
        help=(
            "Enable Qwen3 thinking mode. Off by default: DSpark drafts were "
            "trained non-thinking; <think> traces roughly halve acceptance."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print assistant text as soon as verified tokens are committed.",
    )
    parser.add_argument("--show-stats", action="store_true")
    return parser.parse_args()


def build_prompt(
    history: list[tuple[str, str]], user_message: str, max_turns: int
) -> str:
    if not history:
        return user_message

    turns = history[-max(1, max_turns) :]
    lines = ["Continue this conversation and answer the latest user message."]
    for user, assistant in turns:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print(f"[load target] {args.target_model}")
    print(f"[load draft] {args.draft_model}")
    runner = DSparkGenerator(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_quant_bits=args.draft_quant_bits,
        draft_quant_group_size=args.draft_quant_group_size,
        seed=args.seed,
    )
    print("Type a message. Use /exit or Ctrl-D to quit.\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            break
        if user_message == "/clear":
            history.clear()
            print("[cleared]")
            continue

        prompt = build_prompt(history, user_message, args.max_turns)
        if args.stream:
            print("assistant> ", end="", flush=True)
            final_event = None
            for event in runner.stream(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                confidence_threshold=args.confidence_threshold,
                speculative_tokens=args.speculative_tokens,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
                skip_special_tokens=True,
                enable_thinking=args.think,
            ):
                if event.finished:
                    if event.delta:
                        print(event.delta, end="", flush=True)
                    final_event = event
                elif event.delta:
                    print(event.delta, end="", flush=True)
            print("\n")
            if final_event is None or final_event.metrics is None:
                raise RuntimeError(
                    "Streaming generation did not produce a final event."
                )
            answer = final_event.text.strip()
            metrics = final_event.metrics
        else:
            result = runner.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                confidence_threshold=args.confidence_threshold,
                speculative_tokens=args.speculative_tokens,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
                skip_special_tokens=True,
                enable_thinking=args.think,
            )
            answer = result.text.strip()
            metrics = result.metrics
            print(f"assistant> {answer}\n")
        if args.show_stats:
            print(
                "[stats] "
                f"gen_tps={metrics['generation_tps']:.2f} "
                f"e2e_tps={metrics['end_to_end_tps']:.2f} "
                f"accept={metrics['avg_acceptance_length']:.2f}\n"
            )
        history.append((user_message, answer))


if __name__ == "__main__":
    main()
