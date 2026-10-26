#!/usr/bin/env python3
"""Ten-turn coding-agent session over HTTP; prints prompt reuse per turn.

Replays what a client like OpenCode does: each turn resends the whole
conversation, echoes the assistant tool call the server produced (without
reasoning_content, as OpenAI clients do), adds the tool result and a new
instruction. `re-read` is the tokens the server prefilled because no
checkpoint matched; `cached` is what it restored. Compare the summed re-read
with the sum of prompt tokens to see how much of the session was redone.
See docs/AGENT-SESSION.md.
"""

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlsplit

SYSTEM = (
    "You edit C++ in a large repository. Use the edit_file tool for every "
    "change, one call per turn, and keep the diff minimal. " * 12)
TOOLS = [{"type": "function", "function": {
    "name": "edit_file",
    "description": "Replace one exact snippet in a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"},
        "old": {"type": "string"},
        "new": {"type": "string"}},
        "required": ["path", "old", "new"]}}}]
FILE_BODY = "\n".join(
    f"  // line {n}: retry bookkeeping shared by every queue worker"
    for n in range(40))


def post(url, body, timeout):
    target = urlsplit(url)
    cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = cls(target.hostname, target.port, timeout=timeout)
    connection.request("POST", target.path.rstrip("/") + "/v1/chat/completions",
                       json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")
    return connection, response


def parse(connection, response):
    """Accumulate an SSE chat stream into a message plus server usage."""
    started = time.monotonic()
    first = None
    text = []
    calls = {}
    usage = None
    stopped = []
    for line in response:
        if not line.startswith(b"data: "):
            continue
        raw = line[6:].strip()
        if raw == b"[DONE]":
            break
        chunk = json.loads(raw)
        if chunk.get("error"):
            raise RuntimeError(f"server stream error: {chunk['error']}")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                stopped.append(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if first is None and (delta.get("content") or delta.get("tool_calls")):
                first = time.monotonic() - started
            text.append(delta.get("content") or "")
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
    connection.close()
    if usage is None:
        raise RuntimeError(
            "stream ended without stream_options.include_usage"
            + (f" (finish_reason {', '.join(stopped)})" if stopped else ""))
    message = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message, usage, first


def discover(url, timeout):
    target = urlsplit(url)
    cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = cls(target.hostname, target.port, timeout=timeout)
    connection.request("GET", target.path.rstrip("/") + "/v1/models")
    response = connection.getresponse()
    body = json.loads(response.read())
    connection.close()
    if response.status != 200 or not body.get("data"):
        raise RuntimeError(f"cannot discover a served model: {body}")
    return body["data"][0]["id"]


def transcript(message, index):
    """Tool result the client would feed back, growing with the session."""
    calls = message.get("tool_calls") or []
    if not calls:
        return [{"role": "user", "content": "Continue with the next call site."}]
    result = (f"--- src/net/queue.cc (before edit {index}) ---\n{FILE_BODY}\n"
              f"--- end of excerpt {index} ---")
    return [{"role": "tool", "tool_call_id": calls[0]["id"], "content": result}]


def self_test():
    """The delta assembly is the only non-trivial logic here."""
    chunks = [
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a",'
        b'"function":{"name":"edit_file","arguments":"{\\"path\\": \\"q"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":".cc\\"}"}}]}}]}\n\n',
        b'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":9,'
        b'"prompt_tokens_details":{"cached_tokens":4},"gufo":{"prefill_tokens":5}}}\n\n',
        b"data: [DONE]\n\n"]

    class Connection:
        def close(self):
            pass

    message, usage, first = parse(Connection(), iter(chunks))
    call = message["tool_calls"][0]
    assert call["function"] == {"name": "edit_file",
                                "arguments": '{"path": "q.cc"}'}, call
    assert usage["gufo"]["prefill_tokens"] == 5
    assert first is not None
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5815")
    parser.add_argument("--model", help="Served model ID (default: the first served)")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    model = args.model or discover(args.url, args.timeout)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         "In src/net/queue.cc, rename the retry counter to retries."}]
    rows = []
    for turn in range(1, args.turns + 1):
        body = {"model": model, "messages": messages, "tools": TOOLS,
                "tool_choice": "auto", "max_tokens": args.max_tokens,
                "temperature": 0, "seed": turn, "stream": True,
                "stream_options": {"include_usage": True}}
        message, usage, first = parse(*post(args.url, body, args.timeout))
        detail = usage.get("gufo", {})
        prompt = usage["prompt_tokens"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        row = {"turn": turn, "prompt": prompt, "cached": cached,
               "re_read": detail.get("prefill_tokens", prompt - cached),
               "ttft_s": detail.get("ttft_ms", 0) / 1000
               or (first if first is not None else 0),
               "decode_tok_s": usage.get("completion_tokens_per_second", 0),
               "cache_hit": detail.get("cache_hit", cached > 0),
               "miss_reason": detail.get("cache_miss_reason"),
               "common_prefix": detail.get("cache_common_prefix_tokens"),
               "checkpoint": detail.get("cache_checkpoint_tokens")}
        rows.append(row)
        print(f"turn {turn:2d} prompt {prompt:6d} cached {cached:6d} "
              f"re-read {row['re_read']:6d} {row['ttft_s']:6.2f}s "
              f"{row['decode_tok_s']:6.1f} tok/s"
              + (f" {row['miss_reason']} common {row['common_prefix']}"
                 f"/{row['checkpoint']}" if row["miss_reason"] else ""),
              flush=True)
        messages.append(message)
        messages.extend(transcript(message, turn))
        messages.append({"role": "user", "content":
                         f"Now do the same at call site {turn + 1}."})
    total_prompt = sum(row["prompt"] for row in rows)
    total_re_read = sum(row["re_read"] for row in rows)
    report = {"model": model, "turns": rows, "prompt_tokens": total_prompt,
              "re_read_tokens": total_re_read,
              "re_read_fraction": total_re_read / total_prompt if total_prompt else 0,
              "decode_median_tok_s": sorted(
                  row["decode_tok_s"] for row in rows)[len(rows) // 2],
              "first_token_total_s": sum(row["ttft_s"] for row in rows)}
    print(f"re-read {total_re_read} of {total_prompt} prompt tokens "
          f"({100 * report['re_read_fraction']:.1f}%), "
          f"median decode {report['decode_median_tok_s']:.1f} tok/s, "
          f"first tokens {report['first_token_total_s']:.1f}s")
    if args.json:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
