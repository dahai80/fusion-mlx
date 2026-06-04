"""OpenClaw Agent Protocol example — multi-turn agent with tool calling.

Demonstrates:
1. Creating an agent session with tools
2. Executing turns with the agent
3. Submitting tool results
4. Steering the conversation mid-flow
5. Streaming agent events via SSE
"""
import json
import urllib.request

BASE = "http://localhost:8000/v1/openclaw/agent"


def create_session(model: str = "Qwen2.5-3B-Instruct-4bit") -> str:
       """Create an agent session with tool definitions."""
    tools = [{
         "type": "function",
         "function": {
              "name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "city": {"type": "string", "description": "City name"},
                  },
                  "required": ["city"],
              },
          },
      }]

    payload = {
         "model": model,
         "system_prompt": "You are a helpful assistant with access to weather data.",
         "tools": tools,
      }
    req = urllib.request.Request(
         f"{BASE}/sessions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
      )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"Session created: {result['session_id']}")
    return result["session_id"]


def execute_turn(session_id: str, message: str) -> dict:
       """Execute one agent turn and return the response."""
    payload = {
         "messages": [{"role": "user", "content": message}],
         "max_tokens": 4096,
         "temperature": 0.7,
      }
    req = urllib.request.Request(
         f"{BASE}/turns?session_id={session_id}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
      )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"Agent: {result.get('content', '')[:200]}")
    if result.get("tool_calls"):
        print(f"  Tool calls: {len(result['tool_calls'])}")
        for tc in result["tool_calls"]:
            print(f"    - {tc.get('function', {}).get('name', '?')}")
    return result


def submit_tool_result(session_id: str, tool_call_id: str, result: str):
       """Submit the result of a tool execution back to the agent."""
    payload = {
         "session_id": session_id,
         "tool_call_id": tool_call_id,
         "result": result,
      }
    req = urllib.request.Request(
         f"{BASE}/tool-results",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
      )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def steer_agent(session_id: str, message: dict, mode: str = "append"):
       """Inject a steering message into the conversation."""
    payload = {
         "session_id": session_id,
         "message": message,
         "mode": mode,
      }
    req = urllib.request.Request(
         f"{BASE}/steer",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
      )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"Steered: mode={mode}, messages={result['messages_count']}")
    return result


def get_session_info(session_id: str) -> dict:
       """Get session metadata."""
    req = urllib.request.Request(f"{BASE}/sessions/{session_id}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
       # 1. Create session
    session_id = create_session()

       # 2. Execute a turn
    response = execute_turn(session_id, "What's the weather in Tokyo?")

       # 3. If the agent returned tool calls, submit results
    if response.get("tool_calls"):
        tc = response["tool_calls"][0]
        if hasattr(tc, "get"):
            tool_id = tc.get("id", tc.get("tool_call_id", "call_1"))
            result_data = submit_tool_result(
                session_id,
                tool_id,
                '{"temperature": 22, "condition": "sunny", "humidity": 45}',
               )
            print(f"Tool result accepted: {result_data}")

           # 4. Continue the turn with tool result
        response2 = execute_turn(session_id, "")
        print(f"Final response: {response2.get('content', '')[:300]}")

       # 5. Steer the conversation
    steer_agent(session_id, {
         "role": "system",
         "content": "Now respond in Japanese.",
       }, mode="append")

       # 6. Check session state
    info = get_session_info(session_id)
    print(f"\nSession: turns={info['turn_count']}, active={info['active']}")

       # 7. Clean up
    req = urllib.request.Request(
         f"{BASE}/sessions/{session_id}",
        method="DELETE",
      )
    with urllib.request.urlopen(req) as resp:
        print(f"Deleted: {json.loads(resp.read())}")
