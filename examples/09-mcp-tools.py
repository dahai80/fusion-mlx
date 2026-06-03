"""MCP (Model Context Protocol) — list and execute tools."""
import json
import urllib.request

BASE = "http://localhost:8000"

# List available tools
req = urllib.request.Request(f"{BASE}/v1/mcp/tools")
with urllib.request.urlopen(req) as resp:
    tools = json.loads(resp.read())
    print("Available MCP tools:")
    for t in tools.get("tools", []):
        print(f"  - {t['name']}: {t['description']}")

# Execute a tool (replace with an actual tool name from the list above)
TOOL_NAME = "example_tool"
TOOL_ARGS = {"query": "test"}

payload = {"tool_name": TOOL_NAME, "arguments": TOOL_ARGS}
req = urllib.request.Request(
    f"{BASE}/v1/mcp/execute",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"\nTool '{TOOL_NAME}' result:")
        for item in result.get("content", []):
            print(f"  {item.get('text', '')}")
except urllib.error.HTTPError as e:
    print(f"Tool execution failed: {e.code} — tool may not be available")
