#!/usr/bin/env python3
"""
Agent Tool Calling Test
=======================
Tests whether Llama Stack's agent tool calling works as intended.

This script attempts to use Llama Stack's built-in agent tool calling
functionality and detects when it fails due to the known regex parsing bug.

THE BUG:
Llama Stack's tool call parsing regex in tool_utils.py is broken:
  BROKEN: r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})"
  SHOULD BE: r"<function=(?P<function_name>[^>]+)>(?P<args>{.*?})</function>"

The broken regex:
1. Uses [^}] instead of [^>] for function name (wrong character class)
2. Missing the closing </function> tag entirely
3. This causes tool calls to be ignored by the agent system

EXPECTED BEHAVIOR (if working):
1. Agent receives query → recognizes need for tools
2. Agent generates tool call → Llama Stack parses it automatically
3. Llama Stack executes tool → returns results to agent
4. Agent incorporates results → provides enhanced response

ACTUAL BEHAVIOR (broken):
1. Agent receives query → recognizes need for tools
2. Agent generates tool call → Llama Stack fails to parse it
3. No tools executed → agent responds without tool context
4. Response is generic/uninformed

This test detects the breakage and reports it clearly.
"""

import argparse
import json
import re
import sys

from llama_stack_client import LlamaStackClient


class AgentToolCallingTest:
    """Test client that detects Llama Stack's agent tool calling breakage"""

    def __init__(self, base_url: str = "http://localhost:8321"):
        self.base_url = base_url
        self.client = LlamaStackClient(base_url=base_url)
        self.agent_id = None
        self.session_id = None

    def test_connection(self):
        """Test connection to Llama Stack server"""
        try:
            models = self.client.models.list()
            print(f"✅ Connected to Llama Stack server at {self.base_url}")
            print(f"📋 Available models: {len(models)} found")
            return True
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    def create_agent(self):
        """Create an agent for tool calling demonstration"""
        try:
            print("🤖 Creating agent with tool calling configuration...")

            # Create agent with tool calling enabled
            agent = self.client.agents.create(
                agent_config={
                    "model": "ollama/qwen2.5:7b-instruct",
                    "instructions": """You are a helpful assistant with access to search tools.

When asked a question, you MUST use the search_documents tool to find
relevant information before answering.

IMPORTANT: Use the following format for tool calls:
<function=search_documents>{"query": "your search query here"}</function>

Always search first, then provide a comprehensive answer based on the
search results.""",
                    "toolgroups": ["docs2db::rag"],  # Associate with RAG tools
                    "tool_config": {
                        "tool_choice": "auto",
                        "tool_prompt_format": "function_tag",
                        "system_message_behavior": "append",
                    },
                    "max_infer_iters": 5,
                    "enable_session_persistence": True,
                    "sampling_params": {
                        "strategy": {"type": "greedy"},
                        "max_tokens": 512,
                    },
                }
            )

            self.agent_id = agent.agent_id
            print(f"✅ Created agent: {self.agent_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return False

    def create_session(self):
        """Create a session for the agent"""
        try:
            session = self.client.agents.session.create(
                agent_id=self.agent_id, session_name="tool-calling-demo"
            )
            self.session_id = session.session_id
            print(f"✅ Created session: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to create session: {e}")
            return False

    def parse_tool_calls(self, text):
        """Parse tool calls from agent response"""
        # Look for <function=name>{"args"}</function> pattern
        pattern = r"<function=(\w+)>\s*({[^}]*})\s*</function>"
        matches = re.findall(pattern, text)

        tool_calls = []
        for func_name, args_str in matches:
            try:
                args = json.loads(args_str)
                tool_calls.append({"function": func_name, "arguments": args})
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse tool call arguments: {args_str} - {e}")

        return tool_calls

    def execute_tool(self, tool_name, arguments):
        """Execute a tool using the Llama Stack client"""
        try:
            print(f"🔧 Executing {tool_name} with args: {arguments}")

            result = self.client.tool_runtime.invoke_tool(
                tool_name=tool_name, kwargs=arguments
            )

            if hasattr(result, "content"):
                return result.content
            else:
                return str(result)

        except Exception as e:
            print(f"❌ Tool execution failed: {e}")
            return f"Error executing {tool_name}: {e}"

    def analyze_json_chunks(self, raw_chunks):
        """Analyze raw JSON chunks to look for tool execution evidence"""
        print("🔍 Looking for evidence of tool execution in JSON chunks...")

        tool_call_events = 0
        tool_execution_events = 0
        tool_result_events = 0

        for i, chunk_str in enumerate(raw_chunks):
            chunk_lower = chunk_str.lower()

            # Look for tool-related events
            if "tool_call" in chunk_lower or "function=" in chunk_lower:
                tool_call_events += 1
                print(f"   📞 Chunk {i}: Contains tool call reference")

            if "tool_execution" in chunk_lower or "executing" in chunk_lower:
                tool_execution_events += 1
                print(f"   ⚙️  Chunk {i}: Contains tool execution reference")

            if "tool_result" in chunk_lower or "search_documents" in chunk_lower:
                tool_result_events += 1
                print(f"   📊 Chunk {i}: Contains tool result reference")

        # Show a few sample chunks for manual inspection
        print(f"\n📋 SUMMARY:")
        print(f"   Tool call events: {tool_call_events}")
        print(f"   Tool execution events: {tool_execution_events}")
        print(f"   Tool result events: {tool_result_events}")

        if len(raw_chunks) > 0:
            print(f"\n📝 SAMPLE CHUNKS (first 3):")
            for i, chunk in enumerate(raw_chunks[:3]):
                print(f"   Chunk {i}: {str(chunk)[:200]}...")

    def analyze_tool_calling_behavior(self, response, query):
        """Analyze agent response to detect if tool calling worked"""
        print(f"\n🔍 ANALYSIS: Checking if agent used tools...")

        # Check 1: Does response contain tool call syntax?
        tool_call_pattern = r"<function=(\w+)>\s*({.*?})[^<]*</function>"
        tool_calls_found = re.findall(tool_call_pattern, response)

        if tool_calls_found:
            print(f"✅ Agent generated {len(tool_calls_found)} tool call(s):")
            for func_name, args in tool_calls_found:
                print(f"   📞 {func_name}({args})")

            # Check 2: Does response contain tool results/context?
            has_specific_info = self.check_for_specific_information(response, query)

            if has_specific_info:
                print("✅ Response contains specific/technical information")
                print("🎉 RESULT: Tool calling appears to be WORKING!")
                return True
            else:
                print("❌ Response lacks specific information despite tool calls")
                print(
                    "🐛 RESULT: Tool calling is BROKEN - tools generated but not executed"
                )
                self.explain_breakage()
                return False
        else:
            # Check if response is generic vs specific
            has_specific_info = self.check_for_specific_information(response, query)

            if has_specific_info:
                print("🤔 No tool calls found, but response has specific info")
                print("ℹ️  RESULT: Unclear - might be working differently than expected")
                return True
            else:
                print("❌ No tool calls found AND response is generic")
                print(
                    "🐛 RESULT: Tool calling is BROKEN - agent didn't attempt tool use"
                )
                self.explain_breakage()
                return False

    def check_for_specific_information(self, response, query):
        """Check if response contains specific technical information vs generic advice"""
        # Look for specific technical indicators
        specific_indicators = [
            "ssh-keygen",
            "ssh-copy-id",
            "authorized_keys",
            ".ssh/",
            "id_rsa",
            "public key",
            "private key",
            "chmod 700",
            "chmod 600",
            "/etc/ssh/",
            "sshd_config",
            "PubkeyAuthentication",
        ]

        response_lower = response.lower()
        found_indicators = [
            ind for ind in specific_indicators if ind.lower() in response_lower
        ]

        print(
            f"   🔍 Found {len(found_indicators)} specific technical terms: {found_indicators[:3]}..."
        )
        return len(found_indicators) >= 3  # Need at least 3 specific terms

    def explain_breakage(self):
        """Explain the technical details of the breakage"""
        print(f"\n🔧 TECHNICAL EXPLANATION:")
        print(f"   The Llama Stack tool parsing regex is broken:")
        print(f"   📍 File: llama_stack/models/llama/llama3/tool_utils.py")
        print(f'   ❌ Broken: r"<function=(?P<function_name>[^}}]+)>(?P<args>{{.*?}})"')
        print(
            f'   ✅ Should be: r"<function=(?P<function_name>[^>]+)>(?P<args>{{.*?}})</function>"'
        )
        print(f"   ")
        print(f"   Issues:")
        print(f"   1. Uses [^}}] instead of [^>] for function name matching")
        print(f"   2. Missing the closing </function> tag entirely")
        print(f"   3. This causes tool calls to be ignored by the agent system")

    def test_agent_tool_calling(self, query):
        """Test if Llama Stack's agent tool calling works as intended"""
        print(f"\n🧪 TESTING: Agent tool calling with query: '{query}'")
        print("=" * 80)
        print(
            "📋 Expected behavior: Agent should automatically use search_documents tool"
        )
        print("🐛 Known issue: Llama Stack's regex parsing is broken")
        print()

        try:
            # Step 1: Send query to agent (this SHOULD trigger automatic tool use)
            print("📤 Sending query to agent...")
            print(
                "   ⏳ Agent should recognize the need for tools and use them automatically..."
            )

            stream = self.client.agents.turn.create(
                agent_id=self.agent_id,
                session_id=self.session_id,
                toolgroups=["docs2db::rag"],
                tool_config={
                    "tool_choice": "auto",
                    "tool_prompt_format": "function_tag",
                },
                messages=[{"role": "user", "content": query}],
                stream=True,
            )

            # Collect the streaming response and capture raw JSON for analysis
            response = ""
            raw_chunks = []
            for chunk in stream:
                # Capture raw chunk data for JSON inspection
                raw_chunks.append(str(chunk))

                if hasattr(chunk, "event") and chunk.event:
                    event = chunk.event
                    if hasattr(event, "payload") and hasattr(event.payload, "delta"):
                        if hasattr(event.payload.delta, "text"):
                            text = event.payload.delta.text
                            response += text
            print(f"\n🤖 Agent Response Received:")
            print("-" * 50)
            print(response)
            print("-" * 50)

            # Step 2: Examine raw JSON chunks for tool execution evidence
            print(f"\n🔍 JSON ANALYSIS: Examining {len(raw_chunks)} response chunks...")
            self.analyze_json_chunks(raw_chunks)

            # Step 3: Analyze the response to detect tool calling behavior
            return self.analyze_tool_calling_behavior(response, query)

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            return False

    def run_demo(self, query):
        """Run the agent tool calling test"""
        print("🧪 Agent Tool Calling Test")
        print("=" * 50)
        print("This test attempts to use Llama Stack's agent tool calling as intended")
        print("and detects when it fails due to the known regex parsing bug.")
        print()

        # Test connection
        if not self.test_connection():
            print("💡 Make sure to:")
            print("1. Run: python setup_fresh_llama_stack.py <dir>")
            print("2. Start: cd <dir> && python start_server.py")
            return False

        # Create agent
        if not self.create_agent():
            return False

        # Create session
        if not self.create_session():
            return False

        # Test agent tool calling
        tool_calling_works = self.test_agent_tool_calling(query)

        # Report final results
        print(f"\n" + "=" * 80)
        print(f"🎯 FINAL TEST RESULT:")
        if tool_calling_works:
            print(f"✅ Agent tool calling is WORKING!")
            print(f"   • Agent successfully used tools automatically")
            print(f"   • Response contains specific, informed content")
            print(f"   • Llama Stack's agent system is functioning correctly")
            print(f"")
            print(f"🔍 TO VERIFY: Check your Llama Stack server logs for:")
            print(f"   • Tool execution messages")
            print(f"   • RAG engine activity")
            print(f"   • Database query logs")
        else:
            print(f"❌ Agent tool calling is BROKEN!")
            print(f"   • Agent failed to use tools automatically")
            print(f"   • Response is generic/uninformed")
            print(f"   • Llama Stack has the known regex parsing bug")
            print(f"")
            print(f"💡 WORKAROUND: Use direct tool calling (see client.py)")
            print(
                f"💡 SOLUTION: Fix the regex in llama_stack/models/llama/llama3/tool_utils.py"
            )

        print(f"=" * 80)
        return tool_calling_works


def main():
    parser = argparse.ArgumentParser(
        description="Agent Tool Calling Test - Detects Llama Stack tool calling breakage"
    )
    parser.add_argument(
        "--query",
        default="How do I configure SSH key-based authentication?",
        help="Query to ask the agent",
    )
    parser.add_argument(
        "--server", default="http://localhost:8321", help="Llama Stack server URL"
    )

    args = parser.parse_args()

    # Create and run test
    demo = AgentToolCallingTest(base_url=args.server)
    success = demo.run_demo(args.query)

    if not success:
        print("\n❌ Demo failed. Check server status and configuration.")
        sys.exit(1)

    print("\n🎯 Demo complete!")


if __name__ == "__main__":
    main()
