#!/usr/bin/env python3
"""
Agent Tool Calling Demo
=======================
Demonstrates Llama Stack's agent tool calling functionality with RAG integration.

This script shows how agents can automatically use tools to provide informed responses.
"""

import argparse
import sys

from llama_stack_client import LlamaStackClient


class AgentToolCallingDemo:
    """Demo client that showcases Llama Stack's agent tool calling capabilities"""

    def __init__(self, base_url: str = "http://localhost:8321"):
        self.base_url = base_url
        self.client = LlamaStackClient(base_url=base_url)
        self.agent_id = None
        self.session_id = None

    def test_connection(self):
        """Test connection to Llama Stack server"""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def create_agent(self):
        """Create an agent with tool calling configuration"""
        try:
            from llama_stack_client.types import AgentConfig
            
            agent_config = AgentConfig(
                model="ollama/qwen2.5:7b-instruct",
                instructions="""You are a helpful assistant with access to search tools.

When asked a question, you MUST use the search_documents tool to find
relevant information before answering.

IMPORTANT: Use the following format for tool calls:
<function=search_documents>{"query": "your search query here"}</function>

Always search first, then provide a comprehensive answer based on the
search results.""",
                tool_groups=["docs2db::rag"],
            )

            agent = self.client.agents.create(agent_config=agent_config)
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

    def send_query(self, query: str):
        """Send a query to the agent and get the response"""
        try:
            turn = self.client.agents.turn.create(
                agent_id=self.agent_id,
                session_id=self.session_id,
                toolgroups=["docs2db::rag"],
                tool_config={
                    "tool_choice": "auto",
                    "tool_prompt_format": "function_tag",
                },
                messages=[{"role": "user", "content": query}],
                stream=True
            )
            return turn
        except Exception as e:
            print(f"❌ Failed to send query: {e}")
            return None

    def display_response(self, turn):
        """Display the agent's response"""
        if not turn:
            return ""

        response_content = ""
        
        # Process the streaming response
        for chunk in turn:
            if hasattr(chunk, "event") and hasattr(chunk.event, "payload"):
                payload = chunk.event.payload
                
                # Extract text from the response
                if hasattr(payload, "delta") and hasattr(payload.delta, "text"):
                    text = payload.delta.text
                    response_content += text

        return response_content

    def run_demo(self, query: str):
        """Run the complete agent tool calling demo"""
        print("🤖 Agent Tool Calling Demo")
        print("=" * 50)
        print(f"Query: {query}")
        print()

        # Test connection
        if not self.test_connection():
            print("❌ Cannot connect to Llama Stack server")
            return False

        print("✅ Connected to Llama Stack server")

        # Create agent
        if not self.create_agent():
            return False

        # Create session
        if not self.create_session():
            return False
        print()

        # Send query
        print(f"📤 Sending query to agent...")
        turn = self.send_query(query)
        
        if not turn:
            return False

        # Display response
        print("🤖 Agent Response:")
        print("-" * 50)
        
        response = self.display_response(turn)
        
        if response:
            print(response)
        else:
            print("No response received")
        
        print("-" * 50)
        print()
        print("✅ Demo completed!")
        print("💡 Check server logs to verify RAG tool execution")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Agent Tool Calling Demo - Shows Llama Stack agent capabilities"
    )
    parser.add_argument(
        "--query",
        default="How does solar energy compare to other renewable energy sources in terms of efficiency and cost?",
        help="Query to ask the agent",
    )
    parser.add_argument(
        "--server", default="http://localhost:8321", help="Llama Stack server URL"
    )

    args = parser.parse_args()

    # Run the demo
    demo = AgentToolCallingDemo(base_url=args.server)
    demo.run_demo(args.query)


if __name__ == "__main__":
    main()