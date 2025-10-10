#!/usr/bin/env python3
"""
Setup Fresh Llama Stack Environment
===================================

This script creates a completely fresh, isolated Llama Stack environment
in a target directory. It sets up uv, installs llama-stack, and creates a
local distribution configuration.

Usage:
    python setup_fresh_llama_stack.py <target_directory>

Example:
    python setup_fresh_llama_stack.py docs2db-llama-stack-3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"🔧 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()

    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )

    if result.returncode != 0 and check:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        sys.exit(1)

    return result


def create_target_directory(target_dir):
    """Create the target directory if it doesn't exist"""
    target_path = Path(target_dir).resolve()

    if target_path.exists():
        print(f"📁 Directory {target_path} already exists")
    else:
        print(f"📁 Creating directory: {target_path}")
        target_path.mkdir(parents=True, exist_ok=True)

    return target_path


def setup_uv_environment(target_path):
    """Set up uv and create virtual environment in target directory"""
    print(f"\n🔧 Setting up uv environment in {target_path}")

    # Change to target directory
    os.chdir(target_path)

    # Initialize uv project with Python 3.12 requirement
    print("📦 Initializing uv project...")
    run_command(["uv", "init", "--no-readme", "--python", "3.12"])

    # Create virtual environment with Python 3.12
    print("🐍 Creating virtual environment with Python 3.12...")
    run_command(["uv", "venv", "--python", "3.12"])

    return target_path


def install_llama_stack(target_path):
    """Install llama-stack and dependencies using uv"""
    print(f"\n📦 Installing llama-stack in {target_path}")

    # Install llama-stack
    print("📦 Installing llama-stack...")
    run_command(["uv", "add", "llama-stack"], cwd=target_path)

    # Install llama-stack-client
    print("📦 Installing llama-stack-client...")
    run_command(["uv", "add", "llama-stack-client"], cwd=target_path)

    # Install additional dependencies
    print("📦 Installing additional dependencies...")
    run_command(
        [
            "uv",
            "add",
            "httpx",
            "pydantic",
            "litellm",
            "sqlalchemy",
            "greenlet",
            "faiss-cpu",
            "ollama",
        ],
        cwd=target_path,
    )


def install_docs2db_rag(target_path):
    """Install Docs2DB RAG package and dependencies"""
    print(f"\n📦 Installing Docs2DB RAG in {target_path}")
    
    # Get the path to the docs2db package (parent of this script's directory)
    script_dir = Path(__file__).parent
    # Go up from demos/llama-stack to docs2db root
    docs2db_path = script_dir.parent.parent
    
    print(f"📦 Installing Docs2DB from: {docs2db_path}")
    run_command(
        ["uv", "add", "--editable", str(docs2db_path)], cwd=target_path
    )
    
    # Install Docs2DB-specific dependencies
    print("📦 Installing Docs2DB RAG dependencies...")
    run_command(
        [
            "uv",
            "add",
            "psycopg2-binary",
            "transformers",
            "torch",
            "sentence-transformers", 
            "scikit-learn",
            "nltk",
        ],
        cwd=target_path,
    )


def setup_docs2db_rag_provider(target_path):
    """Set up Docs2DB RAG provider configuration"""
    print(f"\n⚙️  Setting up Docs2DB RAG provider in {target_path}")
    
    # Create provider configuration directory
    config_dir = target_path / "llama-config"
    providers_dir = config_dir / "providers.d" / "inline" / "tool_runtime"
    providers_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Docs2DB RAG provider configuration
    docs2db_rag_config = {
        "config_class": "docs2db_api.rag.llama_stack.Docs2DBRAGConfig",
        "container_image": None,
        "module": "docs2db_api.rag.llama_stack",
        "pip_packages": [
            "psycopg2-binary",
            "transformers", 
            "torch",
            "sentence-transformers",
            "scikit-learn",
            "nltk",
        ],
        "provider_data_validator": None,
    }
    
    # Write provider config
    provider_config_file = providers_dir / "docs2db_rag.yaml"
    with open(provider_config_file, "w") as f:
        yaml.dump(
            docs2db_rag_config, f, default_flow_style=False, sort_keys=False
        )
    
    print(f"✅ Created Docs2DB RAG provider config: {provider_config_file}")
    return provider_config_file


def create_distribution_config(target_path):
    """Create the llama-stack distribution configuration"""
    print(f"\n⚙️  Creating distribution configuration in {target_path}")

    # Create llama-config directory structure
    config_dir = target_path / "llama-config"
    config_dir.mkdir(exist_ok=True)

    providers_dir = config_dir / "providers.d" / "inline" / "tool_runtime"
    providers_dir.mkdir(parents=True, exist_ok=True)

    data_dir = config_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create the distribution configuration
    distribution = {
        "version": 2,
        "image_name": "docs2db-rag-demo",
        "external_providers_dir": str(providers_dir.parent.parent),
        "apis": [
            "agents",
            "batches",
            "datasetio",
            "eval",
            "files",
            "inference",
            "safety",
            "scoring",
            "telemetry",
            "tool_runtime",
            "vector_io",
        ],
        "providers": {
            "inference": [
                {
                    "provider_id": "ollama",
                    "provider_type": "remote::ollama",
                    "config": {
                        "url": "${env.OLLAMA_URL:=http://localhost:11434}"
                    },
                },
                {
                    "provider_id": "openai",
                    "provider_type": "remote::openai",
                    "config": {
                        "api_key": "${env.OPENAI_API_KEY:=}",
                        "base_url": "${env.OPENAI_BASE_URL:=https://api.openai.com/v1}",
                    },
                },
                {
                    "provider_id": "anthropic",
                    "provider_type": "remote::anthropic",
                    "config": {"api_key": "${env.ANTHROPIC_API_KEY:=}"},
                },
            ],
            "vector_io": [
                {
                    "provider_id": "faiss",
                    "provider_type": "inline::faiss",
                    "config": {
                        "kvstore": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "faiss_store.db"),
                        }
                    },
                }
            ],
            "files": [
                {
                    "provider_id": "meta-reference-files",
                    "provider_type": "inline::localfs",
                    "config": {
                        "storage_dir": str(data_dir / "files"),
                        "metadata_store": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "files_metadata.db"),
                        },
                    },
                }
            ],
            "safety": [
                {
                    "provider_id": "llama-guard",
                    "provider_type": "inline::llama-guard",
                    "config": {"excluded_categories": []},
                }
            ],
            "agents": [
                {
                    "provider_id": "meta-reference",
                    "provider_type": "inline::meta-reference",
                    "config": {
                        "persistence_store": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "agents_store.db"),
                        },
                        "responses_store": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "responses_store.db"),
                        },
                    },
                }
            ],
            "telemetry": [
                {
                    "provider_id": "meta-reference",
                    "provider_type": "inline::meta-reference",
                    "config": {
                        "service_name": "${env.OTEL_SERVICE_NAME:=docs2db-rag-demo}",
                        "sinks": "${env.TELEMETRY_SINKS:=console,sqlite}",
                        "sqlite_db_path": str(data_dir / "trace_store.db"),
                        "otel_exporter_otlp_endpoint": "${env.OTEL_EXPORTER_OTLP_ENDPOINT:=}",
                    },
                }
            ],
            "eval": [
                {
                    "provider_id": "meta-reference",
                    "provider_type": "inline::meta-reference",
                    "config": {
                        "kvstore": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "meta_reference_eval.db"),
                        }
                    },
                }
            ],
            "datasetio": [
                {
                    "provider_id": "localfs",
                    "provider_type": "inline::localfs",
                    "config": {
                        "kvstore": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "localfs_datasetio.db"),
                        }
                    },
                }
            ],
            "scoring": [
                {"provider_id": "basic", "provider_type": "inline::basic"}
            ],
            "tool_runtime": [
                {
                    "provider_id": "docs2db-rag",
                    "provider_type": "inline::docs2db_rag",
                    "config": {
                        "model_name": "${env.DOCS2DB_RAG_MODEL:=granite-30m-english}",
                        "similarity_threshold": "${env.DOCS2DB_RAG_SIMILARITY_THRESHOLD:=0.7}",
                        "max_chunks": "${env.DOCS2DB_RAG_MAX_CHUNKS:=5}",
                        "max_tokens_in_context": "${env.DOCS2DB_RAG_MAX_TOKENS:=4096}",
                        "enable_question_refinement": "${env.DOCS2DB_RAG_ENABLE_REFINEMENT:=true}",
                        "enable_hybrid_search": "${env.DOCS2DB_RAG_ENABLE_HYBRID:=true}",
                    }
                }
            ],
            "batches": [
                {
                    "provider_id": "reference",
                    "provider_type": "inline::reference",
                    "config": {
                        "kvstore": {
                            "type": "sqlite",
                            "db_path": str(data_dir / "batches_store.db"),
                        }
                    },
                }
            ],
        },
        "models": [
            {
                "model_id": "ollama/qwen2.5:7b-instruct",
                "provider_id": "ollama",
                "provider_model_id": "qwen2.5:7b-instruct",
                "metadata": {
                    "description": "Qwen2.5 7B Instruct - excellent at structured output",
                    "context_length": 32768,
                    "supports_tool_calling": True,
                },
            },
            {
                "model_id": "ollama/llama3.1:8b",
                "provider_id": "ollama",
                "provider_model_id": "llama3.1:8b",
                "metadata": {
                    "description": "Llama 3.1 8B - general purpose model",
                    "context_length": 8192,
                    "supports_tool_calling": True,
                },
            },
        ],
        "tool_groups": [
            {
                "toolgroup_id": "docs2db::rag", 
                "provider_id": "docs2db-rag"
            }
        ],
        "server": {"port": "${env.LLAMA_STACK_PORT:=8321}"},
    }
    

    # Write the distribution file
    dist_file = target_path / "docs2db-distribution-local.yaml"
    with open(dist_file, "w") as f:
        yaml.dump(distribution, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created distribution config: {dist_file}")
    return dist_file


def create_startup_script(target_path):
    """Create a startup script for the llama-stack server"""
    startup_script = target_path / "start_server.py"

    script_content = f'''#!/usr/bin/env python3
"""
Start Llama Stack Server
========================

This script starts the Llama Stack server using the local distribution
configuration.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Distribution file path
    dist_file = script_dir / "docs2db-distribution-local.yaml"
    
    if not dist_file.exists():
        print(f"❌ Distribution file not found: {{dist_file}}")
        sys.exit(1)
    
    print(f"🚀 Starting Llama Stack server...")
    print(f"📋 Using distribution: {{dist_file}}")
    print(f"🌐 Server will be available at: http://localhost:8321")
    print()
    
    # Start the server
    cmd = [
        "uv", "run", "llama", "stack", "run",
        str(dist_file),
        "--port", "8321",
        "--image-type", "venv"
    ]
    
    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")

if __name__ == "__main__":
    main()
'''

    with open(startup_script, "w") as f:
        f.write(script_content)

    # Make executable
    startup_script.chmod(0o755)

    print(f"✅ Created startup script: {startup_script}")
    return startup_script


def main():
    parser = argparse.ArgumentParser(
        description="Setup fresh Llama Stack environment"
    )
    parser.add_argument("target_directory", help="Directory to create/setup")

    args = parser.parse_args()

    print(f"🎯 Setting up fresh Llama Stack environment in: {args.target_directory}")
    print()

    try:
        # Step 1: Create target directory
        target_path = create_target_directory(args.target_directory)

        # Step 2: Setup uv environment
        setup_uv_environment(target_path)

        # Step 3: Install llama-stack
        install_llama_stack(target_path)

        # Step 4: Install Docs2DB RAG
        install_docs2db_rag(target_path)

        # Step 5: Setup Docs2DB RAG provider
        setup_docs2db_rag_provider(target_path)

        # Step 6: Create distribution config (now with RAG)
        dist_file = create_distribution_config(target_path)

        # Step 7: Create startup script
        startup_script = create_startup_script(target_path)

        print(f"\n🎉 SUCCESS! Fresh Llama Stack + Docs2DB RAG environment ready!")
        print(f"📁 Location: {target_path}")
        print(f"📋 Distribution: {dist_file}")
        print(f"🚀 Startup script: {startup_script}")
        print()
        print("✅ Configured Features:")
        print("  • Llama Stack server with Ollama integration")
        print("  • Docs2DB RAG with search_documents and search_and_generate tools")
        print("  • RAG Features: Similarity Post Processing, Hybrid Search, Query Refinement")
        print("  • Tool Group: docs2db::rag available for agents")
        print()
        print("Next steps:")
        print(f"  1. cd {args.target_directory}")
        print(f"  2. uv run python start_server.py")
        print("  3. Server will be available at http://localhost:8321")
        print("  4. Use docs2db/demos/llama-stack/client.py to test RAG features")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
