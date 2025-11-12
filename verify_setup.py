#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugo Setup Verification Script
==============================
Tests all components of the Hugo local integration.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Test if all required packages are installed"""
    print("🔍 Checking Python dependencies...")

    required_packages = {
        'faiss': 'faiss-cpu',
        'psycopg2': 'psycopg2-binary',
        'sentence_transformers': 'sentence-transformers',
        'requests': 'requests',
        'numpy': 'numpy',
        'dotenv': 'python-dotenv'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("✅ All dependencies installed\n")
    return True


def check_env_file():
    """Check if .env file exists and has required variables"""
    print("🔍 Checking .env configuration...")

    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        print("  ✗ .env file not found")
        return False

    required_vars = [
        'MODEL_ENGINE',
        'MODEL_NAME',
        'OLLAMA_API',
        'ENABLE_FAISS',
        'EMBEDDING_MODEL'
    ]

    from dotenv import load_dotenv
    load_dotenv()

    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}={value}")
        else:
            print(f"  ✗ {var} (not set)")
            missing.append(var)

    if missing:
        print(f"\n❌ Missing variables: {', '.join(missing)}")
        return False

    print("✅ Configuration valid\n")
    return True


def check_ollama():
    """Test connection to Ollama API"""
    print("🔍 Testing Ollama connection...")

    import requests
    from dotenv import load_dotenv
    load_dotenv()

    ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
    model_name = os.getenv("MODEL_NAME", "llama3:8b")

    try:
        # Test version endpoint first
        version_url = ollama_api.replace('/api/generate', '/api/version')
        response = requests.get(version_url, timeout=5)

        if response.status_code == 200:
            print(f"  ✓ Ollama is running: {response.json()}")

            # Test model availability
            print(f"  🔍 Testing model: {model_name}")
            test_response = requests.post(
                ollama_api,
                json={
                    "model": model_name,
                    "prompt": "Say 'OK' if you're working.",
                    "stream": False
                },
                timeout=30
            )

            if test_response.status_code == 200:
                result = test_response.json()
                reply = result.get('response', '').strip()
                print(f"  ✓ Model response: {reply[:50]}...")
                print("✅ Ollama integration working\n")
                return True
            else:
                print(f"  ✗ Model test failed: {test_response.status_code}")
                return False
        else:
            print(f"  ✗ Ollama not responding (status {response.status_code})")
            return False

    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to Ollama")
        print("  → Start Ollama with: ollama serve")
        print("  → Install model with: ollama pull llama3:8b")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_embedding_model():
    """Test SentenceTransformer model loading"""
    print("🔍 Testing embedding model...")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from dotenv import load_dotenv
        load_dotenv()

        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"  Loading model: {model_name}")

        model = SentenceTransformer(model_name)
        print(f"  ✓ Model loaded")

        # Test embedding generation
        test_text = "Hugo is a local-first AI assistant"
        embedding = model.encode(test_text, convert_to_numpy=True)

        print(f"  ✓ Generated embedding: dimension={embedding.shape[0]}")
        print(f"  ✓ Sample values: {embedding[:5]}")

        print("✅ Embedding model working\n")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_faiss():
    """Test FAISS index creation"""
    print("🔍 Testing FAISS index...")

    try:
        import faiss
        import numpy as np

        dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # Create test index
        index = faiss.IndexFlatL2(dimension)
        print(f"  ✓ Created FAISS index (dimension={dimension})")

        # Add test vectors
        test_vectors = np.random.random((5, dimension)).astype('float32')
        index.add(test_vectors)

        print(f"  ✓ Added {index.ntotal} test vectors")

        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, 3)

        print(f"  ✓ Search completed: found {len(indices[0])} results")

        print("✅ FAISS working\n")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_directories():
    """Check if required directories exist"""
    print("🔍 Checking data directories...")

    required_dirs = [
        'data/memory',
        'data/logs'
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = Path(__file__).parent / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            missing.append(dir_path)

    if missing:
        print("\n📝 Creating missing directories...")
        for dir_path in missing:
            full_path = Path(__file__).parent / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_path}")

    print("✅ Directories ready\n")
    return True


def check_core_modules():
    """Test core Hugo modules"""
    print("🔍 Testing core modules...")

    try:
        from core.cognition import CognitionEngine
        print("  ✓ core.cognition")

        from core.memory import MemoryManager
        print("  ✓ core.memory")

        from core.logger import HugoLogger
        print("  ✓ core.logger")

        print("✅ Core modules importable\n")
        return True

    except Exception as e:
        print(f"  ✗ Error importing modules: {e}")
        return False


def run_integration_test():
    """Run a simple integration test"""
    print("🔍 Running integration test...")

    try:
        import asyncio
        from core.memory import MemoryManager, MemoryEntry
        from core.logger import HugoLogger
        from datetime import datetime

        async def test():
            logger = HugoLogger()
            memory = MemoryManager(None, None, logger)

            # Create test entry
            entry = MemoryEntry(
                id=1,
                session_id="test-001",
                timestamp=datetime.now(),
                memory_type="episodic",
                content="Hugo integration test: Ollama connected successfully",
                embedding=None,
                metadata={"test": True},
                importance_score=0.8
            )

            # Store entry
            await memory.store(entry, persist_long_term=False)
            print("  ✓ Memory stored with embedding")

            # Search semantically
            results = await memory.search_semantic("Ollama connection", limit=5)
            print(f"  ✓ Semantic search found {len(results)} results")

            # Check stats
            stats = memory.get_stats()
            print(f"  ✓ Memory stats: cache={stats['cache_size']}, "
                  f"faiss={stats['faiss_index_size']}, "
                  f"enabled={stats['faiss_enabled']}")

            return True

        result = asyncio.run(test())

        if result:
            print("✅ Integration test passed\n")
            return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Hugo Local Integration Verification")
    print("=" * 60)
    print()

    results = {
        "Dependencies": check_imports(),
        "Configuration": check_env_file(),
        "Directories": check_directories(),
        "Core Modules": check_core_modules(),
        "Ollama Connection": check_ollama(),
        "Embedding Model": check_embedding_model(),
        "FAISS Index": check_faiss(),
        "Integration Test": run_integration_test()
    }

    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20s} {status}")

    print()

    if all(results.values()):
        print("🎉 All checks passed! Hugo is ready to run.")
        print()
        print("Next steps:")
        print("  1. Ensure Ollama is running: ollama serve")
        print("  2. Start Hugo: python -m runtime.cli shell")
        print()
        return 0
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("See SETUP_GUIDE.md for troubleshooting help.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
