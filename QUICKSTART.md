# Hugo Quick Start Guide

Get Hugo up and running in 5 minutes!

---

## Prerequisites

- **Python 3.11+** installed
- **Docker Desktop** with GPU support (optional but recommended for voice features)
- **Git** for cloning
- **Anthropic API Key** ([get one here](https://console.anthropic.com/))

---

## Step 1: Install Hugo

```bash
# Clone repository
git clone https://github.com/yourusername/hugo.git
cd hugo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Hugo
pip install -e .
```

---

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
# Minimum required:
#   ANTHROPIC_API_KEY=your_key_here
```

**Important**: Open `.env` in your editor and set at least:
- `ANTHROPIC_API_KEY` - Your Claude API key

---

## Step 3: Start Services

### Option A: With Docker (Recommended)

```bash
# Start all services (database, whisper, piper, claude proxy)
docker-compose -f configs/docker-compose.yaml up -d

# Verify services are running
docker-compose -f configs/docker-compose.yaml ps
```

### Option B: Without Docker (Development)

```bash
# Set SKIP_DOCKER=true in .env
# Hugo will work without voice services
echo "SKIP_DOCKER=true" >> .env
```

---

## Step 4: Start Hugo

```bash
# Start Hugo
hugo up

# Or use Python directly
python -m runtime.cli up
```

You should see:
```
╔════════════════════════════════════════╗
║           HUGO - The Right Hand        ║
║       Your Second-in-Command AI        ║
╚════════════════════════════════════════╝

→ Validating environment...
  ✓ Environment validated
→ Initializing services...
  ✓ Services initialized
→ Connecting to databases...
  ✓ Databases connected
→ Loading core components...
  ✓ Core components loaded
→ Loading state...
  ✓ State loaded
→ Starting scheduler...
  ✓ Scheduler started

✓ Hugo is ready.
```

---

## Step 5: Start Chatting!

```bash
# Enter interactive shell
hugo shell
```

Try these commands:
```
You: Hello Hugo!

You: What can you do?

You: Tell me about your directives

You: Create a new skill called "greeting"
```

---

## Common Commands

```bash
# Check status
hugo status

# View logs
hugo log --tail 50

# Manage skills
hugo skill --list
hugo skill --new my_skill
hugo skill --validate my_skill

# Generate reflection
hugo reflect --days 1

# Stop Hugo
hugo down
```

---

## Troubleshooting

### "No module named 'core'"
```bash
# Make sure you installed Hugo
pip install -e .
```

### "ANTHROPIC_API_KEY not configured"
```bash
# Edit .env and add your API key
nano .env  # or any text editor
```

### Docker services not starting
```bash
# Check Docker is running
docker ps

# Check logs
docker-compose -f configs/docker-compose.yaml logs

# Rebuild services
docker-compose -f configs/docker-compose.yaml build
```

### GPU not detected
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If not working, GPU features will be disabled but Hugo will still work
```

---

## Next Steps

- Read the [full README](README.md) for detailed features
- Explore [Architecture documentation](docs/ARCHITECTURE.md)
- Create your first skill: `hugo skill --new my_first_skill`
- Check out the demo skill: `skills/demo_skill/`

---

## Getting Help

- Check the [documentation](docs/)
- Review [issues](https://github.com/yourusername/hugo/issues)
- Join [discussions](https://github.com/yourusername/hugo/discussions)

---

**You're ready to go! Start exploring Hugo.** 🚀
