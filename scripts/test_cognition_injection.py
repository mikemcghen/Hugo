"""
Test script for cognition injection into skills
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_cognition_injection():
    """Test that skills receive cognition instance"""

    print("=" * 70)
    print("PHASE 4.3: Testing Cognition Injection Into Skills")
    print("=" * 70)
    print()

    # Initialize core components
    print("[1/5] Initializing core components...")

    from core.logger import HugoLogger
    from data.sqlite_manager import SQLiteManager
    from core.memory import MemoryManager
    from core.cognition import CognitionEngine
    from skills.skill_manager import SkillManager

    logger = HugoLogger()

    # Use mock memory and sqlite for simpler testing
    from unittest.mock import MagicMock

    sqlite = MagicMock()

    # Create mock memory
    memory = MagicMock()
    memory.classify_memory = MagicMock(return_value=MagicMock(metadata={}))

    async def mock_retrieve_recent(*args, **kwargs):
        return []
    memory.retrieve_recent = mock_retrieve_recent

    # Initialize cognition
    cognition = CognitionEngine(
        memory_manager=memory,
        logger=logger,
        runtime_manager=None
    )

    print("  [OK] Core components initialized")
    print()

    # Initialize skill manager WITH cognition
    print("[2/5] Initializing skill manager with cognition...")

    skill_manager = SkillManager(
        logger=logger,
        sqlite_manager=sqlite,
        memory_manager=memory,
        cognition=cognition
    )

    print("  [OK] Skill manager has cognition:", skill_manager.cognition is not None)
    print()

    # Load skills
    print("[3/5] Loading skills...")

    skill_manager.load_skills()

    skills = skill_manager.list_skills()
    print(f"  [OK] Loaded {len(skills)} skills")
    print()

    # Check extract_and_answer skill
    print("[4/5] Verifying extract_and_answer skill has cognition...")

    extract_skill = skill_manager.get_skill("extract_and_answer")

    if not extract_skill:
        print("  [FAIL] extract_and_answer skill not found!")
        return

    print(f"  Skill name: {extract_skill.name}")
    print(f"  Skill version: {extract_skill.version}")
    print(f"  Cognition available: {extract_skill.cognition is not None}")

    if extract_skill.cognition is None:
        print("  [FAIL] Skill does not have cognition!")
        return

    print("  [OK] Skill has cognition instance")
    print()

    # Test that cognition is the same instance
    print("[5/5] Verifying cognition instance is correct...")

    if extract_skill.cognition is cognition:
        print("  [OK] Skill has the SAME cognition instance (correct!)")
    else:
        print("  [WARN] Skill has a DIFFERENT cognition instance")

    print()
    print("=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    checks = {
        "SkillManager has cognition": skill_manager.cognition is not None,
        "ExtractAndAnswerSkill loaded": extract_skill is not None,
        "ExtractAndAnswerSkill has cognition": extract_skill.cognition is not None,
        "Cognition is same instance": extract_skill.cognition is cognition,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[SUCCESS] All checks passed! Cognition injection working correctly.")
    else:
        print("[FAILURE] Some checks failed. Review the output above.")

    print()


if __name__ == "__main__":
    asyncio.run(test_cognition_injection())
