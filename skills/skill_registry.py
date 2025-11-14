"""
Skill Registry
--------------
Global registry for dynamically loaded skills.

Maintains a singleton registry of all available skills.
"""

from typing import Dict, Optional, List
from .base_skill import BaseSkill


class SkillRegistry:
    """
    Global registry for skill instances.
    
    Provides centralized access to all loaded skills.
    """
    
    _instance = None
    _registry: Dict[str, BaseSkill] = {}
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, name: str, skill_instance: BaseSkill):
        """
        Register a skill instance.
        
        Args:
            name: Unique skill identifier
            skill_instance: Instance of BaseSkill subclass
        """
        self._registry[name] = skill_instance
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a skill.
        
        Args:
            name: Skill identifier
            
        Returns:
            True if skill was removed, False if not found
        """
        if name in self._registry:
            del self._registry[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseSkill]:
        """
        Get a skill by name.
        
        Args:
            name: Skill identifier
            
        Returns:
            BaseSkill instance or None if not found
        """
        return self._registry.get(name)
    
    def all(self) -> Dict[str, BaseSkill]:
        """
        Get all registered skills.
        
        Returns:
            Dictionary mapping skill names to instances
        """
        return self._registry.copy()
    
    def list_names(self) -> List[str]:
        """
        Get list of all registered skill names.
        
        Returns:
            List of skill identifiers
        """
        return list(self._registry.keys())
    
    def clear(self):
        """Clear all registered skills"""
        self._registry.clear()
    
    def count(self) -> int:
        """Get number of registered skills"""
        return len(self._registry)
    
    def exists(self, name: str) -> bool:
        """
        Check if a skill is registered.
        
        Args:
            name: Skill identifier
            
        Returns:
            True if skill exists, False otherwise
        """
        return name in self._registry
    
    def __repr__(self) -> str:
        return f"<SkillRegistry(count={self.count()}, skills={self.list_names()})>"
