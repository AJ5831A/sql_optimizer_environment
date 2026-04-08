# ─────────────────────────────────────────────────────────────────────────────
# models.py
#
# Type-safe contracts between the agent and the environment.
# These dataclasses define exactly what the agent sends and receives.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, List, Optional , Union
from pydantic import Field, model_validator
from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────────────────────────────────────────
# ACTION REGISTRY
#
# Single source of truth for all actions.
# Each action declares:
#   name          — human-readable identifier
#   requires      — list of Postgres extensions needed (empty = always available)
#   params_schema — expected params keys and their types (for validation/docs)
#   description   — shown to the agent in legal_actions
#
# To add a new action: add one entry here. Nothing else needs to change.
# ─────────────────────────────────────────────────────────────────────────────

ACTION_REGISTRY: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "add_index_hint",
        "requires": ["pg_hint_plan"],
        "params_schema": {"table": str, "index": str},
        "description": "Add /*+ IndexScan(table index) */ hint to force index usage",
    },
    2: {
        "name": "add_join_order_hint",
        "requires": ["pg_hint_plan"],
        "params_schema": {"table_order": list},
        "description": "Add /*+ Leading(...) */ hint to force join order",
    },
    3: {
        "name": "add_join_method_hint",
        "requires": ["pg_hint_plan"],
        "params_schema": {"table_a": str, "table_b": str, "method": str},
        "description": "Add /*+ HashJoin/NestLoop/MergeJoin(...) */ hint to force join method",
    },
    4: {
        "name": "push_predicate",
        "requires": [],
        "params_schema": {"target_table": str},
        "description": "Move WHERE filter into JOIN ON clause for early filtering",
    },
    5: {
        "name": "replace_subquery_with_join",
        "requires": [],
        "params_schema": {},
        "description": "Rewrite IN (SELECT ...) subquery as an equivalent JOIN",
    },
    6: {
        "name": "remove_redundant_join",
        "requires": [],
        "params_schema": {"table": str},
        "description": "Drop a JOIN whose columns are never referenced in SELECT or WHERE",
    },
    7: {
        "name": "replace_select_star",
        "requires": [],
        "params_schema": {},
        "description": "Expand SELECT * to only the columns actually needed",
    },
    8: {
        "name": "materialize_cte",
        "requires": [],
        "params_schema": {"cte_name": str},
        "description": "Add MATERIALIZED keyword to a WITH clause to force one-time execution",
    },
    9: {
        "name": "submit",
        "requires": [],
        "params_schema": {},
        "description": "End the episode and return the current optimized query",
    },
}


def get_action_name(action_id: int) -> str:
    """Return the name of an action by its ID."""
    return ACTION_REGISTRY[action_id]["name"]


def get_legal_action_ids(available_extensions: List[str]) -> List[int]:
    """
    Return action IDs whose extension requirements are all satisfied.
    The environment calls this at reset() and after each step() to
    compute the legal_actions list sent to the agent.
    """
    ext_set = set(available_extensions)
    return [
        action_id
        for action_id, meta in ACTION_REGISTRY.items()
        if all(ext in ext_set for ext in meta["requires"])
    ]


def build_legal_actions(available_extensions: List[str]) -> List[Dict[str, Any]]:
    """
    Build the full legal_actions list sent inside SQLObservation.
    Each entry contains everything the agent needs to construct a valid SQLAction.
    """
    return [
        {
            "action_id": action_id,
            "name": ACTION_REGISTRY[action_id]["name"],
            # Stringify type objects so the schema is JSON-serializable
            "params_schema": {
                k: v.__name__ if isinstance(v, type) else str(v)
                for k, v in ACTION_REGISTRY[action_id]["params_schema"].items()
            },
            "description": ACTION_REGISTRY[action_id]["description"],
        }
        for action_id in get_legal_action_ids(available_extensions)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class SQLAction(Action):
    """
    What the agent sends to the environment each step.
    """
    action_id: int = 9
    params: Dict[str, Any] = Field(default_factory=dict)
    
    # We use Pydantic's model_validator so validation happens automatically 
    # upon initialization (e.g., action = SQLAction(action_id=1, params={...}))
    @model_validator(mode='after')
    def validate_action_params(self) -> 'SQLAction':
        """Raise ValueError if action_id is unknown or params are missing required keys."""
        if self.action_id not in ACTION_REGISTRY:
            raise ValueError(f"Unknown action_id: {self.action_id}")
            
        schema = ACTION_REGISTRY[self.action_id]["params_schema"]
        
        for key, expected_type in schema.items():
            if key not in self.params:
                raise ValueError(
                    f"action_id={self.action_id} ({get_action_name(self.action_id)}) "
                    f"requires param '{key}'"
                )
            if not isinstance(self.params[key], expected_type):
                raise ValueError(
                    f"Param '{key}' expected {expected_type.__name__}, "
                    f"got {type(self.params[key]).__name__}"
                )
        return self


class SQLObservation(Observation):
    """
    What the agent receives after every reset() and step().
    """
    current_query: str = ""
    observation_vector: List[float] = Field(default_factory=list)
    legal_actions: List[Dict[str, Any]] = Field(default_factory=list)
    explain_plan: Dict[str, Any] = Field(default_factory=dict)
    
    # Inherited fields
    done: bool = False
    
    # Matched base type to satisfy Pylance's invariant checking
    reward: Union[bool, int, float, None] = 0.0


class SQLState(State):
    """
    Episode metadata — tracks the full history of one optimization run.
    Returned by env.state() at any point during an episode.
    """
    episode_id: Optional[str] = None
    original_query: str = ""
    current_query: str = ""
    baseline_time_ms: float = 0.0
    current_time_ms: float = 0.0
    rewrites_applied: List[str] = Field(default_factory=list)
    available_extensions: List[str] = Field(default_factory=list)
    step_count: int = 0
    total_reward: float = 0.0
    improvement_pct: float = 0.0