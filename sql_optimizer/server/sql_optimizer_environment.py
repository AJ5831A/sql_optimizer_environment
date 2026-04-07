# ─────────────────────────────────────────────────────────────────────────────
# server/sql_optimizer_environment.py
#
# Core RL environment.
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
from typing import Any, Dict, List, Optional

import sqlglot
import sqlglot.expressions as exp
from sqlglot.expressions.core import Expression

from openenv.core.env_server import Environment

from sql_optimizer.models import (
    ACTION_REGISTRY,
    SQLAction,
    SQLObservation,
    SQLState,
    get_action_name,
)

from sql_optimizer.db import PostgreSQLExecutor


# Inherit with full generics to bind Pydantic parsing automatically
class SQLOptimizerEnvironment(Environment[SQLAction, SQLObservation, SQLState]):
    """
    RL environment that teaches an agent to rewrite slow SQL queries.

    The environment has no knowledge of any schema upfront.
    At reset() time the user provides via kwargs:
      - query:  the slow SQL query they want optimized
      - db_url: connection string to their Postgres database

    The environment then discovers everything it needs automatically
    from pg_catalog and information_schema.
    """

    def __init__(self):
        self._max_steps = int(os.getenv("MAX_STEPS", "10"))

        # Active DB connection — replaced at every reset() call
        self._db: Optional[PostgreSQLExecutor] = None

        # Episode state — all reset at reset() time
        self._episode_id: str = ""
        self._original_query: str = ""
        self._current_query: str = ""
        self._baseline_time_ms: float = 0.0
        self._current_time_ms: float = 0.0
        self._rewrites_applied: List[str] = []
        self._available_extensions: List[str] = []
        self._step_count: int = 0
        self._total_reward: float = 0.0

        # Discovered at reset() time from pg_catalog
        # maps table_name → list of index names
        self._available_indexes: Dict[str, List[str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # CORE INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any
    ) -> SQLObservation:
        """
        Start a new optimization episode.
        """
        query = kwargs.get("query", "")
        db_url = kwargs.get("db_url", "")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty. Pass it via reset kwargs.")
        if not db_url:
            raise ValueError("db_url must be provided. Pass it via reset kwargs.")
        if not query.upper().startswith("SELECT"):
            raise ValueError(
                "Only SELECT queries are supported. "
                "DML statements (INSERT, UPDATE, DELETE) cannot be optimized."
            )

        # Close any previous connection cleanly
        if self._db:
            self._db.close()

        # Connect — db.connect() probes for extensions internally
        self._db = PostgreSQLExecutor(db_url)
        self._db.connect()

        self._available_extensions = self._db.available_extensions

        # Reset episode state
        self._episode_id       = episode_id or str(uuid.uuid4())
        self._original_query   = query
        self._current_query    = query
        self._rewrites_applied = []
        self._step_count       = 0
        self._total_reward     = 0.0

        # Discover indexes from pg_catalog
        self._available_indexes = self._db.get_available_indexes()

        # Warm up the Postgres cache so the baseline is measured accurately from RAM
        self._db.get_explain_plan(self._original_query)
        self._db.get_explain_plan(self._original_query)

        # Measure true baseline
        plan = self._db.get_explain_plan(self._original_query)
        self._baseline_time_ms = plan.get("Execution Time", 999999.0)
        self._current_time_ms  = self._baseline_time_ms

        return self._build_observation(plan=plan, reward=0.0, done=False)

    def step(
        self,
        action: SQLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> SQLObservation:
        """
        Apply one rewrite action to the current query.
        """
        if self._db is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Terminal action
        if action.action_id == 9:
            return self._end_episode()

        # Apply the rewrite
        try:
            rewritten = self._apply_action(self._current_query, action)
        except Exception as e:
            print(f"[env] Rewrite failed for action {action.action_id}: {e}")
            plan = self._db.get_explain_plan(self._current_query)
            return self._build_observation(plan=plan, reward=-0.05, done=False)

        # Verify correctness
        if not self._db.verify_correctness(self._original_query, rewritten):
            print(f"[env] Correctness check failed for action {action.action_id}")
            plan = self._db.get_explain_plan(self._current_query)
            return self._build_observation(plan=plan, reward=-1.0, done=False)

        # Measure new execution time
        new_plan = self._db.get_explain_plan(rewritten)
        new_time = new_plan.get("Execution Time", self._current_time_ms)

        # Reward: normalised improvement relative to baseline
        reward = (self._current_time_ms - new_time) / max(self._baseline_time_ms, 1)

        # Update state
        prev_time = self._current_time_ms
        self._current_query    = rewritten
        self._current_time_ms  = new_time
        self._total_reward    += reward
        self._rewrites_applied.append(get_action_name(action.action_id))

        print(
            f"[env] step={self._step_count} "
            f"action={get_action_name(action.action_id)} "
            f"time={prev_time:.1f}ms → {new_time:.1f}ms "
            f"reward={reward:+.3f}"
        )

        # Check termination
        if self._step_count >= self._max_steps:
            return self._end_episode()

        return self._build_observation(plan=new_plan, reward=reward, done=False)

    @property
    def state(self) -> SQLState:
        improvement = 0.0
        if self._baseline_time_ms > 0:
            improvement = (
                (self._baseline_time_ms - self._current_time_ms)
                / self._baseline_time_ms * 100
            )
        return SQLState(**{
            "episode_id": self._episode_id,
            "original_query": self._original_query,
            "current_query": self._current_query,
            "baseline_time_ms": self._baseline_time_ms,
            "current_time_ms": self._current_time_ms,
            "rewrites_applied": self._rewrites_applied,
            "available_extensions": self._available_extensions,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "improvement_pct": round(improvement, 2),
        })

    def close(self):
        """Cleanup Hook."""
        if self._db:
            self._db.close()
            self._db = None

    # ─────────────────────────────────────────────────────────────────────────
    # LEGAL ACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_legal_actions(
        self, sql: str, plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            tree = sqlglot.parse_one(sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            return [{
                "action_id": 9,
                "name": "submit",
                "params": {},
                "description": "submit current query as final answer",
            }]

        signals     = self._extract_signals(plan)
        tables      = [t.name for t in tree.find_all(exp.Table)]
        joins       = list(tree.find_all(exp.Join))
        inner_joins = [
            j for j in joins
            if j.args.get("kind") in (None, "INNER", "")
        ]
        has_subquery = bool(list(tree.find_all(exp.Subquery)))
        has_cte      = bool(tree.find(exp.With))
        has_star     = bool(tree.find(exp.Star))
        where        = tree.find(exp.Where)
        used_tables  = {c.table for c in tree.find_all(exp.Column) if c.table}

        def applicable(action_id: int) -> List[Dict[str, Any]]:
            has_seq_scan = signals[2] == 1.0
            rows_removed = signals[4]

            if action_id == 1:
                if not (has_seq_scan and rows_removed > 1000):
                    return []
                return [
                    {
                        "params": {"table": table, "index": index},
                        "description": f"hint: use index {index} on {table}",
                    }
                    for table in tables
                    for index in self._available_indexes.get(table, [])
                ]

            if action_id == 2:
                if len(inner_joins) < 2:
                    return []
                return [{
                    "params": {"table_order": tables},
                    "description": f"hint: join order {' → '.join(tables)}",
                }]

            if action_id == 3:
                if len(inner_joins) < 1 or len(tables) < 2:
                    return []
                return [
                    {
                        "params": {
                            "table_a": tables[0],
                            "table_b": tables[1],
                            "method": method,
                        },
                        "description": f"hint: use {method} for {tables[0]} × {tables[1]}",
                    }
                    for method in ["HashJoin", "NestLoop", "MergeJoin"]
                ]

            if action_id == 4:
                if not (where and joins):
                    return []
                for condition in where.find_all(exp.EQ):
                    col = condition.find(exp.Column)
                    if col and col.table:
                        return [{
                            "params": {"target_table": col.table},
                            "description": f"push predicate on {col.table} into JOIN ON",
                        }]
                return []

            if action_id == 5:
                if not has_subquery:
                    return []
                return [{"params": {}, "description": "replace IN (SELECT...) subquery with JOIN"}]

            if action_id == 6:
                candidates = []
                for join in joins:
                    t = join.find(exp.Table)
                    a = join.find(exp.TableAlias)
                    ref = (a.name if a else t.name) if t else None
                    if ref and ref not in used_tables:
                        candidates.append({
                            "params": {"table": ref},
                            "description": f"remove unused JOIN on {ref}",
                        })
                return candidates

            if action_id == 7:
                if not has_star:
                    return []
                return [{"params": {}, "description": "expand SELECT * to explicit column list"}]

            if action_id == 8:
                if not has_cte:
                    return []
                return [{"params": {}, "description": "add MATERIALIZED to CTE"}]

            if action_id == 9:
                return [{"params": {}, "description": "submit current query as final answer"}]

            return []

        legal = []
        for action_id, meta in ACTION_REGISTRY.items():
            if not all(ext in self._available_extensions for ext in meta["requires"]):
                continue
            for match in applicable(action_id):
                legal.append({
                    "action_id": action_id,
                    "name": meta["name"],
                    "params": match["params"],
                    "description": match["description"],
                })

        return legal

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self, sql: str, action: SQLAction) -> str:
        dispatch = {
            1: self._add_index_hint,
            2: self._add_join_order_hint,
            3: self._add_join_method_hint,
            4: self._push_predicate,
            5: self._replace_subquery_with_join,
            6: self._remove_redundant_join,
            7: self._replace_select_star,
            8: self._materialize_cte,
        }
        fn = dispatch.get(action.action_id)
        if fn is None:
            return sql
        return fn(sql, **action.params)

    def _add_hint_comment(self, sql: str, new_hint: str) -> str:
        stripped = sql.strip()
        if stripped.startswith("/*+"):
            closing = stripped.index("*/")
            return stripped[:closing].rstrip() + f" {new_hint}" + stripped[closing:]
        return f"/*+ {new_hint} */\n{stripped}"

    def _add_index_hint(self, sql: str, table: str, index: str, **_) -> str:
        return self._add_hint_comment(sql, f"IndexScan({table} {index})")

    def _add_join_order_hint(self, sql: str, table_order: List[str], **_) -> str:
        return self._add_hint_comment(sql, f"Leading({' '.join(table_order)})")

    def _add_join_method_hint(
        self, sql: str, table_a: str, table_b: str, method: str, **_
    ) -> str:
        return self._add_hint_comment(sql, f"{method}({table_a} {table_b})")

    def _push_predicate(self, sql: str, target_table: str, **_) -> str:
        hint_prefix, clean_sql = self._split_hint(sql)

        tree = sqlglot.parse_one(clean_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
        where = tree.find(exp.Where)
        if not where:
            return sql

        to_push = []
        for cond in where.find_all(exp.EQ):
            col = cond.find(exp.Column)
            if col and col.table == target_table:
                to_push.append(cond.copy())

        if not to_push:
            return sql

        def remove_pushed(node: Expression) -> Optional[Expression]:
            if isinstance(node, exp.EQ):
                col = node.find(exp.Column)
                if col and col.table == target_table:
                    return None
            return node

        new_where_expr = where.this.transform(remove_pushed)
        if new_where_expr is None:
            where.pop()
        else:
            where.set("this", new_where_expr)

        for join in tree.find_all(exp.Join):
            t = join.find(exp.Table)
            if t and t.name == target_table:
                existing = join.args.get("on")
                combined: Expression = to_push[0]
                for extra in to_push[1:]:
                    combined = exp.And(this=combined, expression=extra)
                if existing:
                    join.set("on", exp.And(this=existing, expression=combined))
                else:
                    join.set("on", combined)
                break

        result = tree.sql(dialect="postgres", comments=True)
        return hint_prefix + result if hint_prefix else result

    def _replace_subquery_with_join(self, sql: str, **_) -> str:
        hint_prefix, clean_sql = self._split_hint(sql)
        tree = sqlglot.parse_one(clean_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)

        for in_expr in list(tree.find_all(exp.In)):
            subquery = in_expr.args.get("query")
            if not subquery:
                continue

            outer_col   = in_expr.this
            inner_table = subquery.find(exp.Table)
            inner_col   = subquery.find(exp.Column)
            if not all([outer_col, inner_table, inner_col]):
                continue

            alias = f"sq_{inner_table.name}"

            # FIX: Use exp.alias_ helper for correct node structure
            aliased = exp.alias_(inner_table.copy(), alias)

            new_join = exp.Join(
                this=aliased,
                on=exp.EQ(
                    this=outer_col.copy(),
                    expression=exp.Column(
                        this=exp.Identifier(this=inner_col.name),
                        table=exp.Identifier(this=alias),
                    ),
                ),
                kind="INNER",
            )

            inner_where = subquery.find(exp.Where)
            if inner_where:
                outer_where = tree.find(exp.Where)
                if outer_where:
                    outer_where.set(
                        "this",
                        exp.And(
                            this=outer_where.this,
                            expression=inner_where.this.copy()
                        )
                    )

            select_node = tree.find(exp.Select)
            if select_node:
                existing_joins = select_node.args.get("joins") or []
                select_node.set("joins", existing_joins + [new_join])

            # FIX: Replace in_expr with TRUE instead of popping it
            in_expr.replace(exp.Boolean(this=True))

        result = tree.sql(dialect="postgres", comments=True)
        return hint_prefix + result if hint_prefix else result

    def _remove_redundant_join(self, sql: str, table: str, **_) -> str:
        hint_prefix, clean_sql = self._split_hint(sql)

        tree = sqlglot.parse_one(clean_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
        used = {c.table for c in tree.find_all(exp.Column) if c.table}

        for join in tree.find_all(exp.Join):
            t = join.find(exp.Table)
            a = join.find(exp.TableAlias)
            ref = (a.name if a else t.name) if t else None
            if ref == table and ref not in used:
                join.pop()
                break

        result = tree.sql(dialect="postgres", comments=True)
        return hint_prefix + result if hint_prefix else result

    def _replace_select_star(self, sql: str, **_) -> str:
        if not self._db:
            return sql

        hint_prefix, clean_sql = self._split_hint(sql)
        tree = sqlglot.parse_one(clean_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
        if not tree.find(exp.Star):
            return sql

        explicit = []
        top_level_tables = []

        # FIX: Restrict table discovery to direct FROM/JOIN tables only
        from_node = tree.find(exp.From)
        if from_node:
            for t in from_node.find_all(exp.Table):
                top_level_tables.append(t)

        for join in tree.find_all(exp.Join):
            t = join.find(exp.Table)
            if t:
                top_level_tables.append(t)

        for t in top_level_tables:
            # FIX: t.alias is a plain string in sqlglot
            alias_name = t.alias or t.name   
            columns = self._db.get_column_names(t.name)
            for col_name in columns:
                explicit.append(
                    exp.Column(
                        this=exp.Identifier(this=col_name),
                        table=exp.Identifier(this=alias_name),
                    )
                )

        if explicit:
            select_node = tree.find(exp.Select)
            if select_node:
                select_node.set("expressions", explicit)

        result = tree.sql(dialect="postgres", comments=True)
        return hint_prefix + result if hint_prefix else result

    def _materialize_cte(self, sql: str, cte_name: str = "", **_) -> str:
        hint_prefix, clean_sql = self._split_hint(sql)

        tree = sqlglot.parse_one(clean_sql, dialect="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
        for cte in tree.find_all(exp.CTE):
            if not cte_name or cte.alias == cte_name:
                # FIX: Set materialized directly on the CTE node, not the inner Select
                if not cte.args.get("materialized"):
                    cte.set("materialized", True)
                break

        result = tree.sql(dialect="postgres", comments=True)
        return hint_prefix + result if hint_prefix else result

    # ─────────────────────────────────────────────────────────────────────────
    # HINT COMMENT HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _split_hint(self, sql: str):
        stripped = sql.strip()
        if stripped.startswith("/*+"):
            try:
                closing = stripped.index("*/")
                hint_prefix = stripped[: closing + 2] + "\n"
                clean_sql   = stripped[closing + 2:].strip()
                return hint_prefix, clean_sql
            except ValueError:
                pass
        return "", stripped

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_signals(self, plan: Dict[str, Any]) -> List[float]:
        signals = {
            "execution_time_ms":       plan.get("Execution Time", 0.0),
            "total_plan_cost":         0.0,
            "has_seq_scan":            0.0,
            "has_subquery":            0.0,
            "max_rows_removed":        0.0,
            "num_joins":               0.0,
            "has_redundant_join":      0.0,
            "has_cte":                 0.0,
            "has_select_star":         0.0,
            "estimated_vs_actual_gap": 0.0,
        }

        def walk(node: Dict[str, Any]) -> None:
            node_type = node.get("Node Type", "")
            signals["total_plan_cost"] = max(
                signals["total_plan_cost"], node.get("Total Cost", 0.0)
            )
            if node_type == "Seq Scan":
                signals["has_seq_scan"] = 1.0
                signals["max_rows_removed"] = max(
                    signals["max_rows_removed"],
                    float(node.get("Rows Removed by Filter", 0))
                )
                signals["estimated_vs_actual_gap"] = max(
                    signals["estimated_vs_actual_gap"],
                    float(abs(node.get("Plan Rows", 0) - node.get("Actual Rows", 0)))
                )
            if node_type in ("Subquery Scan", "InitPlan"):
                signals["has_subquery"] = 1.0
            if node_type in ("Hash Join", "Nested Loop", "Merge Join"):
                signals["num_joins"] += 1.0
            for child in node.get("Plans", []):
                walk(child)

        if "Plan" in plan:
            walk(plan["Plan"])

        return list(signals.values())

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self, plan: Dict[str, Any], reward: float, done: bool
    ) -> SQLObservation:
        signals = self._extract_signals(plan)
        legal   = self._compute_legal_actions(self._current_query, plan) if not done else []

        return SQLObservation(**{
            "current_query": self._current_query,
            "observation_vector": signals,
            "legal_actions": legal,
            "explain_plan": plan,
            "done": done,
            "reward": reward,
            "metadata": {
                "episode_id":       self._episode_id,
                "step":             self._step_count,
                "baseline_ms":      self._baseline_time_ms,
                "current_ms":       self._current_time_ms,
                "improvement_pct":  round(
                    (self._baseline_time_ms - self._current_time_ms)
                    / max(self._baseline_time_ms, 1) * 100, 2
                ),
                "rewrites_applied": self._rewrites_applied,
            }
        })

    def _end_episode(self) -> SQLObservation:
        if self._db is None:
            raise RuntimeError("Database connection lost.")

        plan       = self._db.get_explain_plan(self._current_query)
        final_time = plan.get("Execution Time", self._current_time_ms)
        reward     = (self._baseline_time_ms - final_time) / max(self._baseline_time_ms, 1)

        self._current_time_ms = final_time
        self.close()

        return SQLObservation(**{
            "current_query": self._current_query,
            "observation_vector": self._extract_signals(plan),
            "legal_actions": [],
            "explain_plan": plan,
            "done": True,
            "reward": reward,
            "metadata": {
                "episode_id":       self._episode_id,
                "step":             self._step_count,
                "baseline_ms":      self._baseline_time_ms,
                "final_ms":         final_time,
                "improvement_pct":  round(reward * 100, 2),
                "rewrites_applied": self._rewrites_applied,
            }
        })