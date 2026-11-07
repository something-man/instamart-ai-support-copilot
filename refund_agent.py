
from pathlib import Path
import json
import re
import sqlite3
import time
import uuid

import mock_llm
from part_d import PolicyRetriever, load_policy_chunks


BASE_DIR = Path(__file__).resolve().parent
ORDER_DB = BASE_DIR / "data" / "instamart.db"
SCENARIO_DB = BASE_DIR / "database" / "instamart.db"
POLICY_DIR = BASE_DIR / "data" / "policies"

OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

DEFAULT_TOKEN_LIMIT = 6000
DEFAULT_TIME_LIMIT = 30
MAX_TOOL_STEPS = 6


for folder in (OUTPUT_DIR, STATE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# JSONL UTILITIES
# --------------------------------------------------

def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


# --------------------------------------------------
# DATABASE TOOLS
# --------------------------------------------------

def get_order(order_id):
    conn = sqlite3.connect(ORDER_DB)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT * FROM orders WHERE order_id = ?",
        (order_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return {"error": "Order not found", "order_id": order_id}

    return dict(row)


def search_policy(query):
    chunks = load_policy_chunks()
    retriever = PolicyRetriever(chunks)

    results = retriever.retrieve(query, k=3)

    if not results:
        return {
            "found": False,
            "message": "No matching policy information found."
        }

    return {
        "found": True,
        "results": [
            {
                "source": item["source"],
                "score": item["score"],
                "text": item["text"]
            }
            for item in results
        ]
    }


# --------------------------------------------------
# REFUND TOOLS
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery time is unavailable."
        }

    if float(delivery_minutes) <= 45:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery was within 45 minutes."
        }

    return {
        "refund_inr": round(float(order_amount), 2),
        "reason": "Completed order delivered after 45 minutes."
    }



def issue_refund(order_id, refund_inr, run_id):
    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {
            "issued": False,
            "reason": "Refund amount must be positive.",
            "order_id": order_id,
            "refund_inr": amount
        }

    if amount > 1000:
        return {
            "issued": False,
            "reason": "Refund above INR 1000 requires human approval.",
            "order_id": order_id,
            "refund_inr": amount
        }

    existing = read_jsonl(LEDGER_PATH)

    previous = next(
        (
            entry for entry in existing
            if entry.get("order_id") == order_id
            and entry.get("status") == "REFUND"
        ),
        None
    )

    # Idempotency: return the original result without issuing twice.
    if previous:
        return {
            "issued": True,
            "order_id": order_id,
            "refund_inr": previous["refund_inr"],
            "status": "REFUND",
            "already_recorded": True
        }

    record = {
        "run_id": run_id,
        "order_id": order_id,
        "refund_inr": amount,
        "status": "REFUND",
        "approval": "automatic"
    }

    append_jsonl(LEDGER_PATH, record)

    return {
        "issued": True,
        "order_id": order_id,
        "refund_inr": amount,
        "status": "REFUND",
        "already_recorded": False
    }

def escalate_to_human(order_id, refund_inr):
    return {
        "status": "ESCALATE",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "reason": "Human support lead approval required."
    }


# --------------------------------------------------
# TOOL DISPATCH
# --------------------------------------------------

def execute_tool(name, arguments, run_id):

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "get_order":
        return get_order(arguments["order_id"])

    if name == "search_policy":
        return search_policy(arguments["query"])

    if name == "calculate_refund":
        return calculate_refund(
            arguments["order_amount"],
            arguments["delivery_minutes"]
        )

    if name == "issue_refund":
        return issue_refund(
            arguments["order_id"],
            arguments["refund_inr"],
            run_id
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments["order_id"],
            arguments["refund_inr"]
        )

    return {"error": f"Unknown tool: {name}"}


# --------------------------------------------------
# CHECKPOINTING
# --------------------------------------------------

def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    path = checkpoint_path(state["run_id"])

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary.replace(path)


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run {run_id}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------
# AGENT EXECUTION
# --------------------------------------------------

def create_state(scenario, token_limit, time_limit_seconds):

    run_id = str(uuid.uuid4())

    return {
        "run_id": run_id,
        "scenario_id": scenario["scenario_id"],
        "user_message": scenario["message"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT. You are an Instamart refund assistant. "
                    "Use tools to verify orders and apply refund policy. "
                    "Never issue refunds above INR 1000 automatically."
                )
            },
            {
                "role": "user",
                "content": scenario["message"]
            }
        ],
        "tokens_used": 0,
        "tool_steps": 0,
        "tools_executed": [],
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "started_at": time.time(),
        "status": "RUNNING",
        "final_response": None
    }


def finalize(state, status, response):
    state["status"] = status
    state["final_response"] = response

    save_checkpoint(state)

    append_jsonl(RUNS_PATH, {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "tokens_used": state["tokens_used"],
        "tool_steps": state["tool_steps"],
        "tools_executed": state["tools_executed"],
        "final_response": response
    })

    return state


def run_agent(
    scenario,
    token_limit=DEFAULT_TOKEN_LIMIT,
    time_limit_seconds=DEFAULT_TIME_LIMIT,
    crash_after_tools=None
):

    state = create_state(
        scenario,
        token_limit,
        time_limit_seconds
    )

    save_checkpoint(state)

    return continue_agent(
        state,
        crash_after_tools=crash_after_tools
    )


def continue_agent(state, crash_after_tools=None):

    while True:

        if time.time() - state["started_at"] >= state["time_limit_seconds"]:
            return finalize(
                state,
                "FALLBACK_TIMEOUT",
                "Execution time limit reached."
            )

        if state["tokens_used"] >= state["token_limit"]:
            return finalize(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "Token budget exhausted."
            )

        if state["tool_steps"] >= MAX_TOOL_STEPS:
            return finalize(
                state,
                "FALLBACK_MAX_ITERATIONS",
                "Maximum tool execution steps reached."
            )

        response = mock_llm.chat(state["messages"])

        usage = response.usage.total_tokens

        state["tokens_used"] += usage

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return finalize(
                state,
                "COMPLETED",
                message.content or ""
            )

        for tool_call in tool_calls:

            if state["tool_steps"] >= MAX_TOOL_STEPS:
                return finalize(
                    state,
                    "FALLBACK_MAX_ITERATIONS",
                    "Maximum tool execution steps reached."
                )

            if (
                time.time() - state["started_at"]
                >= state["time_limit_seconds"]
            ):
                return finalize(
                    state,
                    "FALLBACK_TIMEOUT",
                    "Execution time limit reached."
                )

            if state["tokens_used"] >= state["token_limit"]:
                return finalize(
                    state,
                    "FALLBACK_TOKEN_LIMIT",
                    "Token budget exhausted."
                )

            tool_name = tool_call.name
            arguments = tool_call.arguments

            result = execute_tool(
                tool_name,
                arguments,
                state["run_id"]
            )

            observation = {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)
            }

            state["messages"].append(observation)

            state["tool_steps"] += 1
            state["tools_executed"].append(tool_name)

            # Save immediately after every tool observation.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and state["tool_steps"] == crash_after_tools
            ):
                print(
                    f"\nSIMULATED CRASH after tool "
                    f"{state['tool_steps']}: {tool_name}"
                )

                return {
                    **state,
                    "status": "CRASH_SIMULATED"
                }


def resume_agent(run_id):
    state = load_checkpoint(run_id)

    if state["status"] != "RUNNING":
        return state

    print("\nRESUMING RUN:", run_id)

    return continue_agent(state)


# --------------------------------------------------
# SCENARIO LOADING
# --------------------------------------------------

def load_scenarios():

    conn = sqlite3.connect(SCENARIO_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM agent_scenarios ORDER BY scenario_id"
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# --------------------------------------------------
# SCENARIO VALIDATION
# --------------------------------------------------



def check_scenario(scenario, result):

    expected = scenario["expected_outcome"]
    expected_amount = round(
        float(scenario["expected_refund_inr"]), 2
    )

    status = result["status"]
    response = result["final_response"] or ""

    if expected == "FALLBACK":
        return status == "FALLBACK_MAX_ITERATIONS"

    if expected == "NOT_FOUND":
        return status == "COMPLETED" and "NOT_FOUND" in response

    if expected == "DENIED":
        return status == "COMPLETED" and "DENIED" in response

    if expected == "ESCALATE":
        for message in result["messages"]:
            if (
                message.get("role") == "tool"
                and message.get("name") == "escalate_to_human"
            ):
                observation = json.loads(message["content"])

                return (
                    status == "COMPLETED"
                    and observation.get("status") == "ESCALATE"
                    and float(observation.get("refund_inr", 0))
                    == expected_amount
                )

        return False

    if expected == "REFUND":
        match = re.search(r"INR\s+([\d.]+)", response)
        actual_amount = float(match.group(1)) if match else None

        return (
            status == "COMPLETED"
            and "REFUND" in response
            and actual_amount == expected_amount
        )

    return False

# --------------------------------------------------
# FULL TEST SUITE
# --------------------------------------------------

def run_agent_suite():

    # Start each test suite with clean output files.
    for path in (LEDGER_PATH, RUNS_PATH):
        if path.exists():
            path.unlink()

    scenarios = load_scenarios()

    print("\n" + "=" * 65)
    print("REFUND AGENT SCENARIO TESTS")
    print("=" * 65)

    results = []

    for scenario in scenarios:

        mock_llm.reset()

        result = run_agent(scenario)

        passed = check_scenario(scenario, result)

        results.append(result)

        print(
            f"{scenario['scenario_id']} | "
            f"Expected: {scenario['expected_outcome']} | "
            f"Actual: {result['status']} | "
            f"Tokens: {result['tokens_used']} | "
            f"Tools: {result['tools_executed']} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # Stop-safely demonstrations.
    print("\nSTOP-SAFELY DEMONSTRATIONS")

    mock_llm.reset()

    token_result = run_agent(
        scenarios[0],
        token_limit=200
    )

    print(
        "Token limit:",
        token_result["status"],
        "| Tokens:",
        token_result["tokens_used"]
    )

    mock_llm.reset()

    timeout_result = run_agent(
        scenarios[0],
        time_limit_seconds=0
    )

    print("Time limit:", timeout_result["status"])

    # Crash after the second tool execution and resume.
    print("\nCRASH AND RESUME DEMONSTRATION")

    mock_llm.reset()

    crash_state = run_agent(
        scenarios[0],
        crash_after_tools=2
    )

    run_id = crash_state["run_id"]

    print("Checkpoint saved:", checkpoint_path(run_id))
    print("Tools before crash:", crash_state["tools_executed"])

    resumed = resume_agent(run_id)

    print("Final status:", resumed["status"])
    print("Final response:", resumed["final_response"])
    print("Tools across crash/resume:", resumed["tools_executed"])

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund"
    ]

    assert resumed["tools_executed"] == expected_tools, (
        "Crash/resume repeated or skipped a tool."
    )

    print("Crash/resume validation: PASS")

    # Final summary.
    passed_count = sum(
        check_scenario(s, r)
        for s, r in zip(scenarios, results)
    )

    print("\nScenario results:", passed_count, "/", len(scenarios))
    print("Refund ledger:", LEDGER_PATH)
    print("Agent runs:", RUNS_PATH)
    print("Checkpoints:", STATE_DIR)


if __name__ == "__main__":
    run_agent_suite()
