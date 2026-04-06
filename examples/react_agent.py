"""Complex demo: ReAct Agent with multi-turn reasoning + tool calling.

Simulates a real-world scenario where an agent:
1. Receives a complex question
2. Thinks step-by-step (Chain of Thought)
3. Decides which tool to call
4. Executes the tool
5. Observes the result
6. Repeats until it has enough info to answer

Usage:
    export GLM_API_KEY=your-key-here
    python examples/react_agent.py
    agentlife ui
"""

import json
import os
import math
import agentlife
from openai import OpenAI

agentlife.init()

client = OpenAI(
    api_key=os.environ["GLM_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

# ── Tool implementations ──

TOOLS_SCHEMA = [
    {"name": "search", "description": "Search for information about a topic", "parameters": {"query": "string"}},
    {"name": "calculator", "description": "Calculate a math expression", "parameters": {"expression": "string"}},
    {"name": "database", "description": "Query product database", "parameters": {"product_name": "string"}},
    {"name": "final_answer", "description": "Give the final answer to the user", "parameters": {"answer": "string"}},
]

PRODUCT_DB = {
    "iPhone 16 Pro": {"price": 8999, "stock": 152, "rating": 4.8, "monthly_sales": 12000},
    "Huawei Mate 70": {"price": 6999, "stock": 89, "rating": 4.9, "monthly_sales": 15000},
    "Xiaomi 15 Ultra": {"price": 5999, "stock": 230, "rating": 4.7, "monthly_sales": 8000},
}

SEARCH_RESULTS = {
    "手机市场趋势": "2026年中国智能手机市场回暖，AI手机渗透率达45%，高端市场竞争加剧。华为、苹果、小米三家占据高端市场80%份额。",
    "用户购买因素": "调研显示消费者购买手机最关注：1.性能(35%) 2.拍照(25%) 3.价格(20%) 4.品牌(15%) 5.续航(5%)",
    "AI手机功能": "主流AI手机功能包括：智能摘要、实时翻译、AI修图、智能助手。华为和苹果在AI功能上领先。",
}


@agentlife.trace(name="tool:search")
def tool_search(query: str) -> str:
    for key, value in SEARCH_RESULTS.items():
        if any(k in query for k in key):
            return value
    return f"未找到关于'{query}'的相关信息。"


@agentlife.trace(name="tool:calculator")
def tool_calculator(expression: str) -> str:
    try:
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@agentlife.trace(name="tool:database")
def tool_database(product_name: str) -> str:
    for name, info in PRODUCT_DB.items():
        if product_name in name or name in product_name:
            return json.dumps(info, ensure_ascii=False)
    return f"未找到产品: {product_name}"


TOOL_MAP = {
    "search": lambda args: tool_search(args.get("query", "")),
    "calculator": lambda args: tool_calculator(args.get("expression", "")),
    "database": lambda args: tool_database(args.get("product_name", "")),
}

# ── ReAct Agent ──

SYSTEM_PROMPT = """你是一个智能分析助手，使用 ReAct（Reasoning + Acting）方式解决问题。

每一步你需要输出：
1. Thought: 你的推理过程
2. Action: 要调用的工具（JSON格式）

可用工具：
- search(query): 搜索信息
- calculator(expression): 数学计算
- database(product_name): 查询产品数据库（支持：iPhone 16 Pro, Huawei Mate 70, Xiaomi 15 Ultra）
- final_answer(answer): 给出最终答案（当你收集够信息时使用）

输出格式（严格JSON）：
{"thought": "你的推理", "action": "工具名", "args": {"参数名": "参数值"}}

注意：每次只调用一个工具。收集足够信息后用 final_answer 结束。"""


@agentlife.trace(name="llm:reason")
def reason(messages: list[dict]) -> dict:
    """One round of LLM reasoning — returns thought + action."""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        max_tokens=500,
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass

    return {"thought": content, "action": "final_answer", "args": {"answer": content}}


@agentlife.trace(name="execute_tool")
def execute_tool(action: str, args: dict) -> str:
    """Execute a tool and return the observation."""
    if action in TOOL_MAP:
        return TOOL_MAP[action](args)
    return f"未知工具: {action}"


@agentlife.trace(name="react_loop")
def react_agent(question: str, max_turns: int = 6) -> str:
    """Run the full ReAct loop: think → act → observe → repeat."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        print(f"\n  --- Turn {turn + 1} ---")

        result = reason(messages)
        thought = result.get("thought", "")
        action = result.get("action", "final_answer")
        args = result.get("args", {})

        print(f"  Thought: {thought[:80]}...")
        print(f"  Action:  {action}({args})")

        if action == "final_answer":
            return args.get("answer", thought)

        observation = execute_tool(action, args)
        print(f"  Observe: {observation[:80]}...")

        messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation}\n\n请继续分析，或者如果信息足够，使用 final_answer 给出最终答案。"})

    return "达到最大轮次，无法得出结论。"


def main():
    question = "帮我分析一下目前高端手机市场的情况，对比iPhone 16 Pro、Huawei Mate 70和Xiaomi 15 Ultra这三款手机，从价格、销量、评分等维度给出购买建议。"

    print("=" * 70)
    print("AgentLife 复杂场景测试 — ReAct Agent (多轮推理 + 工具调用)")
    print("=" * 70)
    print(f"\n  问题: {question}")

    with agentlife.session("ReAct-手机分析Agent"):
        answer = react_agent(question)

    print(f"\n{'=' * 70}")
    print(f"  最终答案:\n")
    print(f"  {answer}")
    print(f"\n{'=' * 70}")
    print("  运行 agentlife ui 查看完整多轮 trace")
    print("=" * 70)


if __name__ == "__main__":
    main()
