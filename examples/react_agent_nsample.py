"""N-Sample Rollout Demo: 4 parallel rollouts of a ReAct Agent.

Simulates an RL training scenario where the same prompt is sampled N times
with different temperatures to get diverse trajectories. AgentLife's group()
feature lets you compare all rollouts side-by-side.

Usage:
    export GLM_API_KEY=your-key-here
    python examples/react_agent_nsample.py
    agentlife ui   # then click "Groups" tab
"""

import json
import os
import math
import random
import agentlife
from openai import OpenAI

agentlife.init()

client = OpenAI(
    api_key=os.environ["GLM_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

N_SAMPLES = 4
TEMPERATURES = [0.1, 0.4, 0.7, 0.95]

# ── Tools (same as react_agent.py) ──

PRODUCT_DB = {
    "iPhone 16 Pro": {"price": 8999, "stock": 152, "rating": 4.8, "monthly_sales": 12000},
    "Huawei Mate 70": {"price": 6999, "stock": 89, "rating": 4.9, "monthly_sales": 15000},
    "Xiaomi 15 Ultra": {"price": 5999, "stock": 230, "rating": 4.7, "monthly_sales": 8000},
    "Samsung S25 Ultra": {"price": 9499, "stock": 67, "rating": 4.6, "monthly_sales": 5000},
}

SEARCH_RESULTS = {
    "手机市场": "2026年中国智能手机市场回暖，AI手机渗透率达45%，高端市场竞争加剧。华为、苹果、小米、三星四家占据高端市场85%份额。",
    "用户": "调研显示消费者购买手机最关注：1.性能(35%) 2.拍照(25%) 3.价格(20%) 4.品牌(15%) 5.续航(5%)",
    "AI功能": "主流AI手机功能包括：智能摘要、实时翻译、AI修图、智能助手。华为和苹果在AI功能上领先。三星Galaxy AI主打跨设备协同。",
    "性价比": "2026年高端手机性价比排行：1.Xiaomi 15 Ultra(综合评分92) 2.Huawei Mate 70(90) 3.iPhone 16 Pro(88) 4.Samsung S25 Ultra(85)",
}


@agentlife.trace(name="tool:search")
def tool_search(query: str) -> str:
    for key, value in SEARCH_RESULTS.items():
        if any(k in query for k in key):
            return value
    return f"未找到关于'{query}'的相关信息。可以尝试搜索：手机市场、用户、AI功能、性价比"


@agentlife.trace(name="tool:calculator")
def tool_calculator(expression: str) -> str:
    try:
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
        }
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算错误: {e}"


@agentlife.trace(name="tool:database")
def tool_database(product_name: str) -> str:
    for name, info in PRODUCT_DB.items():
        if product_name in name or name in product_name:
            return json.dumps(info, ensure_ascii=False)
    available = ", ".join(PRODUCT_DB.keys())
    return f"未找到产品: {product_name}。可查询的产品: {available}"


TOOL_MAP = {
    "search": lambda args: tool_search(args.get("query", "")),
    "calculator": lambda args: tool_calculator(args.get("expression", "")),
    "database": lambda args: tool_database(args.get("product_name", "")),
}

SYSTEM_PROMPT = """你是一个智能分析助手，使用 ReAct（Reasoning + Acting）方式解决问题。

每一步你需要输出：
1. Thought: 你的推理过程
2. Action: 要调用的工具（JSON格式）

可用工具：
- search(query): 搜索信息（关键词：手机市场、用户、AI功能、性价比）
- calculator(expression): 数学计算（Python表达式）
- database(product_name): 查询产品数据库（iPhone 16 Pro, Huawei Mate 70, Xiaomi 15 Ultra, Samsung S25 Ultra）
- final_answer(answer): 给出最终答案（当你收集够信息时使用）

严格输出JSON格式：
{"thought": "你的推理", "action": "工具名", "args": {"参数名": "参数值"}}

每次只调用一个工具。收集足够信息后用 final_answer 结束。"""


@agentlife.trace(name="llm:reason")
def reason(messages: list[dict], temperature: float) -> dict:
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        max_tokens=600,
        temperature=temperature,
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
    if action in TOOL_MAP:
        return TOOL_MAP[action](args)
    return f"未知工具: {action}"


@agentlife.trace(name="react_loop")
def react_agent(question: str, temperature: float, max_turns: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        result = reason(messages, temperature)
        thought = result.get("thought", "")
        action = result.get("action", "final_answer")
        args = result.get("args", {})

        print(f"      Turn {turn + 1}: {action}({list(args.values())[:1]})")

        if action == "final_answer":
            return args.get("answer", thought)

        observation = execute_tool(action, args)

        messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation}\n\n请继续分析，或者如果信息足够，使用 final_answer 给出最终答案。"})

    return "达到最大轮次，无法得出结论。"


def main():
    question = "对比 iPhone 16 Pro、Huawei Mate 70、Xiaomi 15 Ultra、Samsung S25 Ultra 四款手机，综合价格、销量、评分和AI功能，给出最佳购买推荐和理由。"

    print("=" * 70)
    print(f"  AgentLife N-Sample Rollout Demo")
    print(f"  Question: {question[:60]}...")
    print(f"  N_SAMPLES = {N_SAMPLES}, temperatures = {TEMPERATURES}")
    print("=" * 70)

    with agentlife.group("nsample-手机推荐"):
        for i in range(N_SAMPLES):
            temp = TEMPERATURES[i]
            print(f"\n  ── Sample {i} (temperature={temp}) ──")

            with agentlife.session(f"sample-{i} (T={temp})", sample_index=i):
                try:
                    answer = react_agent(question, temperature=temp)
                    print(f"    Result: {answer[:80]}...")
                except Exception as e:
                    print(f"    ERROR: {e}")

    print(f"\n{'=' * 70}")
    print("  Done! Run `agentlife ui` and click the 'Groups' tab")
    print("  to see the N-sample comparison view.")
    print("=" * 70)


if __name__ == "__main__":
    main()
