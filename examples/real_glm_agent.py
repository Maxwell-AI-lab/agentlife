"""Real business test: multi-step agent using GLM-4 (Zhipu AI).

Usage:
    export GLM_API_KEY=your-key-here
    python examples/real_glm_agent.py
    agentlife ui
"""

from openai import OpenAI
import agentlife

agentlife.init()

import os

client = OpenAI(
    api_key=os.environ["GLM_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
)


@agentlife.trace
def analyze_requirement(user_input: str) -> str:
    """Step 1: Analyze user requirement and extract key info."""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是一个需求分析助手。分析用户需求，提取关键信息点，用编号列出（最多3点）。简洁回答。"},
            {"role": "user", "content": user_input},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


@agentlife.trace
def generate_solution(requirement_analysis: str) -> str:
    """Step 2: Generate technical solution based on analysis."""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是一个技术方案架构师。根据需求分析，给出简洁的技术方案（包含技术选型和核心步骤）。控制在200字以内。"},
            {"role": "user", "content": f"需求分析如下：\n{requirement_analysis}\n\n请给出技术方案。"},
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


@agentlife.trace
def estimate_effort(solution: str) -> str:
    """Step 3: Estimate development effort."""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是一个项目经理。根据技术方案，估算开发工时和难度。格式：难度（高/中/低）、预计工时、主要风险点。简洁回答。"},
            {"role": "user", "content": f"技术方案如下：\n{solution}\n\n请估算开发工作量。"},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


@agentlife.trace
def generate_summary(requirement: str, analysis: str, solution: str, effort: str) -> str:
    """Step 4: Generate final summary report."""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是一个项目总结助手。将需求分析、技术方案、工作量评估整合成一份简洁的项目摘要报告。控制在300字以内。"},
            {"role": "user", "content": f"原始需求：{requirement}\n\n需求分析：{analysis}\n\n技术方案：{solution}\n\n工作量评估：{effort}"},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""


def main():
    user_requirement = "我需要开发一个智能客服系统，能够自动回答用户关于产品的常见问题，支持多轮对话，并且能在无法回答时转接人工客服。"

    print("=" * 60)
    print("AgentLife 真实业务测试 — 智能客服需求分析 Agent")
    print("=" * 60)

    with agentlife.session("智能客服需求分析"):
        print("\n[Step 1] 分析需求...")
        analysis = analyze_requirement(user_requirement)
        print(f"  结果：{analysis[:100]}...\n")

        print("[Step 2] 生成技术方案...")
        solution = generate_solution(analysis)
        print(f"  结果：{solution[:100]}...\n")

        print("[Step 3] 评估工作量...")
        effort = estimate_effort(solution)
        print(f"  结果：{effort[:100]}...\n")

        print("[Step 4] 生成总结报告...")
        summary = generate_summary(user_requirement, analysis, solution, effort)
        print(f"  结果：{summary[:200]}...\n")

    print("=" * 60)
    print("完成！运行 agentlife ui 查看完整 trace")
    print("=" * 60)


if __name__ == "__main__":
    main()
