from collections import deque

from utils import Thought, call_gpt


def main():
    prompt = """\
Given the following observations, please provide the hypothesis about the cause of the observed events.
You can ask close-ended questions to clarify the observations.
Observations:
1. A driver was driving a car at high speed.
2. The driver decelerated the car.
"""
    context = [prompt]
    res = call_gpt(context, model="gpt-4o-mini", temperature=0.5, format=Thought)
    hypothesis_queue = deque()
    depth = 0
    hypothesis_queue += [
        (hyp, score, depth) for hyp, score in zip(res.possible_hypotheses, res.hypothesis_potential) if score > 0.5
    ]
    while hypothesis_queue:
        current_hypothesis, score, current_depth = hypothesis_queue.popleft()
        print(f"Depth: {current_depth}, Hypothesis: {current_hypothesis}, Score: {score}")
        if current_depth < 3 and res.next_question is not None:
            context.append(res.next_question)
            user_input = input(f"Next question: {res.next_question}\nYour answer: ")
            context.append(user_input)
            res = call_gpt(context, model="gpt-4o-mini", temperature=0.5, format=Thought)


if __name__ == "__main__":
    main()
