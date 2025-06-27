from collections import deque

from utils import TreeOfThought, call_gpt


def main():
    prompt = """\
Given the following observations, please provide the detailed hypothesis about the cause of the observed events.
You can ask questions to clarify the observations.
Your questions must be specific and must be answered with "yes", "no" or "irrelevant".
Observations:
1. A driver was driving a car at high speed.
2. The driver decelerated the car.
"""
    context = [prompt]
    res = call_gpt(context, model="gpt-4o-mini", temperature=0.5, format=TreeOfThought)
    thought_queue = deque()
    current_depth = 0
    thought_queue.append((res.possible_thoughts, current_depth))
    while thought_queue:
        thoughts, current_depth = thought_queue.popleft()
        current_hypotheses, scores, next_questions = [], [], []
        for thought in thoughts:
            if thought.score > 0.5:
                current_hypotheses.append(thought.hypothesis)
                scores.append(thought.score)
                next_questions.append(thought.next_question)
        for hyp, score, next_question in zip(current_hypotheses, scores, next_questions):
            print(f"Current Hypothesis: {hyp}, Score: {score}, Next Question: {next_question}")
            if current_depth < 3 and next_question is not None:
                context.append(next_question)
                user_input = input(f"Next question ({current_depth + 1}/3): {next_question}\nYour answer: ")
                context.append(user_input)
            elif res.is_terminal:
                user_input = input(f"Final Hypothesis: {hyp}\nAccept? (yes/no): ")
                if user_input.lower() == "yes":
                    print(f"Final Hypothesis: {hyp}, Score: {score}")
                    return
                else:
                    context.append(f"Answer: {hyp}")
                    context.append(user_input)
                    continue
            elif current_depth == 3:
                print(f"Final Hypothesis: {hyp}, Score: {score}")
                continue
        if current_depth < 3:
            res = call_gpt(context, model="gpt-4o-mini", temperature=0.5, format=TreeOfThought)
            thought_queue.append((res.possible_thoughts, current_depth + 1))
            if current_depth + 1 == 3:
                print("====== Thought Process Reached Maximum Depth ======")
    print("====== End of Thought Process ======")


if __name__ == "__main__":
    main()
