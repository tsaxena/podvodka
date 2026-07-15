"""
qwen2vl_tifa_vllm.py

vLLM-batched version of the Qwen2-VL TIFA VQA backend. The transformers-based
version in qwen2vl_tifa.py does one (image, question) forward pass at a time --
fine for offline eval, too slow for a GRPO reward loop where you're scoring
many images x many questions every training step. This version submits the
whole batch to vLLM in a single llm.generate() call and lets vLLM's
continuous batching handle throughput.

Setup
-----
    pip install vllm --break-system-packages
    # Qwen2-VL-7B-Instruct: ~16-18GB VRAM. Use Qwen2-VL-2B-Instruct if tight.

Usage
-----
    from qwen2vl_tifa_vllm import QwenVLVQABatched, compute_tifa_scores_batch

    vqa = QwenVLVQABatched("Qwen/Qwen2-VL-7B-Instruct")

    # items: list of (prompt, image_path, questions) tuples, where
    # `questions` is the list returned by tifascore's
    # get_llama2_question_and_answers() for that prompt.
    scores, details = compute_tifa_scores_batch(items, vqa)
    # scores: List[float], one TIFA score per (prompt, image) item

Integration with the GRPO reward loop
--------------------------------------
Within one training step you typically have G prompts x K sampled images
each (GRPO group sampling). Flatten that into one `items` list and call
compute_tifa_scores_batch() ONCE per step rather than per-sample -- that's
the whole point of the vLLM swap. Question generation (LLaMA-2, from
tifa_reward.py) should still be cached per unique prompt beforehand via
prefetch_questions(), since it's a separate, much cheaper step.
"""

from typing import Dict, List, Tuple

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class QwenVLVQABatched:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    def _build_request(self, image_path: str, question: str, choices: List[str]) -> Dict:
        choice_str = ", ".join(choices)
        text_prompt = (
            f"{question}\n"
            f"Choose exactly one answer from these options: {choice_str}.\n"
            f"Respond with only the chosen option, nothing else."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert("RGB")
        return {"prompt": chat_prompt, "multi_modal_data": {"image": image}}

    def batch_multiple_choice_vqa(
        self, requests: List[Tuple[str, str, List[str]]]
    ) -> List[str]:
        """requests: list of (image_path, question, choices) tuples.
        Returns matched answer string per request, in the same order."""
        vllm_inputs = [
            self._build_request(img, q, choices) for (img, q, choices) in requests
        ]
        outputs = self.llm.generate(vllm_inputs, self.sampling_params)

        answers = []
        for (img, q, choices), out in zip(requests, outputs):
            raw = out.outputs[0].text.strip()
            answers.append(self._match_to_choice(raw, choices))
        return answers

    @staticmethod
    def _match_to_choice(raw_answer: str, choices: List[str]) -> str:
        raw_norm = raw_answer.strip().lower().rstrip(".")
        for c in choices:
            if raw_norm == c.strip().lower():
                return c
        for c in choices:
            if c.strip().lower() in raw_norm or raw_norm in c.strip().lower():
                return c
        return raw_answer


def compute_tifa_scores_batch(
    items: List[Tuple[str, str, List[Dict]]], vqa: QwenVLVQABatched
) -> Tuple[List[float], List[List[Dict]]]:
    """items: list of (prompt, image_path, questions) tuples, where
    `questions` is a list of {'question', 'choices', 'answer'} dicts
    (from tifascore's get_llama2_question_and_answers).

    Returns (scores, details) where scores[i] is the TIFA score for
    items[i], and details[i] is the per-question breakdown.
    """
    # Flatten all (image, question) pairs across all items into one
    # vLLM batch call, tracking which item each request belongs to.
    flat_requests: List[Tuple[str, str, List[str]]] = []
    owner_idx: List[int] = []
    for i, (prompt, image_path, questions) in enumerate(items):
        for q in questions:
            flat_requests.append((image_path, q["question"], q["choices"]))
            owner_idx.append(i)

    if not flat_requests:
        return [0.0] * len(items), [[] for _ in items]

    answers = vqa.batch_multiple_choice_vqa(flat_requests)

    # Regroup answers back by item and compute per-item TIFA score.
    per_item_results: List[List[Dict]] = [[] for _ in items]
    flat_i = 0
    for i, (prompt, image_path, questions) in enumerate(items):
        for q in questions:
            model_answer = answers[flat_i]
            flat_i += 1
            is_correct = model_answer.strip().lower() == q["answer"].strip().lower()
            per_item_results[i].append({**q, "model_answer": model_answer, "correct": is_correct})

    scores = []
    for i, (prompt, image_path, questions) in enumerate(items):
        if not questions:
            scores.append(0.0)
        else:
            correct = sum(d["correct"] for d in per_item_results[i])
            scores.append(correct / len(questions))

    return scores, per_item_results


if __name__ == "__main__":
    vqa = QwenVLVQABatched("Qwen/Qwen2-VL-7B-Instruct")
    items = [
        (
            "a black colored banana",
            "sample/drawbench_8.jpg",
            [
                {"question": "is this a banana?", "choices": ["yes", "no"], "answer": "yes"},
                {"question": "what color is the banana?", "choices": ["black", "yellow", "red", "green"], "answer": "black"},
            ],
        ),
    ]
    scores, details = compute_tifa_scores_batch(items, vqa)
    print("scores:", scores)
    for d in details:
        print(d)