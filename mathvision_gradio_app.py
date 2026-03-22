#!/usr/bin/env python3
"""
MathVision Demo App

A lightweight Gradio app for solving math problems with text + image/webcam input.
Supports switching between base and fine-tuned Qwen3-VL MathVision variants.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import random
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from threading import Thread

import gradio as gr
import markdown as md
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

try:
    from transformers import AutoModelForImageTextToText as AutoVisionGenerateModel
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoVisionGenerateModel
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoVisionGenerateModel


REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE_DATA_PATH = REPO_ROOT / "data" / "mathvision_llamafactory" / "val_split.json"
MEDIA_ROOT = REPO_ROOT / "data" / "mathvision_llamafactory"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_name_or_path: str
    adapter_name_or_path: str | None
    template: str
    enable_thinking: bool
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "ft_base": ModelSpec(
        key="ft_base",
        label="Khong thinking (chi dap an)",
        model_name_or_path="Qwen/Qwen3-VL-4B-Instruct",
        adapter_name_or_path=str(REPO_ROOT / "saves" / "qwen3-vl-4b" / "lora" / "mathvision"),
        template="qwen3_vl_nothink",
        enable_thinking=False,
        description="Mo hinh chi tra ve dap an cuoi, khong hien thi chuoi thinking.",
    ),
    "hf_thinking_new": ModelSpec(
        key="hf_thinking_new",
        label="Thinking (co loi giai)",
        model_name_or_path="Qwen/Qwen3-VL-4B-Instruct",
        adapter_name_or_path="anhnq1130/vmmu-thinking-regenerated-qwen3vl",
        template="qwen3_vl",
        enable_thinking=True,
        description="Mo hinh thinking VMMU regenerated tren Hugging Face.",
    ),
}


def _clean_question(text: str) -> str:
    return text.replace("<image>", "").strip()


def _safe_exists(path_str: str | None) -> bool:
    if not path_str:
        return False
    # Treat non-absolute paths that look like "<namespace>/<repo>" as HF repos.
    if not path_str.startswith(("/", "./", "../")) and "/" in path_str:
        return True
    return Path(path_str).exists()


def _list_example_items(limit: int = 5) -> list[dict[str, str]]:
    data = json.loads(EXAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    for item in data:
        images = item.get("images", [])
        if len(images) != 1:
            continue

        user_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "user"), "").strip()
        assistant_msg = next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), "").strip()
        question = _clean_question(user_msg)

        if not question or len(question) > 160:
            continue

        image_path = MEDIA_ROOT / images[0]
        if not image_path.exists():
            continue

        examples.append(
            {
                "title": question if len(question) <= 70 else question[:67] + "...",
                "question": question,
                "answer": assistant_msg,
                "image_path": str(image_path),
            }
        )
        if len(examples) >= limit:
            break

    return examples


EXAMPLE_ITEMS = _list_example_items()


class MathVisionRuntime:
    def __init__(self) -> None:
        self.current_key: str | None = None
        self.model: Any | None = None
        self.processor: Any | None = None
        self.lock = threading.Lock()

    def unload_model(self) -> None:
        old_model = self.model
        old_processor = self.processor
        self.model = None
        self.processor = None
        self.current_key = None
        if old_model is not None:
            del old_model
        if old_processor is not None:
            del old_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def ensure_model(self, model_key: str) -> tuple[ModelSpec, str]:
        spec = MODEL_SPECS[model_key]
        if spec.adapter_name_or_path and not _safe_exists(spec.adapter_name_or_path):
            raise FileNotFoundError(f"Khong tim thay adapter: {spec.adapter_name_or_path}")

        if self.current_key == model_key and self.model is not None and self.processor is not None:
            return spec, f"Da san sang: {spec.label}"

        # Always unload old model before switching to another one.
        self.unload_model()

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        base_model = AutoVisionGenerateModel.from_pretrained(spec.model_name_or_path, **load_kwargs)
        processor = None
        processor_note = ""

        # Prefer base processor to avoid tokenizer config incompatibility
        # from adapters trained/saved with newer transformers versions.
        try:
            processor = AutoProcessor.from_pretrained(spec.model_name_or_path, trust_remote_code=True)
        except Exception:
            processor = None

        if spec.adapter_name_or_path:
            model = PeftModel.from_pretrained(base_model, spec.adapter_name_or_path)
            if processor is None:
                processor = AutoProcessor.from_pretrained(spec.adapter_name_or_path, trust_remote_code=True)
                processor_note = " (processor from adapter)"
        else:
            model = base_model
            if processor is None:
                processor = AutoProcessor.from_pretrained(spec.model_name_or_path, trust_remote_code=True)

        self.model = model.eval()
        self.processor = processor
        self.current_key = model_key
        return spec, f"Da load: {spec.label}{processor_note}"

    def generate(self, model_key: str, question: str, image: Image.Image | None) -> dict[str, str]:
        with self.lock:
            spec, status = self.ensure_model(model_key)
            response = self._run_generation(spec, question, image)
            thinking, final_answer = split_reasoning_response(response, spec.enable_thinking)
            return {
                "status": status,
                "response": response,
                "thinking": thinking,
                "final_answer": final_answer,
            }

    def stream_generate(self, model_key: str, question: str, image: Image.Image | None):
        with self.lock:
            spec, status = self.ensure_model(model_key)
            raw_response = ""

            for chunk in self._stream_generation(spec, question, image):
                raw_response += chunk
                thinking, final_answer = split_reasoning_stream_state(
                    text=raw_response,
                    enable_thinking=spec.enable_thinking,
                    completed=False,
                )
                yield {
                    "status": status,
                    "response": raw_response,
                    "thinking": thinking,
                    "final_answer": final_answer,
                }

            # Some adapter/model combos may produce empty stream chunks.
            # Fallback to regular generation so UI still gets content.
            if not raw_response.strip():
                raw_response = self._run_generation(spec, question, image)

            thinking, final_answer = split_reasoning_stream_state(
                text=raw_response,
                enable_thinking=spec.enable_thinking,
                completed=True,
            )
            yield {
                "status": status,
                "response": raw_response,
                "thinking": thinking,
                "final_answer": final_answer,
            }

    def _prepare_inputs(self, spec: ModelSpec, question: str, image: Image.Image | None) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model chua duoc load.")

        prompt = build_math_prompt(question=question, enable_thinking=spec.enable_thinking)
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if image is not None:
            proc_kwargs["images"] = [image.convert("RGB")]

        inputs = self.processor(**proc_kwargs)
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs

    def _stream_generation(self, spec: ModelSpec, question: str, image: Image.Image | None):
        if self.model is None or self.processor is None:
            raise RuntimeError("Model chua duoc load.")

        inputs = self._prepare_inputs(spec, question, image)
        if not hasattr(self.processor, "tokenizer"):
            raise RuntimeError("Processor khong co tokenizer de stream output.")

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": 1024 if spec.enable_thinking else 256,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "streamer": streamer,
        }
        worker = Thread(target=self.model.generate, kwargs=generate_kwargs, daemon=True)
        worker.start()
        for new_text in streamer:
            yield new_text
        worker.join()

    def _run_generation(self, spec: ModelSpec, question: str, image: Image.Image | None) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model chua duoc load.")

        inputs = self._prepare_inputs(spec, question, image)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024 if spec.enable_thinking else 256,
                do_sample=False,
                repetition_penalty=1.05,
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[:, input_length:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response


RUNTIME = MathVisionRuntime()


def build_math_prompt(question: str, enable_thinking: bool) -> str:
    question = question.strip()
    if enable_thinking:
        return (
            f"{question}\n\n"
            "Hay giai bai toan nay. Neu can, dat phan suy luan trong the <think>...</think> "
            "va ket thuc bang mot dong dap an ngan gon."
        )

    return f"{question}\n\nHay tra loi gon va dua ra dap an cuoi cung ro rang."


def split_reasoning_response(text: str, enable_thinking: bool) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""

    if "<think>" in text and "</think>" in text:
        match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            final_answer = extract_final_line(match.group(2).strip())
            return thinking, final_answer

    if "<think>" in text:
        thinking = text.split("<think>", 1)[1].strip()
        return thinking, extract_final_line(thinking)

    if enable_thinking:
        return text, extract_final_line(text)

    final_answer = extract_final_line(text) or text
    return "(Model dang chay o che do khong-thinking)", final_answer


def split_reasoning_stream_state(text: str, enable_thinking: bool, completed: bool) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""

    if "<think>" in text:
        after_open = text.split("<think>", 1)[1]
        if "</think>" in after_open:
            thinking_raw, after_close = after_open.split("</think>", 1)
            thinking = thinking_raw.strip()
            final_answer = extract_final_line(after_close.strip()) if after_close.strip() else ""
            if completed and not final_answer:
                final_answer = extract_final_line(thinking)
            return thinking, final_answer

        thinking = after_open.strip()
        final_answer = extract_final_line(thinking) if completed else ""
        return thinking, final_answer

    if enable_thinking:
        thinking = text
        final_answer = extract_final_line(text) if completed else ""
        return thinking, final_answer

    thinking = "(Model dang chay o che do khong-thinking)"
    # For non-thinking models, show answer progressively instead of waiting for stream end.
    final_answer = extract_final_line(text) or text.strip()
    if completed and not final_answer:
        final_answer = text.strip()
    return thinking, final_answer


def extract_final_line(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    last_line = lines[-1].rstrip(".")
    frac_matches = re.findall(r"\\frac\{[^}]+\}\{[^}]+\}", last_line)
    if frac_matches:
        return frac_matches[-1]

    symbolic_matches = re.findall(r"-?\d+(?:\.\d+)?(?:\\sqrt\{[^}]+\})+", last_line)
    if symbolic_matches:
        return symbolic_matches[-1]

    numeric_matches = re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", last_line)
    if numeric_matches:
        return numeric_matches[-1]

    choice_matches = re.findall(r"\b([A-E])\b", last_line, flags=re.IGNORECASE)
    if choice_matches:
        return choice_matches[-1].upper()

    cleaned = re.sub(r"^(?:dap an|answer)(?: cuoi cung)?\s*[:\-]\s*", "", last_line, flags=re.IGNORECASE)
    return cleaned.strip() or last_line


def get_model_choices() -> list[str]:
    return [spec.label for spec in MODEL_SPECS.values()]


def get_model_key_by_label(label: str) -> str:
    for key, spec in MODEL_SPECS.items():
        if spec.label == label:
            return key
    raise KeyError(f"Unknown model label: {label}")


def render_model_status(selected_label: str) -> str:
    spec = MODEL_SPECS[get_model_key_by_label(selected_label)]
    badge = "Co loi giai (thinking)" if spec.enable_thinking else "Chi dap an (khong thinking)"
    adapter_status = "Co adapter LoRA" if _safe_exists(spec.adapter_name_or_path) else "Khong dung adapter"
    return (
        f"### {spec.label}\n"
        f"- Backend: Qwen3-VL-4B\n"
        f"- Template: `{spec.template}`\n"
        f"- Che do: `{badge}`\n"
        f"- Trang thai: `{adapter_status}`\n\n"
        f"{spec.description}"
    )


def load_example(index: int) -> tuple[str, str, str]:
    item = EXAMPLE_ITEMS[index]
    note = f"Dap an mau: `{item['answer']}`"
    return item["image_path"], item["question"], note


def load_random_example() -> tuple[str, str, str]:
    item = random.choice(EXAMPLE_ITEMS)
    note = f"Dap an mau: `{item['answer']}`"
    return item["image_path"], item["question"], note


def render_math_html(text: str) -> str:
    text = (text or "").strip()
    if not text:
        text = "_Khong co noi dung_"

    element_id = f"math-block-{uuid.uuid4().hex}"
    content = md.markdown(text, extensions=["fenced_code", "nl2br", "tables"])
    return f"""
    <div id="{element_id}" class="math-render">{content}</div>
    <script>
    (() => {{
      const node = document.getElementById("{element_id}");
      if (node && window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise([node]);
      }}
    }})();
    </script>
    """


def solve_math_problem_stream(model_label: str, question: str, image: Image.Image | None):
    question = (question or "").strip()
    if not question and image is None:
        raise gr.Error("Hay nhap cau hoi hoac chon mot hinh anh de bat dau.")

    model_key = get_model_key_by_label(model_label)
    spec = MODEL_SPECS[model_key]
    yield (
        "Dang tai model / suy luan...",
        render_math_html("_Model khong-thinking: se khong hien thi chuoi suy luan._")
        if not spec.enable_thinking
        else render_math_html("_Dang suy luan..._"),
        render_math_html("_Dang doi dap an cuoi..._"),
    )

    try:
        for result in RUNTIME.stream_generate(
            model_key=model_key,
            question=question or "Hay giai bai toan trong hinh.",
            image=image,
        ):
            thinking_text = result["thinking"]
            final_text = result["final_answer"]

            if not spec.enable_thinking:
                thinking_text = "Model nay chi tra dap an cuoi, khong hien thi thinking."
                if not final_text.strip():
                    final_text = (result.get("response") or "").strip()

            yield (
                result["status"],
                render_math_html(thinking_text),
                render_math_html(final_text),
            )
    except Exception as exc:
        raise gr.Error(f"Khong the sinh ket qua: {exc}") from exc


def clear_ui() -> tuple[None, str, str, str, str, str]:
    return None, "", "", "", "", ""


def update_question_preview(question: str) -> str:
    return render_math_html(question)


def build_image_input() -> gr.Image:
    kwargs = {
        "type": "pil",
        "label": "Anh bai toan",
        "height": 320,
    }
    try:
        return gr.Image(sources=["upload", "webcam"], **kwargs)
    except TypeError:
        return gr.Image(**kwargs)


def build_examples_html() -> str:
    cards = []
    for idx, item in enumerate(EXAMPLE_ITEMS, start=1):
        cards.append(
            f"<div class='sample-card'><strong>Mau {idx}</strong><br>{html.escape(item['title'])}<br>"
            f"<span class='sample-answer'>Dap an: {html.escape(item['answer'])}</span></div>"
        )
    return "".join(cards)


def build_demo() -> gr.Blocks:
    css = """
    .app-shell {max-width: 1240px; margin: 0 auto;}
    .hero {padding: 22px 24px; border-radius: 18px; background: linear-gradient(135deg, #13315c 0%, #0b6e4f 100%); color: #fff;}
    .sample-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;}
    .sample-card {padding: 10px 12px; border-radius: 12px; background: #f3f7fa; border: 1px solid #dbe5ec; min-height: 84px;}
    .sample-answer {display: inline-block; margin-top: 6px; color: #184e77; font-size: 12px;}
    .live-thinking {min-height: 320px; max-height: 520px; overflow-y: auto; border: 1px solid #dbe5ec; border-radius: 12px; padding: 14px;}
    .final-card {border: 2px solid #0b6e4f; border-radius: 12px; padding: 12px; background: #f4fffa;}
    .final-card .math-render,
    .final-card .math-render *,
    .final-card .math-render p,
    .final-card .math-render span,
    .final-card .math-render strong,
    .final-card .math-render code {
      color: #052e16 !important;
      opacity: 1 !important;
      font-weight: 700;
    }
    .live-thinking .math-render {color: #e5e7eb;}
    """

    with gr.Blocks(title="MathVision Demo", theme=gr.themes.Soft(), css=css) as demo:
        gr.HTML(
            """
            <div class="app-shell">
              <div class="hero">
                <h1>MathVision Demo</h1>
                <p>Nhap cau hoi, chup anh bai toan bang camera, va xem loi giai thinking cung dap an cuoi.</p>
              </div>
            </div>
            <script>
              window.MathJax = {
                tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] },
                svg: { fontCache: 'global' }
              };
            </script>
            <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=get_model_choices(),
                    value=MODEL_SPECS["ft_base"].label,
                    label="Chon model",
                )
                model_info = gr.Markdown(render_model_status(MODEL_SPECS["ft_base"].label))
                image_input = build_image_input()
                question_input = gr.Textbox(
                    label="Cau hoi",
                    lines=5,
                    placeholder="Vi du: Hay tinh dien tich hinh tam giac trong anh.",
                )
                question_preview = gr.HTML(render_math_html("_Cau hoi xem truoc se hien thi o day_"))
                with gr.Row():
                    solve_btn = gr.Button("Giai bai toan", variant="primary")
                    random_btn = gr.Button("Random")
                    clear_btn = gr.Button("Xoa")
                status_box = gr.Textbox(label="Trang thai model", interactive=False)

            with gr.Column(scale=1):
                thinking_box = gr.HTML(label="Thinking (Live)", elem_classes=["live-thinking"])
                final_answer_box = gr.HTML(label="Dap an cuoi", elem_classes=["final-card"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Bai mau tu MathVision val")
                gr.HTML(f"<div class='sample-grid'>{build_examples_html()}</div>")
                example_answer_note = gr.Markdown("")
                with gr.Row():
                    example_buttons = []
                    for idx, _ in enumerate(EXAMPLE_ITEMS, start=1):
                        example_buttons.append(gr.Button(f"Mau {idx}", size="sm"))

        model_dropdown.change(fn=render_model_status, inputs=[model_dropdown], outputs=[model_info])
        question_input.change(fn=update_question_preview, inputs=[question_input], outputs=[question_preview])
        random_btn.click(
            fn=load_random_example,
            outputs=[image_input, question_input, example_answer_note],
        ).then(
            fn=update_question_preview,
            inputs=[question_input],
            outputs=[question_preview],
        )

        solve_btn.click(
            fn=solve_math_problem_stream,
            inputs=[model_dropdown, question_input, image_input],
            outputs=[status_box, thinking_box, final_answer_box],
        )
        question_input.submit(
            fn=solve_math_problem_stream,
            inputs=[model_dropdown, question_input, image_input],
            outputs=[status_box, thinking_box, final_answer_box],
        )
        clear_btn.click(
            fn=clear_ui,
            outputs=[image_input, question_input, question_preview, status_box, thinking_box, final_answer_box],
        )

        for idx, btn in enumerate(example_buttons):
            btn.click(
                fn=lambda i=idx: load_example(i),
                outputs=[image_input, question_input, example_answer_note],
            ).then(
                fn=update_question_preview,
                inputs=[question_input],
                outputs=[question_preview],
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVision Gradio demo app")
    parser.add_argument("--server-name", default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7861")))
    parser.add_argument("--share", action="store_true", default=os.getenv("GRADIO_SHARE", "").lower() == "true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preload_key = "ft_base"
    try:
        _, preload_status = RUNTIME.ensure_model(preload_key)
        print(f"[MathVision Demo] {preload_status}")
    except Exception as exc:
        print(f"[MathVision Demo] Khong preload duoc model mac dinh ({MODEL_SPECS[preload_key].label}): {exc}")

    demo = build_demo()
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
