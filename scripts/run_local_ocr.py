"""Execute DeepSeek OCR 2 in an isolated, preconfigured CUDA environment."""
import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.image.is_file():
        parser.error("Provide a local checkpoint directory and a readable image file.")

    with redirect_stdout(sys.stderr), TemporaryDirectory(prefix="generationengine-ocr-") as scratch:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeek OCR 2 requires a visible CUDA GPU.")
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().cuda()
        text = model.infer(
            tokenizer,
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            image_file=str(args.image.resolve()),
            output_path=scratch,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            eval_mode=True,
        )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("DeepSeek OCR 2 returned no text.")
    print(json.dumps({"text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
