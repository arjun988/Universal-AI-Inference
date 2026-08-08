from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print("encode Hi", t.encode("Hi"))
print("decode", repr(t.decode(t.encode("Hi"))))
ids = t.apply_chat_template(
    [{"role": "user", "content": "hi"}],
    tokenize=True,
    add_generation_prompt=True,
)
print("chat ids", ids)
print("chat text", t.decode(ids))
