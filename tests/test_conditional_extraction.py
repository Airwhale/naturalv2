from naturalv2.models.lm import VLLMModel
from naturalv2.pipeline.conditional_extraction import _detokenize_kwargs


def test_detokenize_disabled_for_in_process_vllm():
    llm = object.__new__(VLLMModel)

    assert _detokenize_kwargs(llm) == {"detokenize": False}


def test_detokenize_omitted_for_hosted_model():
    assert _detokenize_kwargs(object()) == {}
