from naturalv2.models.lm import VLLMModel
from naturalv2.pipeline.conditional_extraction import _detokenize_kwargs


def test_detokenize_kwargs():
    assert _detokenize_kwargs(object.__new__(VLLMModel)) == {"detokenize": False}
    assert _detokenize_kwargs(object()) == {}
