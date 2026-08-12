# Generic vLLM OpenAI-compatible server for the NATURAL probs_model runs on Dispersed.
#
# Why a custom image at all: Dispersed has no container command override, so we bake a wrapper
# ENTRYPOINT that reads the served model + options from ENV (which Dispersed CAN set via the job's
# `env` field). One image serves any model — select it at launch via env, no rebuild.
#
# Base is pinned to vLLM 0.22.0 to MATCH naturalv2's pinned `vllm==0.22.0`, so the prompt_logprobs
# wire format the server returns matches exactly what NATURAL's client parses.
#
# Weights are NOT baked in (keeps the image small and within CI disk limits); vLLM downloads the
# model from HuggingFace at container start. hf_transfer speeds that up (~a few min of GPU time).
#
# prompt_logprobs (NATURAL's teacher-forced scoring) is a request-time parameter the stock vLLM
# OpenAI server already returns — no special launch flag needed.
FROM vllm/vllm-openai:v0.22.0

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install --no-cache-dir hf_transfer

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip any CRLF (Windows) so the container's bash reads the shebang cleanly, then make executable.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
