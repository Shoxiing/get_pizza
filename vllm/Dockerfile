FROM vllm/vllm-openai:v0.32.0-cu129
# torchcodec в этом образе собран под CUDA 13 и падает на импорте (libnvrtc.so.13);
# для текстовой LLM он не нужен
RUN pip uninstall -y torchcodec
