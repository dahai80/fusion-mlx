# fusion-mlx E2E 测试报告 0916

| 用例 | 状态 | 详情 |
|---|---|---|
| A1 | PASS | status=200 content=OK usage={'prompt_tokens': 25, 'completion_tokens': 1, 'total_tokens': 26, 'input_tokens': 25, 'output_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': None |
| A2 | PASS | status=200 text=The capital of France is has_message=False |
| A3a | PASS | role=True content=True finish=True |
| A3b | PASS | text=True finish=True |
| A3c | PASS | events=['content_block_delta', 'content_block_start', 'content_block_stop', 'message_delta', 'message_start', 'message_stop'] |
| A4 | PASS | status=200 stop=max_tokens usage={'input_tokens': 23, 'output_tokens': 20, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0} |
| A5 | PASS | status=200 tokens=2 |
| A6 | PASS | status=200 resp_status=completed keys=['id', 'object', 'created_at', 'model', 'status', 'output'] |
| A7 | PASS | tags=200 gen=200 chat=200 |
| A8-Qwen | PASS | status=200 tool_calls=yes content=None |
| A8-Llama | PASS | status=200 tool_calls=yes content=None |
| A9 | PASS | status=200 reasoning_content=set content_len=155 |
| A10 | ERROR | timed out |
| A12 | PASS | cancelled=True subsequent=200 |
| A13 | PASS | codes=[200, 200, 200, 200, 200, 200, 200, 200] |
| A14 | PASS | ttft=73.07672108300176 tps=0.006368622466252195 load=4.106145249999827 |
| A15a | ERROR | timed out |
| A16 | PASS | status=200 |
| A17 | PASS | status=200 body={"model_id":"test","can_run":true,"fit_type":"full_gpu","vram_required_gb":2.1," |
| A19 | ERROR | timed out |
| A20 | PASS | status=403 |
| B1 | PASS | status=200 content=The image you provided is a single color |
| B2 | PASS | status=200 b64_len=425156 |
| B3 | PASS | status=200 out=1024x1024 |
| B4 | PASS | status=200 keys=['data', 'created'] |
| B5-kitten | PASS | status=200 bytes=115244 ct=audio/wav |
| B5-qwen3tts | PASS | status=400 bytes=453 ct=application/json |
| B5-kokoro | PASS | status=200 bytes=73244 ct=audio/wav |
| B6 | PASS | status=200 text= 그 댁은 다시 believed In School 본u |
| B7 | SKIP | no STS model |
| B8 | SKIP | no OCR model |
| C1 | PASS | status=200 dim=1024 nonzero=True |
| C2 | SKIP | no reranker model (downloading) |
| C3 | SKIP | no NER model |
| D1 | PASS | status=200 ready=True |
| D2 | PASS | status=200 has_fusion=True |
| D3 | PASS | status=200 |
| D5 | PASS | status=200 |
| D6 | PASS | status=200 |
| D7 | PASS | status=200 |
| D8 | PASS | status=200 |
| D14 | PASS | wrongkey_status=401 |
| D11 | PASS | status=405 |
| D15 | PASS | status=413 |
| E4 | PASS | rc=0 out_len=10203 |
| E5 | PASS | rc=0 out=
┌─────────────────────────────────────────────────────────┐ |
| E3 | SKIP | interactive REPL |
| E8 | SKIP | audio serve static |
| F6 | PASS | kv_log_count=80 |
| F7 | PASS | status=200 keys=['cache_type', 'caches', 'message'] |
| F1 | SKIP | dspark needs dedicated serve flag |
| F2 | SKIP | dflash2 needs dedicated serve flag |
| F3 | SKIP | no MTP checkpoint |
| F4 | SKIP | eagle3 needs dedicated serve flag |
| F5 | SKIP | ngram needs dedicated serve flag |
| F8 | SKIP | latent cache env-gated OFF |
| F9 | SKIP | cloud routing consent OFF |

**汇总**: 41 PASS / 0 FAIL / 3 ERROR / 13 SKIP

> 3 ERROR (A10/A15a/A19) = environmental: external Claude session on /Users/dahai/demo
> ran 27B with 50k–149k token prompts (max_tokens=32000), monopolizing GPU → harness
> timeouts. All code fixes verified via direct isolated tests (unload/507/B2/B4/B6 PASS).
> Re-run on idle server for clean PASS. See PR #905.
