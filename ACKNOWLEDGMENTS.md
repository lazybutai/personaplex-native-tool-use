# Acknowledgments and research provenance

## DuplexSLA

The research direction for this project was prompted by
[`hyzhang24/DuplexSLA`](https://github.com/hyzhang24/DuplexSLA): **“A
Full-Duplex Spoken Language Model with Synchronized Speech, Language, and
Action.”** DuplexSLA argues that continuous user audio, assistant speech, and a
structured action stream should share one conversational clock so planning,
turn-taking, and tool calls are native model behavior rather than an external
cascade.

That framing determined how we approached this task:

1. reject ASR → planner → TTS as the core architecture;
2. add a model-owned action channel to a duplex speech backbone;
3. keep listening, speaking, actions, and tool observations causally aligned;
4. return tool results to the continuing neural state rather than restarting a
   text agent after the call;
5. judge success on the full speech → call → result → native speech trajectory,
   not only on valid JSON generation.

This repository is **not** the official DuplexSLA implementation and does not
contain DuplexSLA code or weights. At the time this work began, its public
repository described the architecture and technical report while listing its
inference code, model checkpoint, and benchmark as forthcoming. We therefore
built an independent experiment on the available
[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
backbone.

### Architectural relationship

| Topic | DuplexSLA | This repository |
| --- | --- | --- |
| Backbone | Step-Audio-2-mini, approximately 7B | PersonaPlex/Moshi 7B |
| Shared timing | 160 ms chunk timeline | 80 ms PersonaPlex frame clock |
| Action representation | Rate-limited textual action stream | Typed five-slot action lane |
| External observations | Described as synchronized action/tool behavior | Separate typed environment lane |
| Tool identity | Structured textual calls | Grammar IDs plus schema validation |
| Call/result binding | Duplex action chronology | Authenticated `REF` leases and event FIFO |
| Release here | Not applicable | Independent research adapter and runtime |

### Citation

```bibtex
@article{zhang2026duplexsla,
  title   = {{DuplexSLA}: A Full-Duplex Spoken Language Model with Synchronized Speech, Language, and Action},
  author  = {Zhang, Haoyang and Chen, Jun and Wu, Donghang and Li, Yuxin and Zhang, Yuxin and Zhang, Xiangyu Tony and Liu, Che and Lin, Qingjian and Peng, Yizhou and Liu, Hexin and Chng, Eng Siong and Yan, Chao and Wu, Boyong and Huang, Yechang and Yang, Xuerui and Yu, Gang and Tian, Fei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## PersonaPlex and Moshi

The implementation builds on NVIDIA PersonaPlex and its Kyutai Moshi lineage.
The base weights are not redistributed here. Source and model licensing details
are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
