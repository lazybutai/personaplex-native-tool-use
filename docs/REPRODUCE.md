# Reproduce the v169 timer proof on Windows

## Requirements

- Windows 11 and PowerShell
- Python 3.11
- CUDA-capable PyTorch
- sufficient VRAM for PersonaPlex 7B plus Mimi on separate GPUs
- Hugging Face access to `nvidia/personaplex-7b-v1`

The proven configuration used Mimi on `cuda:1`, PersonaPlex on `cuda:3`, and did
not use GPU 2.

## 1. Clone and patch PersonaPlex

```powershell
git clone https://github.com/lazybutai/personaplex-native-tool-use.git
cd personaplex-native-tool-use
.\scripts\install_personaplex_patch.ps1
```

The installer clones the pinned upstream PersonaPlex commit and copies the
published Moshi extensions into `fork/personaplex-agent-moshi`.

## 2. Create the environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -e .\fork\personaplex-agent-moshi\moshi
.\.venv\Scripts\python.exe -m pip install -e . pytest sounddevice
```

## 3. Download model assets

Accept NVIDIA’s PersonaPlex model terms first, then run:

```powershell
hf download nvidia/personaplex-7b-v1 --local-dir .\models\personaplex-7b-v1
hf download lazybutai/personaplex-native-tool-use `
  micro-head-v169-v151-dual-state-margin05.safetensors `
  --local-dir .\checkpoints
```

Verify the adapter:

```powershell
(Get-FileHash -Algorithm SHA256 `
  .\checkpoints\micro-head-v169-v151-dual-state-margin05.safetensors).Hash
```

Expected:

```text
F9C5DB254F8FEA477155A9233B063B03248605C07E95EFB987D88BC3626CDCBA
```

## 4. Run selected CPU contracts

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q tests
```

## 5. Run the frozen causal replay

```powershell
.\.venv\Scripts\python.exe .\scripts\live_session_v2.py `
  --mode replay `
  --replay-wav .\examples\timer-v169\input.wav `
  --mimi-device cuda:1 `
  --model-device cuda:3 `
  --tail-silence-frames 64 `
  --profile-components `
  --trace-output .\artifacts\v169-reproduction.json `
  --input-audio-output .\artifacts\v169-reproduction-input.wav `
  --assistant-audio-output .\artifacts\v169-reproduction-assistant.wav
```

This is a seed-pinned sampling trajectory. Performance and exact output can
change with PyTorch, CUDA, hardware, or altered kernels.

## 6. Verify the new trace

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_v169_trace.py `
  .\artifacts\v169-reproduction.json `
  --checkpoint .\checkpoints\micro-head-v169-v151-dual-state-margin05.safetensors
```

Success requires one exact call, 120 scheduled seconds, one visible `OK` result,
the pinned generated text, audible assistant metadata, and the checkpoint hash.
