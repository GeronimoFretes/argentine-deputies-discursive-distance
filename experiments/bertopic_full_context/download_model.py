"""
One-time download helper for BAAI/bge-m3 to local HF cache.

This script must NOT be committed to the repository in its current form because
it applies a global SSL workaround (required on this Windows machine due to a
corporate/system certificate store issue). The SSL patch is intentional and
documented here, not hidden.

To run:
    .\.venv-bertopic-full\Scripts\python.exe experiments/bertopic_full_context/download_model.py

The model is saved to the default HF cache (~/.cache/huggingface/hub/).
Once downloaded, run_benchmark.py loads it locally without network access.
"""

import ssl
import os
import warnings

# --- SSL workaround for Windows certificate store issues ---
# This is intentional and documented. Do NOT copy this pattern into run_benchmark.py.
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import httpx
    _orig_client = httpx.Client.__init__

    def _ssl_client(self, *a, **kw):
        kw["verify"] = False
        _orig_client(self, *a, **kw)

    httpx.Client.__init__ = _ssl_client
    _orig_async = httpx.AsyncClient.__init__

    def _ssl_async(self, *a, **kw):
        kw["verify"] = False
        _orig_async(self, *a, **kw)

    httpx.AsyncClient.__init__ = _ssl_async
except ImportError:
    pass

from huggingface_hub import snapshot_download

print("Downloading BAAI/bge-m3 to local HF cache...")
path = snapshot_download(
    "BAAI/bge-m3",
    ignore_patterns=["*.gguf", "colbert*", "sparse*"],
)
print(f"Downloaded to: {path}")
print("You can now run run_benchmark.py without network access.")
