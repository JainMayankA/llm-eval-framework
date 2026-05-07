"""
Pytest bootstrap.

Forces EVAL_USE_MOCK=true before any test module imports api.server, so the
suite is insulated from a developer's local .env (which may legitimately hold
real or placeholder OPENAI_API_KEY / EVAL_OPENAI_MODELS values).

python-dotenv's load_dotenv() defaults to override=False, so the value set
here is preserved when api.server later calls load_dotenv() at import time.
"""

import os

os.environ["EVAL_USE_MOCK"] = "true"
