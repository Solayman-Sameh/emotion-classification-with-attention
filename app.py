"""Launch the Six Emotions + Attention Gradio Application."""
import os
import sys
import pathlib
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parent
APP_DIR = ROOT / 'application'
sys.path.insert(0, str(APP_DIR))

spec = importlib.util.spec_from_file_location("app_internal", APP_DIR / "app.py")
app_internal = importlib.util.module_from_spec(spec)
sys.modules["app_internal"] = app_internal
spec.loader.exec_module(app_internal)

build = app_internal.build

if __name__ == '__main__':
    app_internal.launch(root=APP_DIR)
