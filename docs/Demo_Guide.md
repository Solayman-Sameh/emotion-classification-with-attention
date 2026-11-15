# Final demo recording

Record a clear 45-60 second clip with the model cache preloaded.

1. Launch `python app.py` and open the local URL.
2. Enter `I am so happy you made it!`; show the dominant emotion and six bars.
3. Try `I am not happy about this.` and show the changed distribution.
4. Inspect BiLSTM + Attention and hover over token weights. Explain that it is a separate model.
5. Try `happy happy sad happy` to show separate repeated-token occurrences.
6. Leave any model disagreement visible.

The completed recording is saved at `media/emotion-classification-demo.mp4`. Update the README demo
paragraph. Use H.264 MP4 with readable text and modest resolution; keep the file
comfortably below GitHub's per-file limit.

## Publication

From the project root, after adding the clip:

```bash
git add .
python -m unittest discover -s tests -v
python tools/verify_release.py
git diff --cached --stat
git commit -m "Add emotion classification project with attention demo"
git push -u origin main
```

Code, reports and the completed video are prepared together. Check GitHub's README, notebook, images and video after the
push. Publish the LinkedIn draft only after its links resolve.
