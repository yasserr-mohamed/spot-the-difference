"""
Spot the Difference - web app.
Run:  pip install gradio opencv-python numpy      then      python app.py
"""
import os
import cv2
import numpy as np
import gradio as gr
from spot_diff import find_differences, draw_boxes, amplify

HERE = os.path.dirname(os.path.abspath(__file__))


def find_demo(name):
    """Return the path of demo_<name>.jpg (or .png), looking in assets/ first and then next to this file."""
    for folder in (os.path.join(HERE, "assets"), HERE):
        for ext in (".jpg", ".png"):
            p = os.path.join(folder, f"demo_{name}{ext}")
            if os.path.exists(p):
                return p
    return None


DEMO_BEFORE, DEMO_AFTER = find_demo("before"), find_demo("after")      # the two default photos


def run(before_rgb, after_rgb, threshold, min_area_pct, blur, ignore_brightness, x1, y1, x2, y2):
    if before_rgb is None or after_rgb is None:
        return None, None, "Upload BOTH photos (before and after).", []
    if x2 <= x1 or y2 <= y1:
        return None, None, "The region is empty: x2 must be bigger than x1 and y2 bigger than y1.", []

    before = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    after = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)
    after, diff, mask, boxes = find_differences(before, after, threshold, min_area_pct, blur,
                                                ignore_brightness, (x1, y1, x2, y2))
    result = cv2.cvtColor(draw_boxes(after, boxes), cv2.COLOR_BGR2RGB)
    diff_map = amplify(diff)

    H, W = before.shape[:2]
    rows = [[i, f"({x}, {y})", f"{w} x {h}", f"{100 * w * h / (W * H):.2f} %"] for i, (x, y, w, h) in enumerate(boxes, 1)]
    msg = f"Found {len(boxes)} difference(s)." if boxes else "No differences found (try lowering the threshold)."
    if len(boxes) > 25:
        msg += "\nThat is a lot - the camera probably moved between the photos, or the threshold is too low."
    return result, diff_map, msg, rows


with gr.Blocks(title="Spot the Difference") as demo:
    gr.Markdown("# 🔍 Spot the Difference\nUpload two photos of the **same scene taken from the same place** "
                "(before / after). The app boxes everything that changed.\n\nThe page opens with a demo pair already loaded - upload your own photos to replace them.")
    with gr.Row():
        before = gr.Image(value=DEMO_BEFORE, type="numpy", label="1) BEFORE", sources=["upload", "clipboard"], height=330)
        after = gr.Image(value=DEMO_AFTER, type="numpy", label="2) AFTER", sources=["upload", "clipboard"], height=330)
    with gr.Row():
        threshold = gr.Slider(1, 150, value=40, step=1, label="Sensitivity threshold (lower = detects smaller changes)")
        min_area = gr.Slider(0.01, 3, value=0.3, step=0.01, label="Ignore changes smaller than (% of image)")
        blur = gr.Slider(1, 41, value=9, step=2, label="Noise blur (bigger = ignores tiny shifts and JPEG noise)")
    ignore_b = gr.Checkbox(True, label="Ignore overall brightness change (e.g. one photo is a bit darker)")
    with gr.Accordion("Compare only a region (ROI, in % of the image)", open=False):
        with gr.Row():
            x1 = gr.Slider(0, 100, value=0, step=1, label="x1 (left)")
            y1 = gr.Slider(0, 100, value=0, step=1, label="y1 (top)")
            x2 = gr.Slider(0, 100, value=100, step=1, label="x2 (right)")
            y2 = gr.Slider(0, 100, value=100, step=1, label="y2 (bottom)")
    btn = gr.Button("Find differences", variant="primary")
    msg = gr.Textbox(label="Result", lines=2)
    with gr.Row():
        out = gr.Image(label="Differences (red boxes)", height=420)
        dmap = gr.Image(label="Difference map (brighter = bigger change)", height=420)
    table = gr.Dataframe(headers=["#", "position (x, y)", "size (w x h)", "% of image"], label="Differences")

    ins = [before, after, threshold, min_area, blur, ignore_b, x1, y1, x2, y2]
    outs = [out, dmap, msg, table]
    btn.click(run, ins, outs)
    for c in (threshold, min_area, blur, ignore_b, x1, y1, x2, y2):
        c.release(run, ins, outs) if hasattr(c, "release") else c.change(run, ins, outs)

    if DEMO_BEFORE and DEMO_AFTER:
        gr.Examples(examples=[[DEMO_BEFORE, DEMO_AFTER]], inputs=[before, after], label="Click to load the demo pair again")
        demo.load(run, ins, outs)                 # show the result for the default photos right away

if __name__ == "__main__":
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False
    demo.launch(share=in_colab)
