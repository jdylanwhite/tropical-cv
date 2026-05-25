"""
Tropical Cyclone SAM3 Adjudication GUI
======================================
Draws a random SSHS >2 tile, runs SAM3, and lets you:
  - Drag the bounding box corners / edges to adjust it
  - Accept  → saves entry to adjudicated_labels.json
  - Deny    → skips, logs as rejected
  - Next    → skip without saving
  - Cycle through prompt results with Prev / Next Prompt buttons

Usage:
    python tc_adjudicator.py
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image, ImageTk

# ── SAM3 ──────────────────────────────────────────────────────────────────────
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── Config ────────────────────────────────────────────────────────────────────
METADATA_PATH   = "/Users/dylanwhite/Projects/tropical-cv/data/training/image_data.json"
OUTPUT_PATH     = "adjudicated_labels.json"
CONFIDENCE      = 0.5
DISPLAY_SIZE    = 700          # canvas pixels (square)
HANDLE_R        = 7            # corner handle radius px
EDGE_HIT        = 10           # px tolerance for edge drag

TEXT_PROMPTS = [
    "hurricane",
    "tropical cyclone",
    "eye of hurricane",
    "hurricane eye wall",
    "tropical cyclone eye",
    "circular storm",
    "circular storm eye",
]

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL_BG  = "#161b22"
ACCENT    = "#58a6ff"
ACCEPT_C  = "#3fb950"
DENY_C    = "#f85149"
SKIP_C    = "#8b949e"
TEXT_C    = "#e6edf3"
BOX_C     = "#f0c040"
MASK_A    = 100               # 0-255 alpha for mask overlay

# ═══════════════════════════════════════════════════════════════════════════════
class SAM3App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("TC SAM3 Adjudicator")
        self.configure(bg=BG)
        self.resizable(False, False)

        # ── state ──
        self.model     = None
        self.processor = None
        self.image_data   = []
        self.sample       = None
        self.pil_image    = None
        self.tk_image     = None
        self.scale        = 1.0           # display / native
        self.results      = []            # list of (prompt, masks_np, boxes_np, scores_np)
        self.result_idx   = 0
        self.output_records = self._load_output()

        # drag state
        self._drag_mode  = None           # 'move','n','s','e','w','nw','ne','sw','se'
        self._drag_start = None
        self._box_orig   = None

        # current editable box  [x0,y0,x1,y1]  in display coords
        self.cur_box = None

        self._build_ui()
        self.after(100, self._init_model)

    # ── persist ───────────────────────────────────────────────────────────────
    def _load_output(self):
        p = Path(OUTPUT_PATH)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return []

    def _save_output(self):
        with open(OUTPUT_PATH, "w") as f:
            json.dump(self.output_records, f, indent=2)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── left panel: canvas ────────────────────────────────────────────────
        left = tk.Frame(self, bg=BG)
        left.pack(side=tk.LEFT, padx=(16, 8), pady=16)

        self.canvas = tk.Canvas(
            left, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
            bg="#000", highlightthickness=1,
            highlightbackground="#30363d"
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)

        # ── right panel ───────────────────────────────────────────────────────
        right = tk.Frame(self, bg=PANEL_BG, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 16), pady=16)
        right.pack_propagate(False)

        # title
        tk.Label(right, text="TC Adjudicator", font=("Courier", 16, "bold"),
                 fg=ACCENT, bg=PANEL_BG).pack(pady=(20, 4))
        tk.Label(right, text="SAM3 · MLX · Apple Silicon",
                 font=("Courier", 9), fg=SKIP_C, bg=PANEL_BG).pack()

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # status
        self.status_var = tk.StringVar(value="Initialising SAM3…")
        tk.Label(right, textvariable=self.status_var,
                 font=("Courier", 10), fg=TEXT_C, bg=PANEL_BG,
                 wraplength=240, justify=tk.LEFT).pack(padx=12)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # sample info
        self.info_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info_var,
                 font=("Courier", 9), fg=SKIP_C, bg=PANEL_BG,
                 wraplength=240, justify=tk.LEFT).pack(padx=12)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # prompt nav
        tk.Label(right, text="PROMPT", font=("Courier", 9, "bold"),
                 fg=SKIP_C, bg=PANEL_BG).pack(anchor=tk.W, padx=12)

        nav = tk.Frame(right, bg=PANEL_BG)
        nav.pack(fill=tk.X, padx=12, pady=4)
        self._btn(nav, "◀ Prev", self._prev_prompt).pack(side=tk.LEFT)
        self._btn(nav, "Next ▶", self._next_prompt).pack(side=tk.RIGHT)

        self.prompt_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.prompt_var,
                 font=("Courier", 11, "bold"), fg=ACCENT, bg=PANEL_BG,
                 wraplength=240).pack(pady=(0, 2))

        self.score_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.score_var,
                 font=("Courier", 9), fg=TEXT_C, bg=PANEL_BG).pack()

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # box coords readout
        tk.Label(right, text="BOUNDING BOX", font=("Courier", 9, "bold"),
                 fg=SKIP_C, bg=PANEL_BG).pack(anchor=tk.W, padx=12)
        self.box_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.box_var,
                 font=("Courier", 10), fg=TEXT_C, bg=PANEL_BG).pack(pady=(2, 0))

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # action buttons
        self._btn(right, "✓  Accept", self._accept,
                  fg="white", bg=ACCEPT_C, pad=10).pack(fill=tk.X, padx=12, pady=4)
        self._btn(right, "✗  Deny",   self._deny,
                  fg="white", bg=DENY_C,   pad=10).pack(fill=tk.X, padx=12, pady=4)
        self._btn(right, "→  Next image", self._next_image,
                  fg="white", bg=SKIP_C,   pad=8).pack(fill=tk.X, padx=12, pady=4)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=12)

        # stats
        self.stats_var = tk.StringVar(value="Saved: 0  Denied: 0")
        tk.Label(right, textvariable=self.stats_var,
                 font=("Courier", 9), fg=SKIP_C, bg=PANEL_BG).pack()

        tk.Label(right, text=f"Output → {OUTPUT_PATH}",
                 font=("Courier", 8), fg="#444c56", bg=PANEL_BG,
                 wraplength=240).pack(side=tk.BOTTOM, pady=12)

    def _btn(self, parent, text, cmd, fg=TEXT_C, bg="#21262d", pad=6):
        """Label-based button that actually shows colours on macOS."""
        lbl = tk.Label(
            parent, text=text,
            font=("Courier", 11, "bold"),
            fg=fg, bg=bg,
            relief=tk.FLAT, cursor="hand2",
            padx=pad, pady=pad
        )
        lbl.bind("<ButtonRelease-1>", lambda e: cmd())
        # subtle hover darkening
        darker = self._darken(bg)
        lbl.bind("<Enter>", lambda e: lbl.config(bg=darker))
        lbl.bind("<Leave>", lambda e: lbl.config(bg=bg))
        return lbl

    @staticmethod
    def _darken(hex_color, factor=0.75):
        """Return a slightly darker version of a hex colour."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return "#{:02x}{:02x}{:02x}".format(
            int(r * factor), int(g * factor), int(b * factor)
        )

    # ── model init ────────────────────────────────────────────────────────────
    def _init_model(self):
        self.status_var.set("Loading SAM3 weights…")
        self.update()
        try:
            self.model     = build_sam3_image_model()
            self.processor = Sam3Processor(self.model,
                                           confidence_threshold=CONFIDENCE)
        except Exception as e:
            messagebox.showerror("SAM3 load failed", str(e))
            return

        # load metadata
        try:
            with open(METADATA_PATH) as f:
                image_data = json.load(f)
            self.image_data = [x for x in image_data["images"] if x.get("sshs", 0) > 2]
        except Exception as e:
            messagebox.showerror("Metadata error", str(e))
            return

        self.status_var.set(f"Ready — {len(self.image_data)} tiles (SSHS>2)")
        self._next_image()

    # ── image / inference pipeline ────────────────────────────────────────────
    def _next_image(self):
        if not self.image_data:
            self.status_var.set("No images loaded.")
            return

        self.sample = np.random.choice(self.image_data)
        filepath    = self.sample["file_name"]
        invert      = True

        self.status_var.set("Loading tile…")
        self.update()

        try:
            data     = xr.open_dataset(filepath)
            rad_data = data.Rad.values
            rad_data = np.where(np.isnan(rad_data), 0.0, rad_data)
            data.close()

            mn, mx   = np.nanmin(rad_data), np.nanmax(rad_data)
            normed   = (rad_data - mn) / (mx - mn + 1e-9)
            normed   = np.nan_to_num(normed, nan=0.0)
            uint8    = (normed * 255).astype(np.uint8)
            if invert:
                uint8 = 255 - uint8

            # convert grey → RGB so PIL handles masks easily
            self.pil_image = Image.fromarray(uint8).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image load error", str(e))
            return

        # scale to display size
        native_w, native_h = self.pil_image.size
        self.scale = DISPLAY_SIZE / max(native_w, native_h)

        sshs = self.sample.get("sshs", "?")
        name = Path(filepath).name
        self.info_var.set(f"File: {name}\nSSHS: {sshs}")
        self._update_stats()

        self._run_sam3()

    def _run_sam3(self):
        self.status_var.set("Running SAM3 on all prompts…")
        self.update()

        self.results    = []
        self.result_idx = 0

        try:
            state = self.processor.set_image(self.pil_image)
            for prompt in TEXT_PROMPTS:
                state_p = self.processor.set_text_prompt(prompt, state)
                masks_r  = state_p["masks"]
                boxes_r  = state_p["boxes"]
                scores_r = state_p["scores"]

                if len(scores_r) == 0:
                    continue

                masks_np  = np.array(masks_r)
                boxes_np  = np.array(boxes_r)
                scores_np = np.array(scores_r)

                # squeeze to (N, H, W)
                while masks_np.ndim > 3:
                    masks_np = masks_np.squeeze(axis=1)
                if masks_np.ndim == 2:
                    masks_np = masks_np[np.newaxis]

                # filter whole-image masks
                frac = masks_np.sum(axis=(-1, -2)) / (masks_np.shape[-1] * masks_np.shape[-2])
                keep = frac < 0.5
                if not keep.any():
                    continue

                masks_np  = masks_np[keep]
                boxes_np  = boxes_np[keep]
                scores_np = scores_np[keep]

                self.results.append((prompt, masks_np, boxes_np, scores_np))

        except Exception as e:
            messagebox.showerror("SAM3 error", str(e))
            return

        if not self.results:
            self.status_var.set("No detections — loading next…")
            self.after(800, self._next_image)
            return

        self.status_var.set(f"Found results for {len(self.results)} prompt(s).")
        self._show_result()

    def _show_result(self):
        if not self.results:
            return

        prompt, masks_np, boxes_np, scores_np = self.results[self.result_idx]

        # best (highest score) detection
        best = int(np.argmax(scores_np))
        mask  = masks_np[best]
        box   = boxes_np[best]   # native coords
        score = float(scores_np[best])

        # scale box to display coords
        s = self.scale
        self.cur_box = [box[0]*s, box[1]*s, box[2]*s, box[3]*s]

        self.prompt_var.set(f'"{prompt}"')
        self.score_var.set(f"confidence: {score:.3f}")
        self._update_box_readout()

        self._draw(mask, prompt, score)

    def _draw(self, mask, prompt, score):
        """Render image + mask overlay onto canvas."""
        # resize image
        disp = self.pil_image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)

        # mask overlay
        mh, mw = mask.shape[-2], mask.shape[-1]
        mask_bool = (mask > 0).astype(np.uint8)
        overlay = np.zeros((mh, mw, 4), dtype=np.uint8)
        overlay[mask_bool > 0] = (88, 166, 255, MASK_A)   # ACCENT-ish blue
        mask_img = Image.fromarray(overlay, "RGBA")
        mask_img = mask_img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)

        composite = disp.convert("RGBA")
        composite.alpha_composite(mask_img)
        composite = composite.convert("RGB")

        self.tk_image = ImageTk.PhotoImage(composite)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self._draw_box()

        n = self.result_idx + 1
        self.canvas.create_text(
            8, 8, anchor=tk.NW,
            text=f"Prompt {n}/{len(self.results)}: {prompt}  ({score:.2f})",
            font=("Courier", 10, "bold"), fill=BOX_C
        )

    def _draw_box(self):
        """Draw the editable bounding box + handles."""
        self.canvas.delete("bbox")
        if self.cur_box is None:
            return
        x0, y0, x1, y1 = self.cur_box
        r = HANDLE_R

        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            outline=BOX_C, width=2, tags="bbox"
        )
        # corner handles
        for cx, cy in [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]:
            self.canvas.create_rectangle(
                cx-r, cy-r, cx+r, cy+r,
                outline=BOX_C, fill="#0d1117", width=2, tags="bbox"
            )

    # ── drag interaction ───────────────────────────────────────────────────────
    def _hit_test(self, x, y):
        """Return drag mode string for (x,y) relative to cur_box."""
        if self.cur_box is None:
            return None
        x0, y0, x1, y1 = self.cur_box
        r, e = HANDLE_R+2, EDGE_HIT

        on_l = abs(x - x0) < e
        on_r = abs(x - x1) < e
        on_t = abs(y - y0) < e
        on_b = abs(y - y1) < e

        if on_l and on_t: return "nw"
        if on_r and on_t: return "ne"
        if on_l and on_b: return "sw"
        if on_r and on_b: return "se"
        if on_l:          return "w"
        if on_r:          return "e"
        if on_t:          return "n"
        if on_b:          return "s"

        if x0 < x < x1 and y0 < y < y1:
            return "move"
        return None

    def _on_hover(self, ev):
        mode = self._hit_test(ev.x, ev.y)
        cursors = {
            "move":"fleur","n":"sb_v_double_arrow","s":"sb_v_double_arrow",
            "e":"sb_h_double_arrow","w":"sb_h_double_arrow",
            "nw":"size_nw_se","se":"size_nw_se",
            "ne":"size_ne_sw","sw":"size_ne_sw",
        }
        self.canvas.config(cursor=cursors.get(mode, ""))

    def _on_press(self, ev):
        self._drag_mode  = self._hit_test(ev.x, ev.y)
        self._drag_start = (ev.x, ev.y)
        self._box_orig   = list(self.cur_box) if self.cur_box else None

    def _on_drag(self, ev):
        if not self._drag_mode or not self._box_orig:
            return
        dx = ev.x - self._drag_start[0]
        dy = ev.y - self._drag_start[1]
        x0, y0, x1, y1 = self._box_orig
        m = self._drag_mode

        if m == "move":
            x0+=dx; x1+=dx; y0+=dy; y1+=dy
        if "n" in m: y0 += dy
        if "s" in m: y1 += dy
        if "w" in m: x0 += dx
        if "e" in m: x1 += dx

        # clamp
        x0 = max(0, min(x0, DISPLAY_SIZE))
        x1 = max(0, min(x1, DISPLAY_SIZE))
        y0 = max(0, min(y0, DISPLAY_SIZE))
        y1 = max(0, min(y1, DISPLAY_SIZE))
        if x1 - x0 < 4: x1 = x0 + 4
        if y1 - y0 < 4: y1 = y0 + 4

        self.cur_box = [x0, y0, x1, y1]
        self._draw_box()
        self._update_box_readout()

    def _on_release(self, ev):
        self._drag_mode  = None
        self._drag_start = None
        self._box_orig   = None

    # ── prompt navigation ──────────────────────────────────────────────────────
    def _prev_prompt(self):
        if not self.results: return
        self.result_idx = (self.result_idx - 1) % len(self.results)
        self._show_result()

    def _next_prompt(self):
        if not self.results: return
        self.result_idx = (self.result_idx + 1) % len(self.results)
        self._show_result()

    # ── accept / deny ─────────────────────────────────────────────────────────
    def _accept(self):
        if self.cur_box is None or self.sample is None:
            return
        s = self.scale
        x0, y0, x1, y1 = self.cur_box
        native_box = [x0/s, y0/s, x1/s, y1/s]

        prompt, _, _, scores_np = self.results[self.result_idx]
        best  = int(np.argmax(scores_np))
        score = float(scores_np[best])

        record = {
            "file_name": str(self.sample["file_name"]),
            "sshs":      int(self.sample["sshs"]) if self.sample.get("sshs") is not None else None,
            "prompt":    prompt,
            "score":     round(float(score), 4),
            "box_xyxy":  [round(float(v), 2) for v in native_box],
            "status":    "accepted",
        }
        self.output_records.append(record)
        self._save_output()
        self.status_var.set("✓ Accepted — next image…")
        self._update_stats()
        self.after(400, self._next_image)

    def _deny(self):
        if self.sample is None:
            return
        prompt = self.results[self.result_idx][0] if self.results else "—"
        record = {
            "file_name": str(self.sample["file_name"]),
            "sshs":      int(self.sample["sshs"]) if self.sample.get("sshs") is not None else None,
            "prompt":    prompt,
            "status":    "denied",
        }
        self.output_records.append(record)
        self._save_output()
        self.status_var.set("✗ Denied — next image…")
        self._update_stats()
        self.after(400, self._next_image)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _update_box_readout(self):
        if self.cur_box is None:
            self.box_var.set("—")
            return
        s = self.scale
        x0, y0, x1, y1 = self.cur_box
        self.box_var.set(
            f"x0={x0/s:.0f}  y0={y0/s:.0f}\nx1={x1/s:.0f}  y1={y1/s:.0f}"
        )

    def _update_stats(self):
        acc  = sum(1 for r in self.output_records if r.get("status") == "accepted")
        den  = sum(1 for r in self.output_records if r.get("status") == "denied")
        self.stats_var.set(f"Saved: {acc}   Denied: {den}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SAM3App()
    app.mainloop()