"""Golden layout model: built from an OpenRAM GDS.

Responsibilities
----------------
* Flatten the GDS and keep per-layer polygons.
* Find functional regions (bitcell array, decoder, sense amps, ...) from the
  OpenRAM instance hierarchy.
* Find every bitcell instance, assign (row, col), and transform the bitcell's
  pin shapes (bl, br, wl, vdd, gnd, Q, Q_bar) into absolute coordinates so a
  defect can be mapped to real net names (bl_12, wl_5, ...).
* Render the golden image (work scale for inspection, print scale for the
  physical target) and per-layer masks.

Coordinates: layout micrometres (um), y up. Images: pixels, y down.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
# gdstk is only needed to build a design package (not at station runtime, e.g. on Android)

# scn4m_subm GDS layer numbers (OpenRAM technology/scn4m_subm/tech/tech.py)
LAYERS = {
    "nwell": 42, "active": 43, "poly": 46, "poly_contact": 47,
    "active_contact": 48, "m1": 49, "via1": 50, "m2": 51, "via2": 61,
    "m3": 62, "via3": 30, "m4": 31,
}
# Draw order (bottom -> top), BGR colour, alpha.  Chosen to look like a
# colour layout plot and to print well on paper.
RENDER_STYLE = [
    ("nwell", (200, 225, 240), 0.35),
    ("active", (90, 170, 60), 0.55),
    ("poly", (60, 60, 210), 0.55),
    ("m1", (200, 120, 30), 0.55),
    ("m2", (170, 60, 170), 0.50),
    ("m3", (40, 150, 230), 0.45),
    ("m4", (150, 150, 40), 0.40),
    ("poly_contact", (30, 30, 30), 0.95),
    ("active_contact", (30, 30, 30), 0.95),
    ("via1", (60, 30, 30), 0.95),
    ("via2", (30, 60, 60), 0.95),
    ("via3", (40, 40, 40), 0.95),
]
CUT_LAYERS = ("poly_contact", "active_contact", "via1", "via2", "via3")
LINE_LAYERS = ("poly", "m1", "m2", "m3", "m4")

# OpenRAM instance-name fragment -> region.  First match wins, so order matters.
REGION_RULES = [
    ("_replica_column", "DUMMY_ARRAY"),
    ("dummy_array", "DUMMY_ARRAY"),
    ("_replica_bitcell_array", None),          # container only
    ("_capped_replica_bitcell_array", None),   # container only
    ("_bitcell_array", "BITCELL_ARRAY"),
    ("sense_amp_array", "SENSE_AMP"),
    ("write_driver_array", "WRITE_DRIVER"),
    ("precharge_array", "PRECHARGE"),
    ("column_mux_array", "COL_MUX"),
    ("hierarchical_decoder", "ROW_DECODER"),
    ("wordline_driver_array", "ROW_DECODER"),
    ("and2_dec_0", "ROW_DECODER"),
    ("control_logic", "CONTROL"),
    ("row_addr_dff", "CONTROL"),
    ("col_addr_dff", "CONTROL"),
    ("data_dff", "CONTROL"),
    ("wmask_dff", "CONTROL"),
]
PERIPHERY = ("ROW_DECODER", "SENSE_AMP", "WRITE_DRIVER", "PRECHARGE", "COL_MUX", "CONTROL")
# When a defect overlaps several regions, report the most consequential one.
REGION_PRIORITY = ["CONTROL", "ROW_DECODER", "SENSE_AMP", "WRITE_DRIVER", "PRECHARGE",
                   "COL_MUX", "BITCELL_ARRAY", "DUMMY_ARRAY", "ROUTING", "OUTSIDE"]
PIN_NET = {"bl": "bl_{c}", "br": "br_{c}", "wl": "wl_{r}", "Q": "q_{r}_{c}",
           "Q_bar": "qb_{r}_{c}", "vdd": "vdd", "gnd": "gnd"}


def _ref_xform(ref) -> np.ndarray:
    a = np.deg2rad(ref.rotation or 0.0)
    m = np.eye(3)
    if ref.x_reflection:
        m = np.diag([1.0, -1.0, 1.0]) @ m
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.eye(3)
    t[:2, 2] = ref.origin
    return t @ r @ m


def _walk(cell, T=None, depth=0, out=None, maxdepth=8):
    T = np.eye(3) if T is None else T
    out = [] if out is None else out
    for ref in cell.references:
        m = _ref_xform(ref)
        offs = [(0.0, 0.0)]
        if ref.repetition is not None and ref.repetition.size > 0:
            offs = ref.repetition.offsets
        for off in offs:
            to = np.eye(3)
            to[:2, 2] = off
            g = T @ to @ m
            out.append((ref.cell, g, depth))
            if depth < maxdepth:
                _walk(ref.cell, g, depth + 1, out, maxdepth)
    return out


def _xf_rect(g, lo, hi):
    pts = np.array([[lo[0], lo[1], 1], [hi[0], hi[1], 1], [lo[0], hi[1], 1], [hi[0], lo[1], 1]], float).T
    q = (g @ pts)[:2]
    return [float(q[0].min()), float(q[1].min()), float(q[0].max()), float(q[1].max())]


@dataclass
class Design:
    """Everything the station needs to know about one golden design."""
    name: str
    gds: str
    top: str
    macro_bbox: list            # [x0, y0, x1, y1] um
    window: list                # inspected window [x0, y0, x1, y1] um
    work_ppu: float             # work-scale pixels per um
    print_ppu: float            # print-scale pixels per um
    print_border_px: int        # white border around the window in the print image
    marker_px: int              # ArUco marker size in print px
    markers: dict = field(default_factory=dict)   # id -> 4 corners [[x,y]..] in print px
    regions: list = field(default_factory=list)   # [{region, rect}]
    rows: int = 0
    cols: int = 0
    pitch: list = field(default_factory=list)     # [px_um, py_um]
    bitcells: list = field(default_factory=list)  # [{r, c, rect}]
    pins: list = field(default_factory=list)      # [{net, layer, rect, r, c}]
    spare_rows: int = 2
    spare_cols: int = 2
    spares_virtual: bool = True
    technology: str = "scn4m_subm"
    words: int = 0
    word_size: int = 0

    # ---------- coordinate helpers ----------
    @property
    def work_size(self):
        x0, y0, x1, y1 = self.window
        return int(round((x1 - x0) * self.work_ppu)), int(round((y1 - y0) * self.work_ppu))

    def um_to_work(self, x, y):
        x0, y0, x1, y1 = self.window
        return (x - x0) * self.work_ppu, (y1 - y) * self.work_ppu

    def work_to_um(self, px, py):
        x0, y0, x1, y1 = self.window
        return x0 + px / self.work_ppu, y1 - py / self.work_ppu

    def print_to_work_matrix(self) -> np.ndarray:
        """3x3 affine: print-image px -> work-image px."""
        s = self.work_ppu / self.print_ppu
        b = self.print_border_px
        return np.array([[s, 0, -b * s], [0, s, -b * s], [0, 0, 1]], float)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "design.json"), "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(folder) -> "Design":
        with open(os.path.join(folder, "design.json")) as f:
            d = json.load(f)
        return Design(**d)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_design(gds_path: str, name: str, window: str = "bank", margin_um: float = 6.0,
                   work_ppu: float = 4.0, print_ppu: float = 9.0, spare_rows: int = 2,
                   spare_cols: int = 2, bitcell_name: str = "cell_1rw", words: int = 0,
                   word_size: int = 0) -> Design:
    import gdstk
    lib = gdstk.read_gds(gds_path)
    top = lib.top_level()[0]
    bb = top.bounding_box()
    macro = [bb[0][0], bb[0][1], bb[1][0], bb[1][1]]
    insts = _walk(top)

    regions = []
    for cell, g, _ in insts:
        for frag, region in REGION_RULES:
            if frag in cell.name:
                if region is not None:
                    cb = cell.bounding_box()
                    regions.append({"region": region, "rect": _xf_rect(g, cb[0], cb[1]), "cell": cell.name})
                break

    # Inspected window
    if window == "full":
        win = list(macro)
    else:
        bank = next(((c, g) for c, g, _ in insts if c.name.endswith("_bank")), None)
        if bank is None:
            win = list(macro)
        else:
            cb = bank[0].bounding_box()
            win = _xf_rect(bank[1], cb[0], cb[1])
    win = [win[0] - margin_um, win[1] - margin_um, win[2] + margin_um, win[3] + margin_um]

    # Bitcells and their pins
    bcell = next(c for c in lib.cells if c.name == bitcell_name)
    pin_shapes = []  # (pin, layer, lo, hi) in cell coordinates
    for lab in bcell.labels:
        for p in bcell.polygons:
            if p.layer == lab.layer and p.contain(lab.origin):
                b = p.bounding_box()
                pin_shapes.append((lab.text, p.layer, b[0], b[1]))
                break
    # Cell footprint: the boundary shape (layer 63) tiles exactly; fall back to bbox.
    bnd = [p.bounding_box() for p in bcell.polygons if p.layer == 63]
    cell_lo, cell_hi = bnd[0] if bnd else bcell.bounding_box()
    bc_inst = [(g, _xf_rect(g, cell_lo, cell_hi)) for c, g, _ in insts if c is bcell]
    centers = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2] for _, r in bc_inst])
    xs = np.unique(np.round(centers[:, 0], 2))
    ys = np.unique(np.round(centers[:, 1], 2))
    px = float(np.median(np.diff(xs))) if len(xs) > 1 else 7.0
    py = float(np.median(np.diff(ys))) if len(ys) > 1 else 10.0
    bitcells, pins, seen = [], [], set()
    for (g, rect), (cx, cy) in zip(bc_inst, centers):
        c = int(np.argmin(np.abs(xs - round(cx, 2))))
        r = int(np.argmin(np.abs(ys - round(cy, 2))))
        if (r, c) in seen:
            continue
        seen.add((r, c))
        bitcells.append({"r": r, "c": c, "rect": rect})
        for pin, layer, lo, hi in pin_shapes:
            tmpl = PIN_NET.get(pin, pin)
            pins.append({"net": tmpl.format(r=r, c=c), "layer": int(layer),
                         "rect": _xf_rect(g, lo, hi), "r": r, "c": c})

    d = Design(name=name, gds=os.path.abspath(gds_path), top=top.name, macro_bbox=macro,
               window=win, work_ppu=work_ppu, print_ppu=print_ppu, print_border_px=0,
               marker_px=0, regions=regions, rows=len(ys), cols=len(xs), pitch=[px, py],
               bitcells=bitcells, pins=pins, spare_rows=spare_rows, spare_cols=spare_cols,
               words=words or len(ys), word_size=word_size or len(xs))
    return d


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def layer_polygons(gds_path: str):
    import gdstk
    lib = gdstk.read_gds(gds_path)
    top = lib.top_level()[0]
    out = {}
    for lname, num in LAYERS.items():
        out[lname] = [p.points for p in top.get_polygons(layer=num, datatype=0)]
    return out


def render_masks(polys: dict, design: Design, ppu: float, pad_px: int = 0):
    """Rasterise each layer into a uint8 mask (255 = present) at `ppu`."""
    x0, y0, x1, y1 = design.window
    w = int(round((x1 - x0) * ppu)) + 2 * pad_px
    h = int(round((y1 - y0) * ppu)) + 2 * pad_px
    shift = 4
    masks = {}
    for lname, plist in polys.items():
        m = np.zeros((h, w), np.uint8)
        pts = []
        for p in plist:
            q = np.empty_like(p)
            q[:, 0] = (p[:, 0] - x0) * ppu + pad_px
            q[:, 1] = (y1 - p[:, 1]) * ppu + pad_px
            pts.append(np.round(q * (1 << shift)).astype(np.int32))
        if pts:
            cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_8, shift=shift)
        masks[lname] = m
    return masks


def composite(masks: dict, background=(245, 245, 245)) -> np.ndarray:
    """Blend layer masks into a colour layout image (BGR uint8)."""
    any_mask = next(iter(masks.values()))
    img = np.empty(any_mask.shape + (3,), np.float32)
    img[:] = background
    for lname, color, alpha in RENDER_STYLE:
        m = masks.get(lname)
        if m is None or not m.any():
            continue
        a = (m.astype(np.float32) / 255.0)[..., None] * alpha
        img = img * (1 - a) + np.array(color, np.float32) * a
    return np.clip(img, 0, 255).astype(np.uint8)


def build_print_target(design: Design, polys: dict, out_png: str, dict_id=None,
                       marker_um: float = 22.0, border_um: float = 30.0):
    """Golden layout at print scale with four ArUco markers in a white border.

    Marker corner positions (print px) are stored in the design so the station
    can compute a homography from any camera frame to layout coordinates.
    """
    dict_id = cv2.aruco.DICT_4X4_50 if dict_id is None else dict_id
    ppu = design.print_ppu
    border = int(round(border_um * ppu))
    mk = int(round(marker_um * ppu))
    masks = render_masks(polys, design, ppu, pad_px=border)
    img = composite(masks, background=(255, 255, 255))
    h, w = img.shape[:2]
    cv2.rectangle(img, (border, border), (w - border, h - border), (0, 0, 0), max(1, int(ppu * 0.3)))
    adict = cv2.aruco.getPredefinedDictionary(dict_id)
    gap = (border - mk) // 2
    spots = {0: (gap, gap), 1: (w - gap - mk, gap), 2: (w - gap - mk, h - gap - mk), 3: (gap, h - gap - mk)}
    markers = {}
    for mid, (mx, my) in spots.items():
        tile = cv2.aruco.generateImageMarker(adict, mid, mk)
        img[my:my + mk, mx:mx + mk] = tile[..., None]
        markers[str(mid)] = [[mx, my], [mx + mk, my], [mx + mk, my + mk], [mx, my + mk]]
    label = f"{design.name}  |  Golden-Layout Inspector target  |  {design.window[2]-design.window[0]:.0f} x {design.window[3]-design.window[1]:.0f} um"
    cv2.putText(img, label, (border + mk // 4, h - gap // 3), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, ppu / 12), (60, 60, 60), max(1, int(ppu / 6)), cv2.LINE_AA)
    design.print_border_px = border
    design.marker_px = mk
    design.markers = markers
    cv2.imwrite(out_png, img)
    return img


def region_at(design: Design, rect_um) -> str:
    """Most consequential region overlapping rect_um ([x0,y0,x1,y1])."""
    x0, y0, x1, y1 = rect_um
    hits = set()
    for reg in design.regions:
        a = reg["rect"]
        if a[0] < x1 and x0 < a[2] and a[1] < y1 and y0 < a[3]:
            hits.add(reg["region"])
    m = design.macro_bbox
    inside_macro = m[0] < x1 and x0 < m[2] and m[1] < y1 and y0 < m[3]
    if not hits:
        return "ROUTING" if inside_macro else "OUTSIDE"
    for r in REGION_PRIORITY:
        if r in hits:
            return r
    return "ROUTING"
