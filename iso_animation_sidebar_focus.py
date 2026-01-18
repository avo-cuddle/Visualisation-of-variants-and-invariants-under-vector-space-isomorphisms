"""
iso_animation_sidebar_focus.py

Pygame animation to visualize what it *means* for a linear map T: R^2 -> R^2
to be a linear isomorphism (i.e., an invertible linear map). The left and middle
panels are two copies of R^2: a "domain" and its image under T.

We contrast:

- Linear-structure preservation (what makes T linear / an isomorphism):
  additivity, homogeneity, dimension (invertibility), dependence, spans/subspaces

vs

- Euclidean-geometry preservation (NOT guaranteed by an isomorphism):
  norms/lengths, dot products, angles/orthogonality, distances, unit circle shape

The visualization uses a time-varying invertible linear map:
    T(x) = A x
where A is built from parameters (rotation + shear + scaling),
interpolated in parameter space to avoid crossing det(A)=0.

Controls:
- SPACE : pause/unpause the map morph A(t) (vectors still move)
- R     : pick a new random target map
- O     : toggle orthogonal-only mode (rotations only => preserves Euclidean geometry)
- C     : toggle drawing the unit circle / ellipse
- 0..9  : focus modes (show one property + the instantaneous equation)
          press the same digit again to return to overview
- F11   : toggle fullscreen
- ESC   : quit
"""

import math
import random
import pygame

# ============================================================
# 2D LINEAR ALGEBRA HELPERS (no numpy, self-contained)
# ============================================================

# Keeping this file numpy-free makes it easier for classmates to run it anywhere.
# Also: in 2D, everything we need fits in a dozen formulas.

def mat_mul(A, v):
    """Multiply 2x2 matrix A by 2D vector v."""
    return (A[0][0] * v[0] + A[0][1] * v[1],
            A[1][0] * v[0] + A[1][1] * v[1])

def mat_mul2(A, B):
    """Multiply two 2x2 matrices A*B."""
    # Explicit 2x2 multiplication keeps things fast and readable.
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]

def mat_det(A):
    """Determinant of a 2x2 matrix. det(A) != 0 iff A is invertible."""
    # In 2D det(A) is also the signed area scale factor.
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]

def mat_transpose(A):
    """Transpose of a 2x2 matrix."""
    return [[A[0][0], A[1][0]],
            [A[0][1], A[1][1]]]

def mat_sub(A, B):
    """2x2 matrix subtraction A - B."""
    return [
        [A[0][0]-B[0][0], A[0][1]-B[0][1]],
        [A[1][0]-B[1][0], A[1][1]-B[1][1]],
    ]

def frob_norm(A):
    """Frobenius norm sqrt(sum of squares of entries)."""
    # Nice single number to quantify “how close to orthogonal” we are.
    return math.sqrt(A[0][0]**2 + A[0][1]**2 + A[1][0]**2 + A[1][1]**2)

def inv2(A):
    """Inverse of a 2x2 matrix. Returns None if singular (numerically)."""
    # We deliberately treat “almost singular” as singular to avoid ugly blow-ups.
    d = mat_det(A)
    if abs(d) < 1e-12:
        return None
    return [[ A[1][1]/d, -A[0][1]/d],
            [-A[1][0]/d,  A[0][0]/d]]

def dot(u, v):
    """Euclidean dot product."""
    return u[0]*v[0] + u[1]*v[1]

def norm(u):
    """Euclidean norm."""
    return math.sqrt(dot(u, u))

def angle_deg(u, v):
    """Angle between u and v in degrees."""
    # Clamp the cosine for numerical safety (acos gets cranky outside [-1,1]).
    nu, nv = norm(u), norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

# ============================================================
# MATRIX FACTORIES: rotation / shear / scale
# ============================================================

# The map A is built as a product R * Shear * Scale.
# This gives “nice looking” animations without det(A) accidentally crossing zero.

def rot(theta):
    """Rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s],
            [s,  c]]

def shear(kx, ky):
    """
    Shear-like matrix.
    det([[1,kx],[ky,1]]) = 1 - kx*ky, so keep kx*ky away from 1.
    """
    return [[1.0, kx],
            [ky,  1.0]]

def scale(sx, sy):
    """Scaling matrix with positive sx, sy."""
    return [[sx, 0.0],
            [0.0, sy]]

# ============================================================
# PARAMETERIZED ISOMORPHISMS
# ============================================================

class MapParams:
    """Parameters for A = R(theta) * Shear(kx,ky) * Scale(sx,sy)."""
    # Small “struct” object: easier than juggling 5 separate variables everywhere.
    def __init__(self, theta=0.0, kx=0.0, ky=0.0, sx=1.0, sy=1.0):
        self.theta = theta
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy

def build_matrix(p: MapParams):
    """Build A from parameters."""
    # Multiplication order matters. Rightmost factor acts first.
    return mat_mul2(rot(p.theta), mat_mul2(shear(p.kx, p.ky), scale(p.sx, p.sy)))

def interp_params(identity: MapParams, target: MapParams, alpha: float):
    """
    Interpolate between identity and target parameters.
    Scales use geometric interpolation so they stay positive.
    """
    # Linear interpolation is fine for angles/shears.
    # For scales, geometric interpolation avoids crossing through 0.
    def ginterp(a, b, t):
        return math.exp((1-t)*math.log(a) + t*math.log(b))

    out = MapParams()
    out.theta = (1-alpha)*identity.theta + alpha*target.theta
    out.kx = (1-alpha)*identity.kx + alpha*target.kx
    out.ky = (1-alpha)*identity.ky + alpha*target.ky
    out.sx = ginterp(identity.sx, target.sx, alpha)
    out.sy = ginterp(identity.sy, target.sy, alpha)
    return out

def random_invertible_params():
    """
    Random general invertible params.
    Ensure shear stays safely invertible: det(shear)=1-kx*ky away from 0.
    """
    # Wide-ish ranges, but not so wild that the picture flies off screen.
    theta = random.uniform(-math.pi, math.pi)
    kx = random.uniform(-1.0, 1.0)
    ky = random.uniform(-1.0, 1.0)

    # Keep 1 - kx*ky away from 0 (shear piece would be near-singular otherwise).
    if abs(kx*ky) > 0.65:
        kx *= 0.6
        ky *= 0.6

    # Positive scales, avoid extreme squashing/stretching.
    sx = random.uniform(0.6, 2.0)
    sy = random.uniform(0.6, 2.0)
    return MapParams(theta, kx, ky, sx, sy)

def random_orthogonal_params():
    """Orthogonal params = rotations only (preserve Euclidean dot products)."""
    # Rotation matrices have det=1 and preserve lengths/angles/dot products.
    theta = random.uniform(-math.pi, math.pi)
    return MapParams(theta, 0.0, 0.0, 1.0, 1.0)

def is_orthogonal(A):
    """Check A^T A ~ I in Frobenius norm."""
    # This is a quick numerical test; “close enough” is fine for visualization.
    AT = mat_transpose(A)
    ATA = mat_mul2(AT, A)
    I = [[1.0, 0.0],[0.0, 1.0]]
    err = frob_norm(mat_sub(ATA, I))
    return err, err < 1e-2

# ============================================================
# UNICODE-SAFE TEXT RENDERING (fixes tofu/squares in sidebar)
# ============================================================

# Pygame + fonts can be surprisingly fragile with math symbols on some systems.
# If a font doesn't cover a character, you get little squares ("tofu").
# The fallback is: replace fancy Unicode with plain ASCII that always renders.

def _pick_font_path(preferred_names, bold=False):
    """
    Try to find a real .ttf/.otf file on the system for one of the preferred fonts.
    pygame.font.match_font returns a path (or None).
    """
    for name in preferred_names:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return path
    return None

def load_ui_font(size, bold=False):
    """
    Load a font with good Unicode coverage for math-ish symbols.
    Returns (font_obj, unicode_ok_flag).
    """
    preferred = [
        # Windows
        "Segoe UI Symbol", "Segoe UI",
        # Linux
        "DejaVu Sans", "DejaVu Sans Mono", "Noto Sans", "Liberation Sans",
        # macOS / general
        "Arial Unicode MS", "Arial",
        # fallback monospace-ish options
        "Consolas", "Courier New",
    ]
    path = _pick_font_path(preferred, bold=bold)

    # Cheap heuristic: font name/path often hints whether it's a unicode-friendly family.
    unicode_ok = False
    if path:
        low = path.lower()
        unicode_ok = any(k in low for k in ["segoe", "dejavu", "noto", "arialuni", "symbol", "liberation"])

    if path:
        try:
            return pygame.font.Font(path, size), unicode_ok
        except Exception:
            # If it fails, we just fall back to the default system font.
            pass

    return pygame.font.SysFont(None, size, bold=bold), False

UNICODE_REPLACEMENTS = {
    # arrows/logic
    "⇔": "<=>",
    "⇒": "=>",
    "→": "->",
    "←": "<-",
    # comparisons
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    # greek / symbols
    "λ": "lambda",
    "θ": "theta",
    "°": "deg",
    "·": "*",
    "×": "x",
    "−": "-",
    "•": "-",
    "‖": "||",
    "∥": "||",
    "ℝ": "R",
    # dashes that often show as tofu
    "–": "-",
    "—": "-",
}

def sanitize_unicode(text: str) -> str:
    """Replace common math Unicode with ASCII so it never renders as tofu."""
    # Order doesn't matter here: replacements are all disjoint.
    for k, v in UNICODE_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text

def blit_text_safe(surf, font, text, pos, color=(235,235,235), unicode_ok=True):
    """
    Render text safely. If unicode_ok is False, sanitize text first.
    """
    if not unicode_ok:
        text = sanitize_unicode(text)
    surf.blit(font.render(text, True, color), pos)

# ============================================================
# DRAWING HELPERS
# ============================================================

# Coordinate system:
# - “world” is the math plane with y pointing up.
# - Pygame screen has y pointing down.
# So the conversion flips the sign in y.

def world_to_screen(panel_rect, world_pt, world_scale):
    """Convert (x,y) world coords to screen pixel coords inside a panel."""
    cx = panel_rect.centerx
    cy = panel_rect.centery
    x, y = world_pt
    return (int(cx + x * world_scale),
            int(cy - y * world_scale))

def draw_arrow(surf, panel_rect, world_scale, a, b, color, width=3):
    """Draw an arrow from world a to world b."""
    ax, ay = world_to_screen(panel_rect, a, world_scale)
    bx, by = world_to_screen(panel_rect, b, world_scale)
    pygame.draw.line(surf, color, (ax, ay), (bx, by), width)

    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return

    # Unit direction along the arrow + perpendicular for the head triangle.
    ux, uy = dx / L, dy / L
    px, py = -uy, ux

    head_len = 12
    head_w = 7
    tip = (bx, by)
    left = (bx - head_len*ux + head_w*px, by - head_len*uy + head_w*py)
    right = (bx - head_len*ux - head_w*px, by - head_len*uy - head_w*py)
    pygame.draw.polygon(surf, color, [tip, left, right])

def draw_axes_and_grid(surf, panel_rect, world_scale, A=None, grid_n=5,
                       color_axes=(80,80,90), color_grid=(45,45,55)):
    """
    Draw axes & grid. If A is provided, draw the image of the grid under A.
    """
    # The domain panel draws the regular grid.
    # The image panel draws the pushed-forward grid (so you “see” distortion).
    m = grid_n

    def transform(pt):
        return mat_mul(A, pt) if A is not None else pt

    for k in range(-m, m+1):
        # Vertical grid line x = k
        p1 = transform((k, -m))
        p2 = transform((k,  m))
        pygame.draw.line(surf, color_grid,
                         world_to_screen(panel_rect, p1, world_scale),
                         world_to_screen(panel_rect, p2, world_scale), 1)

        # Horizontal grid line y = k
        q1 = transform((-m, k))
        q2 = transform(( m, k))
        pygame.draw.line(surf, color_grid,
                         world_to_screen(panel_rect, q1, world_scale),
                         world_to_screen(panel_rect, q2, world_scale), 1)

    # Axes last so they sit on top.
    x1 = transform((-m, 0))
    x2 = transform(( m, 0))
    y1 = transform((0, -m))
    y2 = transform((0,  m))

    pygame.draw.line(surf, color_axes,
                     world_to_screen(panel_rect, x1, world_scale),
                     world_to_screen(panel_rect, x2, world_scale), 2)
    pygame.draw.line(surf, color_axes,
                     world_to_screen(panel_rect, y1, world_scale),
                     world_to_screen(panel_rect, y2, world_scale), 2)

def draw_curve(surf, panel_rect, world_scale, points, color, width=2):
    """Draw a polyline curve through world points."""
    pts = [world_to_screen(panel_rect, p, world_scale) for p in points]
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, width)

def draw_line_through_origin(surf, panel_rect, world_scale, direction, color, span=6.0, width=2):
    """Draw the line span{direction} through the origin."""
    d = direction
    nd = norm(d)
    if nd < 1e-8:
        return

    # Normalize so “span” behaves consistently for different directions.
    d = (d[0]/nd, d[1]/nd)
    p1 = (-span*d[0], -span*d[1])
    p2 = ( span*d[0],  span*d[1])
    pygame.draw.line(surf, color,
                     world_to_screen(panel_rect, p1, world_scale),
                     world_to_screen(panel_rect, p2, world_scale), width)

# ============================================================
# FORMATTING
# ============================================================

def fnum(x, places=4):
    """Signed fixed-point formatting with NaN/Inf handling."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    fmt = "{:+." + str(places) + "f}"
    return fmt.format(x)

# ============================================================
# FOCUS MODES
# ============================================================

# These keys match the digit controls (0..9). Keep them stable for screenshots.
FOCUS_TITLES = {
    1: "Additivity (linearity)",
    2: "Homogeneity (linearity)",
    3: "Invertibility & basis (dimension preserved)",
    4: "Linear dependence preserved",
    5: "Subspaces / spans preserved",
    6: "Norms / lengths (NOT preserved in general)",
    7: "Dot products (NOT preserved in general)",
    8: "Angles / orthogonality (NOT preserved in general)",
    9: "Distances (NOT preserved in general)",
    0: "Unit circle → ellipse (NOT preserved in general)",
}

# ============================================================
# LAYOUT
# ============================================================

def compute_layout(W, H):
    # Aim: 2 panels + sidebar without feeling cramped.
    margin = 16

    # Sidebar clamps: it's text-heavy, so give it room on wide screens.
    sidebar_w = max(520, min(680, int(0.34 * W)))

    usable_w = W - sidebar_w - 4*margin
    panel_w = max(260, usable_w // 2)
    panel_h = max(260, H - 2*margin)

    # Slightly different split for smaller windows.
    if W < 1200:
        sidebar_w = max(460, min(640, int(0.36 * W)))
        usable_w = W - sidebar_w - 4*margin
        panel_w = max(240, usable_w // 2)

    left = pygame.Rect(margin, margin, panel_w, panel_h)
    mid  = pygame.Rect(2*margin + panel_w, margin, panel_w, panel_h)
    side = pygame.Rect(3*margin + 2*panel_w, margin, sidebar_w, panel_h)

    return margin, sidebar_w, left, mid, side

# ============================================================
# MAIN
# ============================================================

def main():
    pygame.init()

    pygame.display.set_caption("Isomorphisms: linear structure vs Euclidean geometry (focus modes 0–9)")

    # Default window size: wide enough for the sidebar to breathe.
    W, H = 1780, 740
    windowed_size = (W, H)
    fullscreen = False

    # Start in a resizable window (fullscreen is optional).
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    # Unicode-safe fonts (with automatic ASCII fallback)
    font, ok1 = load_ui_font(18, bold=False)
    small, ok2 = load_ui_font(15, bold=False)
    big, ok3 = load_ui_font(22, bold=True)
    UNICODE_OK = (ok1 and ok2 and ok3)

    # Pixels per 1 world unit (tweak if you want a zoomed-in/out feel).
    world_scale = 58

    # Identity is our “home base” map; we morph between identity and target.
    identity = MapParams(0.0, 0.0, 0.0, 1.0, 1.0)
    target = random_invertible_params()

    orthogonal_mode = False
    pause_map = False
    show_circle = True
    focus_mode = None  # None => overview

    # Separate clocks: the map morph and the vector motion are independent.
    t_map = 0.0
    t_vec = 0.0
    speed_map = 0.6
    speed_vec = 1.0

    running = True
    while running:
        # dt in seconds (60 FPS target).
        dt = clock.tick(60) / 1000.0

        # Vectors always animate; the map can be paused.
        t_vec += dt * speed_vec
        if not pause_map:
            t_map += dt * speed_map

        # Smooth back-and-forth interpolation alpha in [0,1].
        alpha = 0.5 * (1.0 + math.sin(t_map))

        # Current transformation matrix A(t).
        p = interp_params(identity, target, alpha)
        A = build_matrix(p)

        # Two “live” vectors in the domain (just trigonometric motion).
        u_base = (2.2*math.cos(0.9*t_vec), 1.6*math.sin(1.1*t_vec))
        v_base = (-1.7*math.sin(0.7*t_vec + 0.6), 2.1*math.cos(0.6*t_vec + 0.2))

        # In dependence focus mode, force v to be a scalar multiple of u.
        if focus_mode == 4:
            s = 0.5 + 0.4*math.sin(1.2*t_vec)
            u = u_base
            v = (s*u[0], s*u[1])
        else:
            u, v = u_base, v_base

        # Push vectors through the map.
        Tu = mat_mul(A, u)
        Tv = mat_mul(A, v)

        # Lambda for the homogeneity check (changes over time).
        lam = 2.0 * math.sin(0.8*t_vec)

        # Two moving points for the distance focus.
        pnt = (2.0*math.cos(0.55*t_vec + 0.2), 1.6*math.sin(0.7*t_vec))
        qnt = (1.8*math.cos(0.62*t_vec + 2.1), 1.4*math.sin(0.8*t_vec + 1.0))
        Tp = mat_mul(A, pnt)
        Tq = mat_mul(A, qnt)

        # Diagnostics: invertibility and “how orthogonal” the matrix is.
        detA = mat_det(A)
        orth_err, is_orth = is_orthogonal(A)

        # Gram matrix G = A^T A drives dot products / norms / angles under the map.
        AT = mat_transpose(A)
        G = mat_mul2(AT, A)

        # Precompute a bunch of scalar values for the sidebar.
        nu, nv = norm(u), norm(v)
        nTu, nTv = norm(Tu), norm(Tv)
        duv = dot(u, v)
        dTuTv = dot(Tu, Tv)
        ang_uv = angle_deg(u, v)
        ang_T = angle_deg(Tu, Tv)

        # Additivity check: T(u+v) should match T(u)+T(v).
        Tsum = mat_mul(A, (u[0]+v[0], u[1]+v[1]))
        sum_images = (Tu[0]+Tv[0], Tu[1]+Tv[1])
        add_err = math.hypot(Tsum[0]-sum_images[0], Tsum[1]-sum_images[1])

        # Homogeneity check: T(lambda*u) should match lambda*T(u).
        Tlu = mat_mul(A, (lam*u[0], lam*u[1]))
        lTu = (lam*Tu[0], lam*Tu[1])
        hom_err = math.hypot(Tlu[0]-lTu[0], Tlu[1]-lTu[1])

        # Distances (Euclidean) are not generally preserved.
        d_pq = norm((pnt[0]-qnt[0], pnt[1]-qnt[1]))
        d_Tpq = norm((Tp[0]-Tq[0], Tp[1]-Tq[1]))

        # -------------------------
        # EVENTS
        # -------------------------
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            # Window resize only matters in windowed mode.
            elif e.type == pygame.VIDEORESIZE and not fullscreen:
                W, H = max(900, e.w), max(520, e.h)
                windowed_size = (W, H)
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_SPACE:
                    pause_map = not pause_map

                # New random target (orthogonal or general depending on mode).
                elif e.key == pygame.K_r:
                    target = random_orthogonal_params() if orthogonal_mode else random_invertible_params()

                elif e.key == pygame.K_c:
                    show_circle = not show_circle

                # Toggle orthogonal-only mode (useful for “geometry preserved” demo).
                elif e.key == pygame.K_o:
                    orthogonal_mode = not orthogonal_mode
                    target = random_orthogonal_params() if orthogonal_mode else random_invertible_params()

                # Fullscreen toggle (nice for lectures / demos).
                elif e.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        W, H = screen.get_size()
                    else:
                        W, H = windowed_size
                        screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

                # Digit keys pick focus modes; press again to return to overview.
                elif pygame.K_0 <= e.key <= pygame.K_9:
                    digit = e.key - pygame.K_0
                    focus_mode = None if focus_mode == digit else digit

        # -------------------------
        # LAYOUT
        # -------------------------
        margin, sidebar_w, left, mid, side = compute_layout(W, H)

        # -------------------------
        # DRAW
        # -------------------------
        screen.fill((14, 14, 17))

        # Rounded rectangles for the three panels.
        for rect in (left, mid, side):
            pygame.draw.rect(screen, (24, 24, 30), rect, border_radius=14)
            pygame.draw.rect(screen, (70, 70, 80), rect, 2, border_radius=14)

        # Headers.
        blit_text_safe(screen, big, "Domain (coordinates)  R^2", (left.x+14, left.y+10), unicode_ok=UNICODE_OK)
        blit_text_safe(screen, big, "Image under T(x)=Ax", (mid.x+14, mid.y+10), unicode_ok=UNICODE_OK)
        blit_text_safe(screen, big, "Isomorphism dashboard", (side.x+14, side.y+10), unicode_ok=UNICODE_OK)

        # Help text along the top.
        blit_text_safe(
            screen, small,
            "Controls: SPACE pause | R new map | O orthogonal | C circle | digits 0–9 focus | F11 fullscreen | ESC quit",
            (margin, max(2, margin-2)),
            (210,210,210),
            unicode_ok=UNICODE_OK
        )

        # Background grids in both panels.
        draw_axes_and_grid(screen, left, world_scale, A=None, grid_n=5)
        draw_axes_and_grid(screen, mid,  world_scale, A=A,  grid_n=5)

        # Standard basis vectors and their images (columns of A).
        e1 = (1.0, 0.0)
        e2 = (0.0, 1.0)
        Ae1 = mat_mul(A, e1)
        Ae2 = mat_mul(A, e2)

        basis_col = (230,230,140)
        draw_arrow(screen, left, world_scale, (0,0), e1, basis_col, 3)
        draw_arrow(screen, left, world_scale, (0,0), e2, basis_col, 3)
        draw_arrow(screen, mid,  world_scale, (0,0), Ae1, basis_col, 3)
        draw_arrow(screen, mid,  world_scale, (0,0), Ae2, basis_col, 3)

        # Unit circle in the domain becomes an ellipse under a general invertible map.
        draw_circle_now = show_circle or (focus_mode in (0, 6))
        if draw_circle_now:
            pts = []
            pts_img = []
            for i in range(260):
                th = 2*math.pi*i/260
                p0 = (math.cos(th), math.sin(th))
                pts.append(p0)
                pts_img.append(mat_mul(A, p0))
            draw_curve(screen, left, world_scale, pts, (120,120,140), 2)
            draw_curve(screen, mid,  world_scale, pts_img, (120,120,140), 2)

        # Colors for the two main vectors.
        col_u = (120, 200, 255)
        col_v = (255, 150, 205)
        col_sum = (160, 255, 160)

        # Most modes show u/v arrows; distance mode uses points instead.
        if focus_mode != 9:
            draw_arrow(screen, left, world_scale, (0,0), u, col_u, 4)
            draw_arrow(screen, left, world_scale, (0,0), v, col_v, 4)
            draw_arrow(screen, mid,  world_scale, (0,0), Tu, col_u, 4)
            draw_arrow(screen, mid,  world_scale, (0,0), Tv, col_v, 4)

        # Additivity demo: parallelogram-ish construction in both panels.
        if focus_mode == 1:
            draw_arrow(screen, left, world_scale, (0,0), (u[0]+v[0], u[1]+v[1]), col_sum, 4)
            draw_arrow(screen, left, world_scale, u, (u[0]+v[0], u[1]+v[1]), col_v, 2)
            draw_arrow(screen, left, world_scale, v, (u[0]+v[0], u[1]+v[1]), col_u, 2)

            draw_arrow(screen, mid, world_scale, (0,0), Tsum, col_sum, 4)
            draw_arrow(screen, mid, world_scale, Tu, sum_images, col_v, 2)
            draw_arrow(screen, mid, world_scale, Tv, sum_images, col_u, 2)

        # Homogeneity demo: show lambda*u and T(lambda*u).
        elif focus_mode == 2:
            draw_arrow(screen, left, world_scale, (0,0), (lam*u[0], lam*u[1]), (255,210,120), 4)
            draw_arrow(screen, mid,  world_scale, (0,0), Tlu, (255,210,120), 4)

        # Span demo: line through u maps to line through Tu.
        elif focus_mode == 5:
            draw_line_through_origin(screen, left, world_scale, u, (150,150,170), span=6.0, width=2)
            draw_line_through_origin(screen, mid,  world_scale, Tu, (150,150,170), span=6.0, width=2)

        # Distance demo: show two points and the segment between them.
        elif focus_mode == 9:
            ppx = world_to_screen(left, pnt, world_scale)
            pqx = world_to_screen(left, qnt, world_scale)
            pygame.draw.circle(screen, col_u, ppx, 6)
            pygame.draw.circle(screen, col_v, pqx, 6)
            pygame.draw.line(screen, (200,200,200), ppx, pqx, 2)

            ppy = world_to_screen(mid, Tp, world_scale)
            pqy = world_to_screen(mid, Tq, world_scale)
            pygame.draw.circle(screen, col_u, ppy, 6)
            pygame.draw.circle(screen, col_v, pqy, 6)
            pygame.draw.line(screen, (200,200,200), ppy, pqy, 2)

        # -------------------------
        # SIDEBAR TEXT
        # -------------------------
        a, b = A[0]
        c, d = A[1]

        # Display A and the explicit coordinate formula for T.
        eq1 = f"A = [[{a:+.3f}, {b:+.3f}], [{c:+.3f}, {d:+.3f}]]"
        eq2 = f"T(x,y) = ({a:+.3f}x + {b:+.3f}y , {c:+.3f}x + {d:+.3f}y)"
        blit_text_safe(screen, small, eq1, (side.x+14, side.y+44), (230,230,230), unicode_ok=UNICODE_OK)
        blit_text_safe(screen, small, eq2, (side.x+14, side.y+64), (230,230,230), unicode_ok=UNICODE_OK)

        # Status lines: whether map is paused and whether we’re in orthogonal mode.
        status = "PAUSED" if pause_map else "RUNNING"
        mode_txt = "ORTHOGONAL (preserves dot/norm/angle/dist)" if orthogonal_mode else "GENERAL INVERTIBLE (geometry can change)"
        blit_text_safe(screen, small, f"Map morph: {status}", (side.x+14, side.y+88),
                       (255,235,150) if pause_map else (150,255,150), unicode_ok=UNICODE_OK)
        blit_text_safe(screen, small, mode_txt, (side.x+14, side.y+108),
                       (200,200,255) if orthogonal_mode else (200,200,200), unicode_ok=UNICODE_OK)

        y = side.y + 138
        if focus_mode is None:
            blit_text_safe(screen, font, "Focus: overview (press 0–9 for one property)", (side.x+14, y),
                           (220,220,220), unicode_ok=UNICODE_OK)
        else:
            blit_text_safe(screen, font, f"Focus [{focus_mode}]: {FOCUS_TITLES[focus_mode]}", (side.x+14, y),
                           (255,255,200), unicode_ok=UNICODE_OK)
        y += 26

        focus_lines = []

        # The focus blocks below are intentionally “mathy” but still concrete.
        # They show an identity + a numerical check so students trust what they see.

        if focus_mode == 1:
            focus_lines += [
                "Key identity (linearity):",
                "  T(u+v) = T(u) + T(v)",
                f"Numeric check: ||T(u+v) - (T(u)+T(v))|| = {add_err:.2e}",
            ]
        elif focus_mode == 2:
            focus_lines += [
                "Key identity (linearity):",
                "  T(λu) = λT(u)",
                f"λ = {lam:+.3f}",
                f"Numeric check: ||T(λu) - λT(u)|| = {hom_err:.2e}",
            ]
        elif focus_mode == 3:
            focus_lines += [
                "Key fact (2D):",
                "  det(A) ≠ 0  ⇔  A invertible  ⇔  {Ae1,Ae2} independent",
                f"det(A) = {detA:+.6f}",
                "Interpretation: area(Ae1,Ae2) = |det(A)|.",
            ]
        elif focus_mode == 4:
            focus_lines += [
                "Key identity (2D determinants):",
                "  det[Tu,Tv] = det(A)·det[u,v]",
                f"det[u,v] (signed area) ≈ {(u[0]*v[1]-u[1]*v[0]):+.3e}",
                f"det[Tu,Tv] (signed area) ≈ {(Tu[0]*Tv[1]-Tu[1]*Tv[0]):+.3e}",
                f"det(A) = {detA:+.6f}",
                "So: det[u,v]=0 ⇔ det[Tu,Tv]=0 (dependence preserved).",
            ]
        elif focus_mode == 5:
            focus_lines += [
                "Key identity (subspaces):",
                "  T(span{u}) = span{T(u)}",
                "Visual: the line through u maps to the line through Tu.",
            ]
        elif focus_mode == 6:
            g11, g12 = G[0]
            g21, g22 = G[1]
            uGu = u[0]*(g11*u[0]+g12*u[1]) + u[1]*(g21*u[0]+g22*u[1])
            focus_lines += [
                "Norm distortion (Euclidean):",
                "  ||T(u)||^2 = u^T (A^T A) u",
                f"G = A^T A = [[{g11:+.3f},{g12:+.3f}],[{g21:+.3f},{g22:+.3f}]]",
                f"||u|| = {fnum(nu,4)}   ||Tu|| = {fnum(nTu,4)}",
                f"Check: ||Tu||^2 = {fnum(nTu*nTu,4)}  and  u^TGu = {fnum(uGu,4)}",
                "Preserved iff A^T A = I (orthogonal).",
            ]
        elif focus_mode == 7:
            g11, g12 = G[0]
            g21, g22 = G[1]
            uGv = u[0]*(g11*v[0]+g12*v[1]) + u[1]*(g21*v[0]+g22*v[1])
            focus_lines += [
                "Dot-product distortion:",
                "  <Tu,Tv> = u^T (A^T A) v",
                f"<u,v> = {fnum(duv,4)}   <Tu,Tv> = {fnum(dTuTv,4)}",
                f"u^T(A^T A)v = {fnum(uGv,4)} (matches <Tu,Tv>)",
                "Preserved iff A^T A = I (orthogonal).",
            ]
        elif focus_mode == 8:
            g11, g12 = G[0]
            g21, g22 = G[1]
            uGu = u[0]*(g11*u[0]+g12*u[1]) + u[1]*(g21*u[0]+g22*u[1])
            vGv = v[0]*(g11*v[0]+g12*v[1]) + v[1]*(g21*v[0]+g22*v[1])
            uGv = u[0]*(g11*v[0]+g12*v[1]) + u[1]*(g21*v[0]+g22*v[1])
            denom = math.sqrt(max(1e-18, uGu*vGv))
            cos_th = max(-1.0, min(1.0, uGv / denom))
            th_prime = math.degrees(math.acos(cos_th))
            focus_lines += [
                "Angle distortion:",
                "  cos(θ') = (u^T G v) / (sqrt(u^T G u) sqrt(v^T G v)),  G=A^T A",
                f"angle(u,v) = {ang_uv:6.2f}°   angle(Tu,Tv) = {ang_T:6.2f}°",
                f"computed θ' from formula = {th_prime:6.2f}°",
                "Preserved iff A is orthogonal.",
            ]
        elif focus_mode == 9:
            g11, g12 = G[0]
            g21, g22 = G[1]
            w = (pnt[0]-qnt[0], pnt[1]-qnt[1])
            wGw = w[0]*(g11*w[0]+g12*w[1]) + w[1]*(g21*w[0]+g22*w[1])
            focus_lines += [
                "Distance distortion:",
                "  ||T(p)-T(q)||^2 = (p-q)^T (A^T A) (p-q)",
                f"||p-q|| = {fnum(d_pq,4)}   ||T(p)-T(q)|| = {fnum(d_Tpq,4)}",
                f"Check: ||T(p)-T(q)||^2 = {fnum(d_Tpq*d_Tpq,4)}  and  w^T G w = {fnum(wGw,4)}",
                "Preserved iff A is orthogonal.",
            ]
        elif focus_mode == 0:
            Ainv = inv2(A)
            if Ainv is not None:
                AinvT = mat_transpose(Ainv)
                Q = mat_mul2(AinvT, Ainv)  # A^{-T} A^{-1}
                q11, q12 = Q[0]
                q21, q22 = Q[1]
                focus_lines += [
                    "Image of the unit circle under y = A x:",
                    "  y(θ) = A (cosθ, sinθ)   (ellipse unless orthogonal)",
                    "Implicit form in y:",
                    "  y^T (A^{-T}A^{-1}) y = 1",
                    f"A^{{-T}}A^{{-1}} ≈ [[{q11:+.3f},{q12:+.3f}],[{q21:+.3f},{q22:+.3f}]]",
                    "If A is orthogonal, then A^{-T}A^{-1}=I → still a circle.",
                ]
            else:
                focus_lines += [
                    "Unit circle → ellipse focus:",
                    "  y(θ) = A (cosθ, sinθ)",
                    "Inverse not available (A nearly singular).",
                ]

        for line in focus_lines:
            blit_text_safe(screen, small, line, (side.x+14, y), (235,235,235), unicode_ok=UNICODE_OK)
            y += 18

        # Divider line.
        y += 8
        pygame.draw.line(screen, (80,80,90), (side.x+14, y), (side.right-14, y), 1)
        y += 10

        # Invariants list (always shown).
        blit_text_safe(screen, font, "Linear-structure preserved by an isomorphism:", (side.x+14, y),
                       (160,255,160), unicode_ok=UNICODE_OK)
        y += 24
        inv_lines = [
            "• [1] T(u+v)=T(u)+T(v)",
            "• [2] T(λu)=λT(u)",
            "• [3] invertible ⇒ basis ↦ basis (dimension)",
            "• [4] linear dependence preserved",
            "• [5] subspaces/spans ↦ subspaces/spans",
        ]
        for s in inv_lines:
            col = (255,255,200) if (focus_mode is not None and s.startswith(f"• [{focus_mode}]")) else (160,255,160)
            blit_text_safe(screen, small, s, (side.x+24, y), col, unicode_ok=UNICODE_OK)
            y += 18

        y += 8

        # Non-invariants list (always shown).
        blit_text_safe(screen, font, "Euclidean geometry NOT preserved in general:", (side.x+14, y),
                       (255,180,180), unicode_ok=UNICODE_OK)
        y += 24
        var_lines = [
            "• [6] norms/lengths",
            "• [7] dot products",
            "• [8] angles / orthogonality",
            "• [9] distances",
            "• [0] unit circle stays a circle",
        ]
        for s in var_lines:
            col = (255,255,200) if (focus_mode is not None and s.startswith(f"• [{focus_mode}]")) else (255,180,180)
            blit_text_safe(screen, small, s, (side.x+24, y), col, unicode_ok=UNICODE_OK)
            y += 18

        # Bottom readout: det + orthogonality error.
        y += 10
        blit_text_safe(
            screen, small,
            f"det(A)={fnum(detA,4)}   ||A^T A - I||_F={orth_err:.2e}",
            (side.x+14, y),
            (160,255,160) if is_orth else (255,180,180),
            unicode_ok=UNICODE_OK
        )

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    # Standard entry point so you can run this file directly.
    main()
