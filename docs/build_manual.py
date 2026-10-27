"""
Builds the student user manual, docs/Wind_Tunnel_Logger_User_Manual.pdf: first takes fresh screenshots from the
real program windows (docs/img), then lays out the PDF.

    .venv\\Scripts\\python.exe -m pip install reportlab pypdfium2
    .venv\\Scripts\\python.exe docs\\build_manual.py                  # screenshots + PDF
    .venv\\Scripts\\python.exe docs\\build_manual.py --no-screenshots # PDF from the existing images
"""
import math
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

DOCS = Path(__file__).resolve().parent
IMG = DOCS / "img"
OUT = DOCS / "Wind_Tunnel_Logger_User_Manual.pdf"
sys.path.insert(0, str(DOCS.parent / "src"))
from wind_tunnel_logger import __version__ as VERSION  # noqa: E402

RELEASES = "https://github.com/Dkaka/COMP0219_Tutorials/releases"
TAG = f"wind-tunnel-logger-v{VERSION}"      # the release tag: the repo also holds the tutorials
WHEEL_FILE = f"wind_tunnel_logger-{VERSION}-py3-none-any.whl"

# ---------------------------------------------------------------- fonts and styles
FONTS = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "segoeuii.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(FONTS / "segoeuiz.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-BoldItalic")

BLUE = colors.HexColor("#0b5d7a")
CYAN = colors.HexColor("#00a8b8")
GREY = colors.HexColor("#5f6368")
LIGHT = colors.HexColor("#f2f5f7")

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=14, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.5, leading=11, textColor=GREY),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=8.5, leading=11, textColor=GREY,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "H1": ParagraphStyle("H1", fontName="Body-Bold", fontSize=17, leading=21, textColor=BLUE, spaceBefore=4,
                         spaceAfter=8),
    "H2": ParagraphStyle("H2", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10,
                         spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=12),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=9, leading=12),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.5, leading=11),
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=30, leading=36, textColor=BLUE),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=14, leading=19, textColor=GREY),
    "toc1": ParagraphStyle("toc1", fontName="Body", fontSize=10.5, leading=16, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Body", fontSize=9.5, leading=13, leftIndent=14, textColor=GREY),
}
WIDTH = A4[0] - 2 * 20 * mm


# ---------------------------------------------------------------- helpers
def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return Paragraph(text, S["H1"])


def H2(text):
    return Paragraph(text, S["H2"])


def bullets(items, numbered=False):
    return ListFlowable([ListItem(P(t), leftIndent=14) for t in items],
                        bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                        bulletFontName="Body", bulletFontSize=10, leftIndent=14, spaceAfter=6)


def callout(kind, text):
    tint = {"IMPORTANT": ("#fdecea", "#c62828"), "TIP": ("#e8f4ea", "#2e7d32"), "WHY": ("#e8f1f8", "#0b5d7a")}[kind]
    t = Table([[P(f"<b>{kind}</b>", "cellb"), P(text, "cell")]], colWidths=[28 * mm, WIDTH - 28 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint[0])),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(tint[1])),
        ("TEXTCOLOR", (0, 0), (0, 0), colors.HexColor(tint[1])),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 8)])


def code(text, wrap=False):
    """Code box; wrap=True breaks a long line after '/' (for URLs) to fit the page."""
    pre = (Preformatted(text, S["code"], maxLineLength=88, splitChars="/", newLineChars="    ") if wrap
           else Preformatted(text, S["code"]))
    t = Table([[pre]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return KeepTogether([t, Spacer(1, 8)])


FIG = [0]


def cropped(name, right):
    """Left part (0..right of the width) of a full-window screenshot, so its text is larger on the page."""
    from PIL import Image as PILImage
    out = IMG / f"crop_{name}"
    with PILImage.open(IMG / name) as im:
        im.crop((0, 0, int(im.width * right), im.height)).save(out)
    return out.name


def figure(name, caption, width=WIDTH):
    from reportlab.lib.utils import ImageReader
    path = IMG / name
    iw, ih = ImageReader(str(path)).getSize()
    w = min(width, WIDTH)
    img = Image(str(path), width=w, height=w * ih / iw)
    img.hAlign = "CENTER"
    FIG[0] += 1
    frame = Table([[img]], colWidths=[w + 2])
    frame.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                               ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                               ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return KeepTogether([frame, P(f"Figure {FIG[0]}: {caption}", "caption")])


def photos(items, caption):
    """Photos side by side, each with its own label, under one figure number. items: [(file, label)]."""
    from reportlab.lib.utils import ImageReader
    w = (WIDTH - 6 * (len(items) - 1)) / len(items)
    cells = []
    for name, label in items:
        iw, ih = ImageReader(str(IMG / name)).getSize()
        cells.append([Image(str(IMG / name), width=w - 2, height=(w - 2) * ih / iw), P(label, "caption")])
    t = Table([cells], colWidths=[w] * len(items))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
    FIG[0] += 1
    return KeepTogether([t, P(f"Figure {FIG[0]}: {caption}", "caption")])


def table(rows, widths, header=True):
    data = [[P(c, "cellb" if header and i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 10)])


# ---------------------------------------------------------------- document with TOC + bookmarks
class Manual(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                         bottomMargin=18 * mm, title="Wind Tunnel Logger - Student User Manual",
                         author="Shengyang Zhuang", subject="COMP0219 2026/2027 wind tunnel setup")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame]),
                               PageTemplate("normal", [frame], onPage=self._footer)])
        self._n = 0

    def beforeDocument(self):
        self._n = 0                 # multiBuild lays the document out several times: same keys every pass

    def _footer(self, canv, doc):
        canv.saveState()
        canv.setFont("Body", 8)
        canv.setFillColor(GREY)
        canv.drawString(20 * mm, 10 * mm, f"Wind Tunnel Logger {VERSION}  -  Student User Manual  -  COMP0219 2026/2027")
        canv.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            key = f"h{self._n}"
            self._n += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


# ---------------------------------------------------------------- content
def zero_table():
    rho, offset = 1.204, 2.5
    rows = [["True speed", "Pressure the pitot sees", f"Reading with a {offset} Pa offset left in", "Error"]]
    for v in (2.0, 4.5, 10.0):
        dp = 0.5 * rho * v * v
        vr = math.sqrt(2 * (dp + offset) / rho)
        rows.append([f"{v:g} m/s", f"{dp:.1f} Pa", f"{vr:.2f} m/s", f"+{(vr / v - 1) * 100:.0f} %"])
    return rows


def story():
    s = []
    # cover
    logo = Image(str(IMG / "logo.png"), width=26 * mm, height=26 * mm)
    logo.hAlign = "CENTER"
    s += [Spacer(1, 18 * mm), logo, Spacer(1, 6 * mm), P("Wind Tunnel Logger", "title"), Spacer(1, 4),
          P("Student user manual", "subtitle"),
          P("COMP0219 Sensors and Signal Conditioning - 2026/2027 wind tunnel setup", "subtitle"),
          Spacer(1, 10 * mm), figure("main_window.png", "Wind Tunnel Logger during a tunnel run (example data)."),
          Spacer(1, 8 * mm),
          P(f"Version {VERSION}  -  Author: Shengyang Zhuang  -  October 2026", "small"),
          P("Live plot and logging of your airspeed sensor against the wind tunnel's own airspeed reading.", "small")]
    s.append(PageBreak())

    # contents
    from reportlab.platypus.doctemplate import NextPageTemplate
    s.insert(0, NextPageTemplate("normal"))
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    s += [P("<b>Contents</b>", "subtitle"), Spacer(1, 6), toc, PageBreak()]

    # 1 overview
    s += [H1("1. What Wind Tunnel Logger does"),
          P("Wind Tunnel Logger shows your sensor's airspeed next to the wind tunnel's own airspeed reading, live, "
            "and saves both to CSV files. The tunnel's reading is the <b>ground truth</b>. The lab has two tunnels, "
            "and each has its own ground truth, which you plug into your laptop by USB:"),
          photos([("vdas_tunnel.jpg", "<b>White tunnel</b> (TecQuipment): its pressure sensor, through its "
                   "<i>VDAS</i> box. Pick <b>Ground truth: Wind tunnel</b>."),
                  ("wood_tunnel.jpg", "<b>Wood tunnel</b>: a GIGA pitot tube (Arduino GIGA R1 on top, its probe in "
                   "the tunnel). Pick <b>Ground truth: GIGA pitot tube</b>.")],
                 "The two tunnels and their ground truth."),
          figure("main_window.png", "The main window during a run (example data)."),
          P("The window, from top to bottom:"),
          bullets(["<b>Toolbar</b>: Start Logging, Re-zero tunnel, Plot run, Open logs folder, Help, and on the right "
                   "<b>Your board</b> (your sensor's port) with Refresh and Connect.",
                   "<b>Settings row</b>: Ground truth (Wind tunnel, Simulated, or a GIGA pitot tube while one is "
                   "plugged in, see section 6), your Team name, today's Air pressure and Air temperature.",
                   "<b>Readouts</b>: the tunnel's speed (cyan) and your sensor's speed (magenta). Under your reading: your "
                   "port and the <b>rate</b> your board achieves (it must reach 200 Hz).",
                   "<b>Status lines</b>: what the program is doing and anything that needs your attention, in red.",
                   "<b>Live plot</b>: the last 15 s. The shaded cyan band is the range the true speed can be in, "
                   "given the tunnel sensor's resolution of 1 Pa; it is wide at low speed (see section 7)."]),
          H2("What your board must do"),
          bullets(["Appear as a USB serial port on the laptop (a NUCLEO does: <i>STMicroelectronics STLink Virtual "
                   "COM Port</i>).",
                   "Print <b>one speed in m/s per line</b>, as a plain number, for example <font name='Mono'>5.432</font>.",
                   "Stream at <b>200 Hz</b>, at <b>115200 baud</b>."]),
          code('/* in your 200 Hz loop, once per estimate (v in m/s) */\nprintf("%.3f\\r\\n", v);'),
          PageBreak()]

    # 2 what you need
    s += [H1("2. What you need"),
          table([["Item", "Why"],
                 ["A laptop", "Windows 10 or 11 (tested). macOS and Linux should also work with the pip version; see "
                  "<i>Using a Mac or Linux</i> in section 6."],
                 ["Internet, once", "To download Wind Tunnel Logger."],
                 ["Your board and its USB cable", "A data cable, not a charge-only cable."],
                 ["White tunnel: its VDAS box's USB cable", "USB-B (printer type) at the box end; it stays at the "
                  "tunnel."],
                 ["Wood tunnel: the GIGA's USB cable", "USB-C at the GIGA end; it stays at the tunnel."]],
                [0.33, 0.67]),
          callout("TIP", "You <b>don't need TecQuipment's VDAS software</b>: Wind Tunnel Logger talks to the "
                         "white tunnel's VDAS box, and to the wood tunnel's GIGA, directly. If the VDAS software is "
                         "open on your laptop, close it.")]

    # 3 install WTL
    s += [H1("3. Install Wind Tunnel Logger"),
          P(f"Both versions are on the course GitHub <b>Releases</b> page, under <i>Assets</i> of the latest "
            f"release:<br/><font name='Mono' size='8.5'>{RELEASES}</font>"),
          H2("Option A - Windows program (recommended, no Python needed)"),
          bullets(["Download <b>WindTunnelLogger.exe</b> and move it somewhere you'll find it again, for example "
                   "your Desktop or <i>Documents</i>.",
                   "Double-click it. The window takes a few seconds to appear: the program unpacks itself each time "
                   "it starts.",
                   "If Windows shows <i>\"Windows protected your PC\"</i>, click <b>More info</b>, then "
                   "<b>Run anyway</b>. This appears because the program isn't code-signed; it is safe to run."],
                  numbered=True),
          callout("TIP", "If your antivirus moves the file to quarantine, restore it or use Option B. Programs "
                         "packaged with Python are sometimes flagged by mistake."),
          H2("Option B - pip install (Windows, macOS or Linux, Python 3.9 or newer)"),
          P("In a terminal (PowerShell on Windows, Terminal on a Mac), run:"),
          code(f"python -m pip install {RELEASES}/download/{TAG}/{WHEEL_FILE}", wrap=True),
          P("Type it as <b>one line</b> (it is only wrapped here to fit the page). Or download "
            f"<font name='Mono'>{WHEEL_FILE}</font> from the Releases page and run "
            f"<font name='Mono'>python -m pip install {WHEEL_FILE}</font> in the folder you saved it to. On macOS "
            "or Linux, type <font name='Mono'>python3</font> instead of <font name='Mono'>python</font>. Then start "
            "the program with:"),
          code("wind-tunnel-logger"),
          P("If that command isn't found, use <font name='Mono'>python -m wind_tunnel_logger</font>. To update "
            "later, install the newer <font name='Mono'>.whl</font> file the same way."),
          PageBreak()]

    # 4 board
    s += [H1("4. Prepare your board"),
          bullets(["Flash your firmware so the board prints one speed in m/s per line at 200 Hz, 115200 baud "
                   "(section 1).",
                   "Close every other program that might use the board's port: PuTTY, CoolTerm, the STM32CubeIDE "
                   "serial console, an Arduino serial monitor. Only one program can open a port at a time.",
                   "Find the port: in Windows, open <b>Device Manager</b>, then <b>Ports (COM &amp; LPT)</b>; a NUCLEO "
                   "appears as <i>STMicroelectronics STLink Virtual COM Port (COMx)</i>. On a Mac it is "
                   "<font name='Mono'>/dev/tty.usbmodem...</font>.",
                   "Test at home first with the simulated ground truth (section 10): you'll see straight away whether "
                   "your board connects, sends numbers and reaches 200 Hz."]),
          ]

    # 5 zeroing
    s += [H1("5. Zero the wind tunnel (every session)"),
          callout("WHY", "A pressure sensor never reads exactly 0 Pa in still air: it has a small <b>offset</b>, "
                         "here a few pascals. Airspeed comes from the pressure the pitot tube sees, "
                         "v = sqrt(2 &#916;p / &#961;), and at low speed that pressure is tiny. At 2 m/s it is only "
                         "about 2.4 Pa, the same size as the offset. Left in, the offset makes low speeds read far too "
                         "high (table below). Zeroing measures the offset with the fan off and removes it from every "
                         "reading."),
          table(zero_table(), [0.2, 0.27, 0.35, 0.18]),
          P("On the <b>white tunnel</b> there are two zeros: the tunnel's own pressure display (step A, at the start "
            "of a session) and Wind Tunnel Logger's software zero (step B, automatic every time you start it). On "
            "the <b>wood tunnel</b>, skip both: the GIGA zeroes itself when you pick it (section 6, <i>On the wood "
            "tunnel</i>), also with the fan off."),
          H2("Step A - zero the tunnel's pressure display (start of a session)"),
          P("Skip this step if your TA has already done it today."),
          bullets(["Switch on the tunnel's instrument power and <b>wait 10-15 minutes</b>: the pressure sensor drifts "
                   "while it warms up.",
                   "Make sure the <b>fan is OFF and has stopped</b>, nothing is blowing at the tunnel, and wait "
                   "about 30 s for the air to settle.",
                   "On the tunnel's pressure display module (the small AFA5 module whose LCD shows the pressure), <b>press and hold the "
                   "zero button for at least 4 seconds</b>. The display should then read 0 or very close to it."],
                  numbered=True),
          callout("IMPORTANT", "Never hold the zero button <b>while switching the power on</b>: that starts the "
                               "module's calibration mode (the display shows <i>- - - -</i> or <i>CAL</i>). If it "
                               "happens, switch the power off again without pressing anything else: the calibration "
                               "is then left unchanged."),
          H2("Step B - the software zero in Wind Tunnel Logger (automatic)"),
          P("As soon as the tunnel's data arrives after the program starts (and again whenever you switch "
            "Ground truth back to Wind tunnel), it averages 5 s of tunnel readings with the fan off and removes "
            "that offset from then on. Until the data arrives, the status line shows <i>Waiting for the "
            "tunnel's data to zero it</i>. Keep the <b>fan OFF</b> until the status line stops showing "
            "<i>Zeroing the tunnel reading</i>:"),
          figure(cropped("zeroing.png", 0.6), "Zeroing in progress: keep the fan off."),
          P("When it finishes, the status line shows <b>tunnel offset</b>, normally between about -3 and +3 Pa. "
            "If the fan was running during the zero, the program can tell (the reading is far from 0), uses a "
            "typical offset of +3 Pa instead, and warns you in red:"),
          figure(cropped("zero_fan_running.png", 0.6), "The zero was taken with the fan running. Stop the fan and press "
                                         "Re-zero tunnel."),
          P("<b>Press Re-zero tunnel (fan OFF)</b>, with the fan stopped and the air still, whenever:"),
          bullets(["you see that red warning;",
                   "the tunnel's instrument power was switched off and on, or you did step A again;",
                   "with the fan off, the tunnel reading isn't close to 0 m/s (up to about 1 m/s is normal, see "
                   "section 7);",
                   "before an important logged run, if you can stop the fan."]),
          callout("TIP", "If your own sensor needs a zero too, do it at the same moment, with the fan off."),
          PageBreak()]

    # 6 at the tunnel
    s += [H1("6. At the tunnel, step by step"),
          bullets(["<b>Fan OFF.</b> Plug the tunnel's ground truth into your laptop by USB: the <b>VDAS box</b> on the "
                   "white tunnel, the <b>GIGA</b> on the wood tunnel (section 1).",
                   "Start Wind Tunnel Logger and pick <b>Ground truth</b>: <b>Wind tunnel</b> on the white tunnel, "
                   "<b>GIGA pitot tube (COMx)</b> on the wood tunnel.",
                   "Within a few seconds Wind Tunnel Logger receives the tunnel's data: the status line changes from "
                   "<i>Waiting for the tunnel's data</i> (or <i>the GIGA's data</i>) to <i>Zeroing ...</i>. Nothing "
                   "else to open: if TecQuipment's VDAS software is running, close it.",
                   "Wait for the zero to finish (section 5).",
                   "Fill in <b>Team</b> (used in your file names), today's <b>Air pressure</b> in hPa (from a lab "
                   "barometer or a weather service's surface pressure) and the <b>Air temperature</b> in °C. These "
                   "set the air density used to turn the tunnel's pressure into speed.",
                   "Plug in your board, click <b>Refresh</b>, pick its port under <b>Your board</b> and click "
                   "<b>Connect</b>. Your reading starts to update; check the rate under it is close to 200 Hz and "
                   "not red.",
                   "Start the fan at your first speed and let it settle.",
                   "Click <b>Start Logging</b>. Run your test: the assessed range is 2-10 m/s, including steps and "
                   "ramps.",
                   "Click <b>Stop &amp; Save</b>. Accept the suggested file name (team, date and time) or type your "
                   "own, and click Save.",
                   "When you have finished, close Wind Tunnel Logger."],
                  numbered=True),
          P("Until the tunnel reading arrives, the status line says what is missing, for example when the VDAS "
            "box isn't plugged in or the tunnel's instrument power is off:"),
          figure(cropped("no_box.png", 0.67), "No tunnel data yet: the status line says why."),
          H2("On the wood tunnel (GIGA pitot tube)"),
          P("The wood tunnel's ground truth is a WindSpeedGiga pitot tube: an Arduino GIGA R1 whose pitot probe sits "
            "in the tunnel. While it is plugged in, <b>Ground truth</b> also offers <b>GIGA pitot tube (COMx)</b>, "
            "and its port is no longer listed under <b>Your board</b>. Keep the fan OFF when you pick it:"),
          bullets(["The program sends the GIGA its calibration (<b>LEQ</b> 102.177 m and <b>C</b> 1, from "
                   "<font name='Mono'>config.yaml</font>) with today's air pressure and temperature. The GIGA forgets "
                   "LEQ whenever it loses power, and then reads 50-60 % low: so the program uses none of its readings "
                   "until the GIGA confirms these values, and sends them again whenever it restarts.",
                   "The GIGA zeroes itself on the board in about 2 s (<b>Re-zero GIGA</b> repeats it). The fan must "
                   "be off, or the GIGA's two tubes pushed into each other.",
                   "Its calibration is within about &#177;3 % at 8-14 m/s and was only checked from 4 to 14 m/s. "
                   "Above its sensor's range (125 Pa, about 14 m/s) its readings aren't used.",
                   "Saved runs end in <font name='Mono'>_giga</font>."]),
          H2("Using a Mac or Linux"),
          P("Install with pip (section 3, option B). Wind Tunnel Logger reads the VDAS box on macOS and Linux "
            "too, but it has only been tested on Windows: try it at home in simulated mode (section 10), and at "
            "the tunnel. On Linux, reading USB devices needs a permission rule (section 11). If it doesn't "
            "work, plug your board into a teammate's Windows laptop, run Wind Tunnel Logger there and copy the "
            "saved files afterwards."),
          figure(cropped("logging_on.png", 0.6), "Logging: the button now reads Stop & Save."),
          figure("saved_dialog.png", "After Stop & Save: the three files of the run. Plot this run opens the plots "
                                     "of section 8.", width=100 * mm),
          PageBreak()]

    # 7 reading the screen
    s += [H1("7. Reading the screen"),
          table([["What you see", "Meaning"],
                 ["<b>Tunnel ground truth</b> m/s", "The tunnel's airspeed, from its pressure reading after the zero, "
                  "using today's air density."],
                 ["dp ... Pa (raw ..., offset ...)", "The pressure behind that speed, the raw value from the VDAS box, "
                  "and the zero offset being removed."],
                 ["<b>Your sensor</b> m/s", "The last value your board sent."],
                 ["COM3   199.2 Hz (target 200)", "Your port and the average rate over the last 5 s. <font color='#c62828'>"
                  "Red</font> if it's below 196 Hz, or if lines arrive that aren't numbers (the last bad line is "
                  "shown)."],
                 ["last 10 s mean ... difference", "Both speeds averaged over 10 s, and how far your sensor is from the "
                  "tunnel."],
                 ["Shaded cyan band", "Where the true speed can be, given the tunnel sensor reports whole pascals "
                  "(&#177;1 Pa). It is wide at low speed because a 1 Pa step is a big change in speed there: about "
                  "&#177;0.8 m/s at 1 m/s, &#177;0.2 m/s at 4.5 m/s and &#177;0.08 m/s at 10 m/s."],
                 ["<font color='#c62828'>Red status lines</font>", "Something needs your attention: no data, fan "
                  "running during the zero, your board not sending."]],
                [0.32, 0.68]),
          callout("TIP", "With the fan off, the tunnel may show up to about 1 m/s instead of 0. That is normal: "
                         "a reading of 1 Pa already corresponds to about 1.3 m/s. The same limit is why the "
                         "assessment starts at 2 m/s."),
          ]

    # 8 plot run
    s += [H1("8. Looking at a saved run"),
          P("<b>Plot this run</b> (after saving) or <b>Plot run...</b> (any time; pick the run's "
            "<font name='Mono'>_aligned.csv</font>) opens three plots:"),
          figure("plot_run.png", "A saved run (example data)."),
          bullets(["<b>Speed over time</b>: every sample of your sensor (magenta) and every tunnel reading (cyan dots, "
                   "a few per second).",
                   "<b>Your sensor vs tunnel</b>: one dot per tunnel reading, paired with your nearest sample. On the "
                   "dashed 1:1 line means exactly right.",
                   "<b>Error vs tunnel</b>: your sensor minus the tunnel, against speed. It shows at which speeds your "
                   "sensor reads high or low."])]

    # 9 files
    s += [H1("9. Your saved files"),
          P("Every run is saved as three CSV files in <font name='Mono'>Documents\\Wind Tunnel Logger\\logs</font> "
            "(click <b>Open logs folder</b>):"),
          table([["File", "Contents"],
                 ["<font name='Mono'>team3_20261022_143005_vdas.csv</font>", "Every tunnel reading: time, speed, "
                  "raw and zero-corrected pressure."],
                 ["<font name='Mono'>team3_20261022_143005_student.csv</font>", "Every sample from your board, with "
                  "the time it arrived."],
                 ["<font name='Mono'>team3_20261022_143005_aligned.csv</font>", "Each tunnel reading paired with your "
                  "nearest sample (columns <i>ground_truth_mps</i> and <i>student_mps</i>). Use this one to work out "
                  "your error statistics (MSE, bias, standard deviation) for the report."]],
                [0.45, 0.55]),
          P("Runs made with the simulated ground truth end in <font name='Mono'>_simulated</font>, and runs with a "
            "GIGA pitot tube as ground truth in <font name='Mono'>_giga</font>, so they can't be mistaken for tunnel "
            "data. Keep your files: the module team uses the same format for assessment."),
          ]

    # 10 bench test
    s += [H1("10. Bench test at home (simulated ground truth)"),
          P("No tunnel needed. Set <b>Ground truth</b> to <b>Simulated (bench test)</b>: the program generates a "
            "made-up tunnel speed, so you can check that your board connects, sends plain numbers, reaches 200 Hz "
            "and that logging and plotting work. The simulated speed is <b>not real wind</b>; don't compare your "
            "sensor with it."),
          figure(cropped("simulated.png", 0.6), "Simulated ground truth for bench tests."),
          PageBreak()]

    # 11 troubleshooting
    s += [H1("11. Troubleshooting"),
          table([["Problem", "What to do"],
                 ["Red <i>NO DATA from the tunnel - tunnel's VDAS box not found</i>", "Check the VDAS box's USB "
                  "cable and that the tunnel's instrument power is on. Close TecQuipment's VDAS software if "
                  "it is open."],
                 ["Red <i>... the VDAS box sends no data</i>", "Check the tunnel's instrument power, then unplug "
                  "and replug the box's USB cable. Close TecQuipment's VDAS software if it is open."],
                 ["Red <i>... none from pressure cell 1</i>", "The AFA5 module's instance switch isn't set to 1: "
                  "ask your TA."],
                 ["Linux: <i>VDAS box not found</i> although it is plugged in", "Allow access to it, then unplug "
                  "and replug it: <font name='Mono' size='8'>echo 'SUBSYSTEM==\"hidraw\", "
                  "ATTRS{idVendor}==\"1781\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/"
                  "99-vdas.rules</font>, then <font name='Mono' size='8'>sudo udevadm control --reload</font>."],
                 ["Red <i>the fan was running</i>", "Stop the fan, wait for still air, press <b>Re-zero tunnel</b>."],
                 ["Tunnel reads far from 0 with the fan off", "Press <b>Re-zero tunnel</b>; if it stays off, do "
                  "step A of section 5 (or ask your TA)."],
                 ["Red <i>The GIGA pitot tube hasn't confirmed LEQ ...</i>", "It is still starting, or isn't running "
                  "the WindSpeedGiga firmware. Unplug and replug it; the end of that line shows what it reports."],
                 ["Red <i>The GIGA rejected its zero</i>", "Air was moving. Stop the fan (or push the GIGA's two "
                  "tubes into each other) and press <b>Re-zero GIGA</b>."],
                 ["<i>cannot open COMx</i>", "Another program has the port open (PuTTY, CubeIDE console, serial "
                  "monitor): close it. Or the board was unplugged: plug it in and click Refresh."],
                 ["Your rate is red", "Your board sends fewer than about 196 lines per second, or some lines aren't "
                  "plain numbers (the bad line is shown). Check your firmware's timing and print format."],
                 ["Your reading doesn't change", "Wrong port selected, or the board isn't printing. Try the port in "
                  "Device Manager, or test in simulated mode."],
                 ["<i>Windows protected your PC</i>", "<b>More info</b>, then <b>Run anyway</b> (section 3)."],
                 ["pip: <i>DLL load failed while importing QtCore</i>", "Your Python is an old Anaconda. Install "
                  "Python from python.org, or use the .exe."],
                 ["pip error about a <i>long path</i>", "Install from a shorter folder, or tick <i>Disable path length "
                  "limit</i> at the end of the python.org installer."]],
                [0.33, 0.67]),
          H2("Getting help"),
          P("<b>Help</b> in the program repeats the main steps. If the program shows an error, details are saved to "
            "<font name='Mono'>%APPDATA%\\Wind Tunnel Logger\\error.log</font> (type that into the File Explorer "
            "address bar). Send that file, and a screenshot of the window, to your TA."),
          ]
    return s


# ---------------------------------------------------------------- screenshots
def take_screenshots():
    """Screenshots from the real Wind Tunnel Logger windows (simulated data, staged to look like a tunnel session:
    COM3 board, team3) into docs/img. Windows appear on screen for about a minute."""
    import tempfile
    import time

    from PySide6 import QtCore, QtWidgets
    from wind_tunnel_logger import app as g
    g.MainWindow.update_giga_choices = lambda self, ports=None: None   # same pictures with a GIGA plugged in

    IMG.mkdir(exist_ok=True)
    SHOWN_LOGS = r"C:\Users\student\Documents\Wind Tunnel Logger\logs"
    qa = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="wtl_shots_"))

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            qa.processEvents()
            time.sleep(0.01)


    def save(widget, name, rect=None):
        pix = widget.grab(rect) if rect is not None else widget.grab()
        pix.save(str(IMG / name))
        print("saved", name, pix.width(), "x", pix.height())


    def top_rect(w, below):
        """Window area from the top down to (and including) widget `below`."""
        bottom = below.mapTo(w, QtCore.QPoint(0, below.height())).y() + 6
        return QtCore.QRect(0, 0, w.width(), bottom)


    def new_window(simulated=True, fake_student=True, name="s"):
        cfg = g.load_config(tmp / "config.yaml")
        cfg["vdas"]["zero_seconds"] = 3
        st = QtCore.QSettings(str(tmp / f"{name}.ini"), QtCore.QSettings.Format.IniFormat)
        st.clear()
        w = g.MainWindow(cfg, st, tmp, simulated=simulated, fake_student=fake_student)
        w.resize(1320, 700)
        w.show()
        return w


    def stage_tunnel(w):
        """Make a simulated session look like the tunnel with a NUCLEO on COM3."""
        w.cmb_source.blockSignals(True)
        w.cmb_source.setCurrentIndex(0)
        w.cmb_source.blockSignals(False)
        w.lbl_gt_title.setText("Tunnel ground truth")
        w.setWindowTitle(f"{g.APP_NAME} {g.VERSION}")
        w.cmb_port.clear()
        w.cmb_port.addItem("COM3 - STMicroelectronics STLink Virtual COM Port (COM3)", "COM3")
        w.btn_connect.setText("Disconnect")
        w.student_port = "COM3"
        w.student_status = "COM3 open"
        w.txt_team.setText("team3")
        w.refresh()
        qa.processEvents()


    # 1. start-up: measuring the tunnel's zero (fan off)
    w = new_window(name="a")
    pump(1.6)
    stage_tunnel(w)
    w.student_status = "COM3 open"
    save(w, "zeroing.png", top_rect(w, w.lbl_status))

    # 2. running, logging off, then logging on
    pump(24)
    stage_tunnel(w)
    save(w, "main_window.png")
    w.toggle_logging()
    pump(12)
    stage_tunnel(w)
    save(w, "logging_on.png", top_rect(w, w.lbl_status))

    # 3. Stop & Save -> "Log saved" box, then Plot run
    w._stop_logging()
    base = tmp / "team3_20261022_143005"
    files = w.write_logs(base.with_suffix(".csv"))
    box, _, _ = w._saved_box(files)
    box.setText(box.text().replace(str(tmp), SHOWN_LOGS))
    box.show()
    pump(0.4)
    save(box, "saved_dialog.png")
    box.close()
    win = w.plot_saved_run(w.last_run)
    win.resize(1150, 780)
    for lbl in win.findChildren(QtWidgets.QLabel):
        lbl.setText(lbl.text().replace(str(tmp), SHOWN_LOGS))
    pump(0.8)
    save(win, "plot_run.png")
    win.close()

    # 5. what happens if the fan was running during the zero (real code path)
    w.zero_samples = [-24.0] * 15
    w._finish_zero()
    w.refresh()
    pump(0.2)
    stage_tunnel(w)
    save(w, "zero_fan_running.png", top_rect(w, w.lbl_status))
    w.close()

    # 6. simulated (bench test) mode, as it really looks
    w = new_window(name="b")
    pump(14)
    w.txt_team.setText("team3")
    save(w, "simulated.png", top_rect(w, w.lbl_status))
    w.close()

    # 7. tunnel mode with no VDAS box: the status line explains (a box that can't be opened, whatever is plugged in)
    from wind_tunnel_logger import vdas_afa5  # noqa: E402


    class NoBox:
        def __init__(self, cell=1):
            raise RuntimeError("tunnel's VDAS box not found (open failed)")


    real_reader, vdas_afa5.GroundTruthReader = vdas_afa5.GroundTruthReader, NoBox
    w = new_window(simulated=False, fake_student=False, name="c")
    w.cmb_port.clear()
    w.cmb_port.addItem("COM3 - STMicroelectronics STLink Virtual COM Port (COM3)", "COM3")
    pump(1.0)
    w.refresh()
    qa.processEvents()
    save(w, "no_box.png", top_rect(w, w.lbl_status))
    w.close()
    vdas_afa5.GroundTruthReader = real_reader
    print("screenshots done:", IMG)


def main():
    if "--no-screenshots" not in sys.argv:
        take_screenshots()
    missing = [n for n in ("main_window.png", "zeroing.png", "zero_fan_running.png", "no_box.png", "logging_on.png",
                           "saved_dialog.png", "plot_run.png", "simulated.png") if not (IMG / n).exists()]
    if missing:
        sys.exit(f"missing screenshots (run without --no-screenshots): {missing}")
    doc = Manual(str(OUT))
    doc.multiBuild(story())
    print("Built", OUT)


if __name__ == "__main__":
    main()
