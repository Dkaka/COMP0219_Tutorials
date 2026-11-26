/* COMP0219 Week 7 lens calibration tutorial: page behaviour and the interactive figures.
   Adapted from the COMP0241 intrinsic calibration tutorial (tutorial.js of that page).
   No build step and no files from other sites: the data comes from sample-data.js (window.CALIB_DATA),
   KaTeX from vendor/katex/, and the code colouring is done by the small highlighter below. */
(function () {
  'use strict';

  var RED = '#e30613', INK = '#000', MUTE = '#6b6b6b', FAINT = '#c9c9c9';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var DATA = window.CALIB_DATA || null;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function fmt(n, d) { var s = Math.abs(n).toFixed(d); return (n < 0 && +s !== 0 ? '−' : '') + s; }
  function fmtS(n, d) { var s = Math.abs(n).toFixed(d); return (+s === 0 ? ' ' : n < 0 ? '−' : '+') + s; }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, extra) {
    var t = el('text', Object.assign({ x: x, y: y }, extra || {}), parent);
    if (cls) t.setAttribute('class', cls);
    t.textContent = s;
    return t;
  }
  function store(key, val) {
    try {
      if (val === undefined) return JSON.parse(localStorage.getItem(key) || 'null');
      localStorage.setItem(key, JSON.stringify(val));
    } catch (e) { return null; }
  }
  function statList(dl, rows) {
    dl.innerHTML = '';
    var out = {};
    rows.forEach(function (r) {
      var d = document.createElement('div'); d.className = 'stat' + (r.hot ? ' hot' : '');
      var dt = document.createElement('dt'); dt.textContent = r.label;
      var dd = document.createElement('dd');
      d.appendChild(dt); d.appendChild(dd); dl.appendChild(d);
      out[r.id] = function (v, unit) { dd.innerHTML = ''; dd.appendChild(document.createTextNode(v)); if (unit) { var u = document.createElement('span'); u.className = 'u'; u.textContent = unit; dd.appendChild(u); } };
    });
    return out;
  }
  function short(name) { var m = name.match(/(\d+)\.jpg$/); return m ? m[1] : name; }

  /* ---------------------------------------------------------------- code colouring */
  // A small highlighter for the Python and Bash snippets on this page. It marks comments, strings, numbers,
  // keywords and function names with the same class names as highlight.js, so site.css colours them.
  var PY_KW = 'and as assert break class continue def del elif else except finally for from global if import in is lambda not or pass raise return try while with yield None True False'.split(' ');
  var PY_BI = 'print len range int float str list dict tuple min max round sorted enumerate zip open abs sum next'.split(' ');
  var SH_KW = 'sudo apt install source cd ls mkdir cp rm git python python3 pip scp ssh export'.split(' ');
  function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function highlight(src, lang) {
    var out = '', i = 0, n = src.length;
    var word = /[A-Za-z_][A-Za-z0-9_]*/y, num = /\d+(\.\d+)?(e-?\d+)?/y;
    function span(cls, s) { return '<span class="hljs-' + cls + '">' + esc(s) + '</span>'; }
    while (i < n) {
      var c = src[i];
      if (c === '#') {  // a comment runs to the end of the line
        var e = src.indexOf('\n', i); if (e < 0) e = n;
        out += span('comment', src.slice(i, e)); i = e; continue;
      }
      if (c === '"' || c === "'") {
        var q = c, j = i + 1;
        if (lang === 'python' && src.substr(i, 3) === q + q + q) {  // a docstring
          j = src.indexOf(q + q + q, i + 3); j = j < 0 ? n : j + 3;
        } else {
          while (j < n && src[j] !== q && src[j] !== '\n') { if (src[j] === '\\') j++; j++; }
          j = Math.min(n, j + 1);
        }
        var pre = out.match(/(^|[^A-Za-z0-9_])([frbFRB]{1,2})$/);  // a prefix such as f"..." belongs to the string
        if (lang === 'python' && pre) { out = out.slice(0, out.length - pre[2].length); out += span('string', pre[2] + src.slice(i, j)); }
        else out += span('string', src.slice(i, j));
        i = j; continue;
      }
      word.lastIndex = i;
      var w = word.exec(src);
      if (w && w.index === i) {
        var t = w[0], prev = i > 0 ? src[i - 1] : ' ';
        if (lang === 'python' && PY_KW.indexOf(t) >= 0) out += span('keyword', t);
        else if (lang === 'python' && /\s*\(/.test(src.slice(i + t.length, i + t.length + 2)) && /def\s+$/.test(src.slice(Math.max(0, i - 6), i))) out += span('title function_', t);
        else if (lang === 'python' && PY_BI.indexOf(t) >= 0 && prev !== '.') out += span('built_in', t);
        else if (lang === 'bash' && SH_KW.indexOf(t) >= 0 && /(^|\n|&&\s*|\|\s*)$/.test(src.slice(0, i).replace(/[ \t]+$/, ''))) out += span('built_in', t);
        else out += esc(t);
        i += t.length; continue;
      }
      num.lastIndex = i;
      var m = num.exec(src);
      if (m && m.index === i && !/[A-Za-z0-9_.]/.test(i > 0 ? src[i - 1] : ' ')) { out += span('number', m[0]); i += m[0].length; continue; }
      out += esc(c); i++;
    }
    return out;
  }

  function setupCode() {
    $$('pre code').forEach(function (code) {
      var raw = code.textContent;
      code.parentNode.parentNode.dataset.raw = raw;
      var lang = (code.className.match(/language-(\w+)/) || [])[1];
      if (lang === 'python' || lang === 'bash') code.innerHTML = highlight(raw, lang);
      var start = parseInt(code.dataset.start || '0', 10);
      if (!start) return;
      var hl = {};
      (code.dataset.hl || '').split(',').forEach(function (part) {
        if (!part) return;
        var ab = part.split('-').map(Number);
        for (var i = ab[0]; i <= (ab[1] || ab[0]); i++) hl[i] = true;
      });
      code.innerHTML = code.innerHTML.split('\n').map(function (line, i) {
        var n = start + i;
        var inner = '<span class="ln">' + n + '</span>' + line;
        return hl[n] ? '<span class="hl-line">' + inner + '</span>' : inner;
      }).join('\n');
    });
    $$('.code .copy').forEach(function (btn) {
      var file = btn.parentNode.querySelector('.file');
      btn.setAttribute('aria-label', 'Copy code: ' + (file ? file.textContent : 'snippet'));
      btn.addEventListener('click', function () {
        var text = btn.closest('.code').dataset.raw || '';
        var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1400); };
        if (navigator.clipboard && window.isSecureContext) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
        } else { fallbackCopy(text); done(); }
      });
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  /* ---------------------------------------------------------------- math */
  function setupMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false
    });
  }

  /* ---------------------------------------------------------------- contents + progress */
  function setupNav() {
    // Narrow screens hide the sticky sidebar; give them the same contents as a collapsible list after the hero.
    var toc = $('.toc'), hero = $('.hero');
    if (toc && hero) {
      var d = document.createElement('details'); d.className = 'toc-inline';
      var parts = $$('section.part-s').length;
      var sm = document.createElement('summary');
      sm.textContent = 'Contents · ' + (parts ? parts + ' parts' : $$('section.s').length + ' sections'); d.appendChild(sm);
      var wrap = document.createElement('div'); wrap.className = 'toc-inline-body';
      Array.prototype.slice.call(toc.children).forEach(function (n) {
        if (n.matches('.part, ol')) wrap.appendChild(n.cloneNode(true));
      });
      d.appendChild(wrap);
      hero.parentNode.insertBefore(d, hero.nextSibling);
      $$('a', d).forEach(function (a) { a.addEventListener('click', function () { d.open = false; }); });
    }
    var links = $$('.toc a');
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var sections = $$('section.part-s, section.s');
    var bar = $('#progress');
    function onScroll() {
      var h = document.documentElement;
      var f = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      if (bar) bar.style.width = (100 * Math.min(1, Math.max(0, f))) + '%';
      var cur = null;
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top < 140) cur = sections[i].id;
      }
      links.forEach(function (a) { a.classList.remove('on'); });
      if (cur && byId[cur]) byId[cur].classList.add('on');
      var part = cur && document.getElementById(cur).closest('section.part-s');
      if (part && byId[part.id]) byId[part.id].classList.add('on');
      var on = cur && byId[cur];
      if (on && toc && toc.scrollHeight > toc.clientHeight) {
        var tr = toc.getBoundingClientRect(), r = on.getBoundingClientRect();
        if (r.bottom > tr.bottom - 8) toc.scrollTop += r.bottom - tr.bottom + tr.height / 3;
        else if (r.top < tr.top + 8) toc.scrollTop -= tr.top - r.top + tr.height / 3;
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* ---------------------------------------------------------------- checklist */
  function setupChecklist() {
    var KEY = 'comp0219-w7-calib-checklist';
    var boxes = $$('#checklist input[type="checkbox"]');
    if (!boxes.length) return;
    var saved = store(KEY) || {};
    var count = $('#check-count');
    function update() {
      var n = boxes.filter(function (b) { return b.checked; }).length;
      count.textContent = n + ' / ' + boxes.length + ' done';
    }
    boxes.forEach(function (b) {
      b.checked = !!saved[b.dataset.k];
      b.addEventListener('change', function () { saved[b.dataset.k] = b.checked; store(KEY, saved); update(); });
    });
    $('#check-reset').addEventListener('click', function () { boxes.forEach(function (b) { b.checked = false; }); saved = {}; store(KEY, saved); update(); });
    update();
  }

  /* ---------------------------------------------------------------- 2.2: inner corners diagram */
  function drawBoard() {
    var svg = $('#board-svg'); if (!svg) return;
    var x0 = 70, y0 = 62, s = 40, C = 8, R = 5;
    for (var r = 0; r < R; r++) for (var c = 0; c < C; c++) {
      el('rect', { x: x0 + c * s, y: y0 + r * s, width: s, height: s, fill: (r + c) % 2 === 0 ? '#111' : '#fff' }, svg);
    }
    el('rect', { x: x0, y: y0, width: C * s, height: R * s, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, svg);
    for (var j = 1; j < R; j++) for (var i = 1; i < C; i++) {
      el('circle', { cx: x0 + i * s, cy: y0 + j * s, r: 5.5, fill: RED, stroke: '#fff', 'stroke-width': 1.5 }, svg);
    }
    el('line', { x1: x0, y1: 44, x2: x0 + C * s, y2: 44, stroke: INK, 'stroke-width': 1 }, svg);
    for (var k = 0; k <= C; k++) el('line', { x1: x0 + k * s, y1: 40, x2: x0 + k * s, y2: 48, stroke: INK, 'stroke-width': 1 }, svg);
    txt(svg, x0, 30, '8 squares across', 'mono');
    txt(svg, x0 + C * s, 30, '7 inner corners', 'mono', { 'text-anchor': 'end', fill: RED, style: 'fill:' + RED });
    var xs = x0 + C * s + 18;
    el('line', { x1: xs, y1: y0, x2: xs, y2: y0 + R * s, stroke: INK, 'stroke-width': 1 }, svg);
    for (var m = 0; m <= R; m++) el('line', { x1: xs - 4, y1: y0 + m * s, x2: xs + 4, y2: y0 + m * s, stroke: INK, 'stroke-width': 1 }, svg);
    txt(svg, xs + 14, y0 + 92, '5 squares down', 'mono');
    txt(svg, xs + 14, y0 + 112, '4 inner corners', 'mono', { fill: RED, style: 'fill:' + RED });
    var yb = y0 + R * s + 22;
    el('line', { x1: x0, y1: yb, x2: x0 + s, y2: yb, stroke: INK, 'stroke-width': 1 }, svg);
    el('line', { x1: x0, y1: yb - 5, x2: x0, y2: yb + 5, stroke: INK }, svg);
    el('line', { x1: x0 + s, y1: yb - 5, x2: x0 + s, y2: yb + 5, stroke: INK }, svg);
    txt(svg, x0 + s + 10, yb + 4, 'square_size = 34 mm', 'mono');
    txt(svg, x0 + C * s, yb + 30, '7 × 4 = 28 corners per photo', 'mono', { 'text-anchor': 'end', fill: RED, style: 'fill:' + RED });
  }

  /* ---------------------------------------------------------------- 2.3: coverage maps */
  // The sample photos: all detected corners on one 640 x 480 frame, with the four corner regions of the frame.
  function drawCoverageSample() {
    var svg = $('#coverage-svg'); if (!svg || !DATA) return;
    var D = DATA.sample, W = D.size[0], H = D.size[1], zw = W * 0.2, zh = H * 0.2;
    el('rect', { x: 0, y: 0, width: W, height: H, fill: '#fff' }, svg);
    for (var gx = 64; gx < W; gx += 64) el('line', { x1: gx, y1: 0, x2: gx, y2: H, stroke: '#eee' }, svg);
    for (var gy = 48; gy < H; gy += 48) el('line', { x1: 0, y1: gy, x2: W, y2: gy, stroke: '#eee' }, svg);
    var pts = [];
    D.images.forEach(function (im) { im.obs.forEach(function (p) { pts.push(p); }); });
    [[0, 0], [W - zw, 0], [0, H - zh], [W - zw, H - zh]].forEach(function (z) {
      var n = pts.filter(function (p) { return p[0] >= z[0] && p[0] < z[0] + zw && p[1] >= z[1] && p[1] < z[1] + zh; }).length;
      el('rect', { x: z[0] + 1, y: z[1] + 1, width: zw - 2, height: zh - 2, fill: 'rgba(227,6,19,0.06)', stroke: RED, 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, svg);
      txt(svg, z[0] + zw / 2, z[1] + zh / 2 + 5, n + (n === 1 ? ' corner' : ' corners'), 'mono', { 'text-anchor': 'middle', fill: RED, style: 'fill:' + RED + ';font-size:14px' });
    });
    pts.forEach(function (p) { el('circle', { cx: p[0], cy: p[1], r: 3.2, fill: INK }, svg); });
    txt(svg, W / 2, 22, '640 × 480 frame · ' + pts.length + ' corners from ' + D.images.length + ' photos', 'mono', { 'text-anchor': 'middle', fill: MUTE, style: 'fill:' + MUTE + ';font-size:13px' });
  }

  // The HQ camera photos: all detected corners on one 2028 x 1520 frame, counted in each ninth of the image.
  function drawCoverageHQ() {
    var svg = $('#coverage-hq-svg'); if (!svg || !DATA) return;
    var D = DATA.hq_all, W = D.size[0], H = D.size[1];
    var pts = [];
    D.images.forEach(function (im) { im.obs.forEach(function (p) { pts.push(p); }); });
    var count = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    pts.forEach(function (p) { count[Math.min(2, Math.floor(3 * p[1] / H))][Math.min(2, Math.floor(3 * p[0] / W))]++; });
    el('rect', { x: 0, y: 0, width: W, height: H, fill: '#fff' }, svg);
    for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) {
      var few = count[r][c] < 0.05 * pts.length;  // less than 5% of all corners: as good as empty
      if (few) el('rect', { x: c * W / 3, y: r * H / 3, width: W / 3, height: H / 3, fill: 'rgba(227,6,19,0.08)' }, svg);
    }
    pts.forEach(function (p) { el('circle', { cx: p[0], cy: p[1], r: 9, fill: INK }, svg); });
    for (var k = 1; k < 3; k++) {
      el('line', { x1: k * W / 3, y1: 0, x2: k * W / 3, y2: H, stroke: MUTE, 'stroke-width': 3, 'stroke-dasharray': '18 12' }, svg);
      el('line', { x1: 0, y1: k * H / 3, x2: W, y2: k * H / 3, stroke: MUTE, 'stroke-width': 3, 'stroke-dasharray': '18 12' }, svg);
    }
    for (r = 0; r < 3; r++) for (c = 0; c < 3; c++) {
      var few2 = count[r][c] < 0.05 * pts.length;
      var x = (c + 0.5) * W / 3, y = (r + 0.5) * H / 3 + 22;
      el('rect', { x: x - 150, y: y - 62, width: 300, height: 86, fill: 'rgba(255,255,255,.9)' }, svg);
      txt(svg, x, y, count[r][c] + ' corners', 'mono', { 'text-anchor': 'middle', style: 'font-size:52px;fill:' + (few2 ? RED : INK) });
    }
    svg.setAttribute('aria-label', 'All ' + pts.length + ' detected corners of the ' + D.images.length + ' HQ camera photos on one 2028 by 1520 frame. Corners per ninth, top row: ' + count[0].join(', ') + '; middle row: ' + count[1].join(', ') + '; bottom row: ' + count[2].join(', ') + '.');
  }

  /* ---------------------------------------------------------------- 1.3: distortion explorer */
  // Works in the pixels of the HQ camera photo (2028 x 1520, f = 1986 px, principal point at the centre) and
  // draws a copy scaled down to the 640 x 480 canvas.
  function setupDistortion() {
    var cv = $('#dist-canvas'); if (!cv) return;
    var W = 2028, H = 1520, F = 1986, CX = W / 2, CY = H / 2, CW = 640, CH = 480, S = CW / W;
    var ctx = cv.getContext('2d');
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = CW * dpr; cv.height = CH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cv.tabIndex = 0;
    var hqD = DATA ? DATA.hq_clean.D : null;
    var PRESETS = {
      none:   { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0 },
      barrel: { k1: -0.3, k2: 0.08, k3: 0, p1: 0, p2: 0 },
      pin:    { k1: 0.25, k2: 0.05, k3: 0, p1: 0, p2: 0 },
      tan:    { k1: 0, k2: 0, k3: 0, p1: 0.02, p2: -0.015 },
      // the result of calibration.py for the HQ camera (20 photos), from sample-data.js
      hq:     hqD ? { k1: hqD[0], k2: hqD[1], k3: hqD[4], p1: hqD[2], p2: hqD[3] } : null
    };
    if (!PRESETS.hq) { var hb = $('#dist-presets button[data-p="hq"]'); if (hb) hb.hidden = true; }
    var KEYS = ['k1', 'k2', 'k3', 'p1', 'p2'];
    var DEC = { k1: 3, k2: 3, k3: 2, p1: 3, p2: 3 };
    var first = PRESETS.hq ? 'hq' : 'barrel';
    var st = Object.assign({}, PRESETS[first]);
    var probe = { u: 300, v: 230 };
    var sliders = {};
    KEYS.forEach(function (k) {
      sliders[k] = $('#' + k);
      sliders[k].addEventListener('input', function () {
        st[k] = parseFloat(sliders[k].value);
        markPreset(null);
        draw();
      });
    });
    var stats = statList($('#dist-stats'), [
      { id: 'xy', label: 'Probe x, y (normalised)' },
      { id: 'r2', label: 'r² = x² + y²' },
      { id: 'rad', label: 'Radial factor 1 + k₁r² + k₂r⁴ + k₃r⁶' },
      { id: 'tan', label: 'Tangential shift (x, y)' },
      { id: 'xyd', label: 'x_d, y_d' },
      { id: 'shift', label: 'Probe moves by', hot: true },
      { id: 'corner', label: 'Largest image-corner shift' }
    ]);
    function distort(u, v) {
      var x = (u - CX) / F, y = (v - CY) / F;
      var r2 = x * x + y * y, rad = 1 + st.k1 * r2 + st.k2 * r2 * r2 + st.k3 * r2 * r2 * r2;
      var tx = 2 * st.p1 * x * y + st.p2 * (r2 + 2 * x * x);
      var ty = st.p1 * (r2 + 2 * y * y) + 2 * st.p2 * x * y;
      var xd = x * rad + tx, yd = y * rad + ty;
      return { x: x, y: y, r2: r2, rad: rad, tx: tx, ty: ty, xd: xd, yd: yd, u: F * xd + CX, v: F * yd + CY };
    }
    function markPreset(name) {
      $$('#dist-presets button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.p === name ? 'true' : 'false'); });
    }
    $$('#dist-presets button').forEach(function (b) {
      b.addEventListener('click', function () {
        st = Object.assign({}, PRESETS[b.dataset.p]);
        markPreset(b.dataset.p);
        draw();
      });
    });
    function polyline(pts, color, width) {
      ctx.beginPath();
      pts.forEach(function (p, i) { if (i) ctx.lineTo(p[0] * S, p[1] * S); else ctx.moveTo(p[0] * S, p[1] * S); });
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    }
    function draw() {
      KEYS.forEach(function (k) {
        sliders[k].value = st[k];
        $('#' + k + 'v').textContent = fmtS(st[k], DEC[k]);
      });
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, CW, CH);
      var sx = W / 16, sy = H / 12, i, u, v, pts;
      ctx.lineCap = 'round';
      for (i = 0; i <= 16; i++) polyline([[i * sx, 0], [i * sx, H]], FAINT, 1);
      for (i = 0; i <= 12; i++) polyline([[0, i * sy], [W, i * sy]], FAINT, 1);
      for (i = 0; i <= 16; i++) {
        u = i * sx; pts = [];
        for (v = 0; v <= H; v += 12) { var d = distort(u, v); pts.push([d.u, d.v]); }
        polyline(pts, INK, (i === 0 || i === 16) ? 2.4 : 1.1);
      }
      for (i = 0; i <= 12; i++) {
        v = i * sy; pts = [];
        for (u = 0; u <= W; u += 12) { var e = distort(u, v); pts.push([e.u, e.v]); }
        polyline(pts, INK, (i === 0 || i === 12) ? 2.4 : 1.1);
      }
      var cx = CX * S, cy = CY * S;
      ctx.strokeStyle = MUTE; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy); ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7); ctx.stroke();
      var p = distort(probe.u, probe.v);
      ctx.beginPath(); ctx.arc(probe.u * S, probe.v * S, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fill(); ctx.strokeStyle = MUTE; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(probe.u * S, probe.v * S); ctx.lineTo(p.u * S, p.v * S); ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.u * S, p.v * S, 5.5, 0, 2 * Math.PI); ctx.fillStyle = RED; ctx.fill();
      var shift = Math.hypot(p.u - probe.u, p.v - probe.v);
      var cs = 0;
      [[0, 0], [W, 0], [0, H], [W, H]].forEach(function (q) { var c = distort(q[0], q[1]); cs = Math.max(cs, Math.hypot(c.u - q[0], c.v - q[1])); });
      stats.xy(fmt(p.x, 3) + ', ' + fmt(p.y, 3));
      stats.r2(fmt(p.r2, 3));
      stats.rad(fmt(p.rad, 4));
      stats.tan(fmt(p.tx, 4) + ', ' + fmt(p.ty, 4));
      stats.xyd(fmt(p.xd, 3) + ', ' + fmt(p.yd, 3));
      stats.shift(shift.toFixed(0), 'px');
      stats.corner(cs.toFixed(0), 'px');
    }
    function toImage(e) {
      var r = cv.getBoundingClientRect();
      return { u: Math.max(0, Math.min(W, (e.clientX - r.left) / r.width * W)), v: Math.max(0, Math.min(H, (e.clientY - r.top) / r.height * H)) };
    }
    var dragging = false;
    cv.addEventListener('pointerdown', function (e) { dragging = true; cv.setPointerCapture(e.pointerId); probe = toImage(e); draw(); });
    cv.addEventListener('pointermove', function (e) { if (dragging) { probe = toImage(e); draw(); } });
    cv.addEventListener('pointerup', function () { dragging = false; });
    cv.addEventListener('pointercancel', function () { dragging = false; });
    cv.addEventListener('keydown', function (e) {
      var d = e.shiftKey ? 120 : 24, k = e.key;
      if (k === 'ArrowLeft') probe.u -= d; else if (k === 'ArrowRight') probe.u += d;
      else if (k === 'ArrowUp') probe.v -= d; else if (k === 'ArrowDown') probe.v += d; else return;
      probe.u = Math.max(0, Math.min(W, probe.u)); probe.v = Math.max(0, Math.min(H, probe.v));
      e.preventDefault(); draw();
    });
    markPreset(first);
    draw();
  }

  /* ---------------------------------------------------------------- 3.4: reprojection residuals */
  function setupResiduals() {
    var svg = $('#res-svg'); if (!svg || !DATA) return;
    var setSeg = $('#res-sets'), pick = $('#res-pick'), mag = $('#res-mag'), magv = $('#res-magv'), bars = $('#res-bars'), foot = $('#res-foot');
    var SETS = {
      sample: { D: DATA.sample, label: 'Sample photos', thumb: function (n) { return `img/sample_${n}.jpg`; },
                foot: 'The six sample photos (640 × 480). Choose photo 02 in the list: it has the largest error (0.61 px RMS; the other five have 0.21–0.41 px), and its worst corner is about 1 px away from the prediction. The printed sheet in the sample photos is bent by up to about 1 mm, and no lens model can describe a bent board.' },
      hq_all: { D: DATA.hq_all, label: 'HQ camera: 22 photos', thumb: function (n) { return `img/hq/hq_${n}.jpg`; },
                foot: 'All 22 HQ camera photos in which the chessboard was found (2028 × 1520). Two bars stand out: photo 19 (2.36 px), in which the board moved while the photo was taken, and photo 04 (2.01 px), a very close and strongly tilted view that is slightly soft. Click a bar to see that photo.' },
      hq_clean: { D: DATA.hq_clean, label: 'HQ camera: without 04 and 19', thumb: function (n) { return `img/hq/hq_${n}.jpg`; },
                foot: 'The same calibration without photos 04 and 19. The RMS error falls from 0.90 to 0.64 px, while fx changes by only 0.5% (1976 to 1986 px). The close-up photos (01 to 08) still have the larger errors, probably because a printed sheet on card is not perfectly flat or because the lens is slightly out of focus at short range.' }
    };
    var setKey = 'sample', sel = 'all';
    Object.keys(SETS).forEach(function (k) {
      var b = document.createElement('button');
      b.textContent = SETS[k].label; b.dataset.v = k;
      // the HQ photos have larger errors in pixels, so they start with a smaller magnification
      b.addEventListener('click', function () { setKey = k; sel = 'all'; mag.value = k === 'sample' ? 100 : 30; fillPick(); draw(); });
      setSeg.appendChild(b);
    });
    function fillPick() {
      pick.innerHTML = '';
      var o = document.createElement('option'); o.value = 'all'; o.textContent = 'All photos'; pick.appendChild(o);
      SETS[setKey].D.images.forEach(function (im) {
        var p = document.createElement('option'); p.value = short(im.name); p.textContent = im.name + ' · ' + im.rms.toFixed(2) + ' px'; pick.appendChild(p);
      });
      pick.value = sel;
    }
    pick.addEventListener('change', function () { sel = pick.value; draw(); });
    var stats = statList($('#res-stats'), [
      { id: 'sel', label: 'RMS error, this selection', hot: true },
      { id: 'max', label: 'Largest single residual' },
      { id: 'rms', label: 'RMS error over all photos' }
    ]);
    mag.addEventListener('input', draw);
    function draw() {
      var set = SETS[setKey], D = set.D, W = D.size[0], H = D.size[1], k = W / 640;
      var M = parseFloat(mag.value) * k;  // the same length on screen for both image sizes
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      magv.textContent = '×' + Math.round(M);
      $$('button', setSeg).forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.v === setKey ? 'true' : 'false'); });
      pick.value = sel;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var ims = sel === 'all' ? D.images : D.images.filter(function (im) { return short(im.name) === sel; });
      el('rect', { x: 0, y: 0, width: W, height: H, fill: '#fff' }, svg);
      if (sel !== 'all') {
        el('image', { href: set.thumb(sel), x: 0, y: 0, width: W, height: H, opacity: 0.5, preserveAspectRatio: 'none' }, svg);
      } else {
        for (var gx = W / 10; gx < W - 1; gx += W / 10) el('line', { x1: gx, y1: 0, x2: gx, y2: H, stroke: '#eee', 'stroke-width': k }, svg);
        for (var gy = H / 10; gy < H - 1; gy += H / 10) el('line', { x1: 0, y1: gy, x2: W, y2: gy, stroke: '#eee', 'stroke-width': k }, svg);
      }
      var sq = 0, n = 0, mx = 0;
      ims.forEach(function (im) {
        im.obs.forEach(function (p, i) {
          var r = im.res[i];
          var len = Math.hypot(r[0], r[1]);
          sq += r[0] * r[0] + r[1] * r[1]; n++; if (len > mx) mx = len;
          // residual = observed - projected; draw from the detected corner towards the projected one
          el('line', { x1: p[0], y1: p[1], x2: p[0] - r[0] * M, y2: p[1] - r[1] * M, stroke: RED, 'stroke-width': 2 * k, 'stroke-linecap': 'round' }, svg);
          el('circle', { cx: p[0], cy: p[1], r: 3 * k, fill: INK, stroke: '#fff', 'stroke-width': k }, svg);
        });
      });
      var L = 0.5 * M, bx = W - 16 * k - L, by = H - 18 * k;
      el('rect', { x: bx - 10 * k, y: by - 22 * k, width: L + 20 * k, height: 34 * k, fill: 'rgba(255,255,255,.88)' }, svg);
      el('line', { x1: bx, y1: by, x2: bx + L, y2: by, stroke: RED, 'stroke-width': 3 * k }, svg);
      txt(svg, bx + L, by - 8 * k, '0.5 px × ' + Math.round(M), 'mono', { 'text-anchor': 'end', style: 'font-size:' + 12 * k + 'px' });
      stats.sel(Math.sqrt(sq / n).toFixed(3), 'px');
      stats.max(mx.toFixed(2), 'px');
      stats.rms(D.rms.toFixed(3), 'px');
      foot.textContent = set.foot;
      drawBars(D);
    }
    function drawBars(D) {
      while (bars.firstChild) bars.removeChild(bars.firstChild);
      var x0 = 30, y0 = 12, w = 262, h = 108;
      var top = Math.max(1, Math.ceil(Math.max.apply(null, D.images.map(function (im) { return im.rms; })) * 2) / 2);
      var bw = w / D.images.length, many = D.images.length > 8;
      for (var t = 0; t <= top + 1e-9; t += top > 1.5 ? 1 : 0.5) {
        var y = y0 + h - t / top * h;
        el('line', { x1: x0, y1: y, x2: x0 + w, y2: y, stroke: t ? '#e2e2e2' : INK, 'stroke-width': 1 }, bars);
        txt(bars, x0 - 5, y + 4, t.toFixed(1), 'mono', { 'text-anchor': 'end', style: 'font:500 10px var(--mono);fill:' + MUTE });
      }
      D.images.forEach(function (im, i) {
        var s = short(im.name), on = sel === s, v = im.rms;
        var bh = v / top * h, x = x0 + i * bw + bw * 0.15;
        var big = v > 1.5 * D.rms;
        var r = el('rect', { x: x, y: y0 + h - bh, width: bw * 0.7, height: bh, fill: on ? RED : INK, style: 'cursor:pointer' }, bars);
        r.addEventListener('click', function () { sel = s; draw(); });
        var tt = el('title', {}, r); tt.textContent = im.name + ': ' + v.toFixed(3) + ' px RMS';
        if (!many || i % 2 === 0 || on || big) txt(bars, x + bw * 0.35, y0 + h + 14, s, 'mono', { 'text-anchor': 'middle', style: 'font:500 ' + (many ? 8.5 : 10) + 'px var(--mono);fill:' + (on ? RED : INK) });
        if (!many || on || big) txt(bars, x + bw * 0.35, y0 + h - bh - 4, v.toFixed(2), 'mono', { 'text-anchor': 'middle', style: 'font:500 9.5px var(--mono);fill:' + (on ? RED : MUTE) });
      });
      var yr = y0 + h - D.rms / top * h;
      el('line', { x1: x0, y1: yr, x2: x0 + w, y2: yr, stroke: INK, 'stroke-dasharray': '4 3', 'stroke-width': 1 }, bars);
      txt(bars, x0, y0 + h + 34, 'RMS per photo (px)', 'mono', { style: 'font:500 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;fill:' + MUTE });
      el('line', { x1: x0, y1: y0 + h + 48, x2: x0 + 18, y2: y0 + h + 48, stroke: INK, 'stroke-dasharray': '4 3', 'stroke-width': 1 }, bars);
      txt(bars, x0 + 24, y0 + h + 52, 'all photos: RMS = ' + D.rms.toFixed(3) + ' px', 'mono', { style: 'font:500 10px var(--mono);fill:' + INK });
    }
    fillPick();
    draw();
  }

  /* ---------------------------------------------------------------- 3.5: undistortion before and after */
  function setupUndistort() {
    var img = $('#und-img'); if (!img) return;
    var PHOTOS = {
      hq: { orig: 'img/hq/photo_00.jpg', name: 'HQ camera, calibration_image_00', origSize: '2028 × 1520', ratio: '4/3', V: {
        a0:    { src: 'img/hq/undistort_alpha0.jpg', cap: 'alpha = 0', size: '2028 × 1520', w: 100, K: [1701.7, 1835.8, 1016.0, 790.3], roi: '(0, 0, 2027, 1519)' },
        a1:    { src: 'img/hq/undistort_alpha1.jpg', cap: 'alpha = 1', size: '2028 × 1520', w: 100, K: [1544.8, 1548.9, 1012.4, 790.5], roi: '(90, 124, 1840, 1282)' },
        a1roi: { src: 'img/hq/undistort_alpha1_roi.jpg', cap: 'alpha = 1, ROI in red', size: '2028 × 1520', w: 100, K: [1544.8, 1548.9, 1012.4, 790.5], roi: '(90, 124, 1840, 1282)' },
        crop:  { src: 'img/hq/undistort_alpha1_cropped.jpg', cap: 'Cropped to the ROI (what the script saves)', size: '1840 × 1282', w: 1840 / 2028 * 100, K: [1544.8, 1548.9, 1012.4, 790.5], roi: '(90, 124, 1840, 1282)' }
      } },
      sample: { orig: 'img/sample_02.jpg', name: 'sample photo calibration_image_02', origSize: '640 × 480', ratio: '4/3', V: {
        a0:    { src: 'img/undistort_alpha0.jpg', cap: 'alpha = 0', size: '640 × 480', w: 100, K: [561.2, 560.9, 362.0, 229.5], roi: '(0, 0, 639, 479)' },
        a1:    { src: 'img/undistort_alpha1.jpg', cap: 'alpha = 1', size: '640 × 480', w: 100, K: [558.2, 556.4, 361.9, 230.1], roi: '(2, 3, 636, 475)' },
        a1roi: { src: 'img/undistort_alpha1_roi.jpg', cap: 'alpha = 1, ROI in red', size: '640 × 480', w: 100, K: [558.2, 556.4, 361.9, 230.1], roi: '(2, 3, 636, 475)' },
        crop:  { src: 'img/undistort_alpha1_cropped.jpg', cap: 'Cropped to the ROI (what the script saves)', size: '636 × 475', w: 636 / 640 * 100, K: [558.2, 556.4, 361.9, 230.1], roi: '(2, 3, 636, 475)' }
      } }
    };
    var photo = 'hq', variant = 'crop';
    var stats = statList($('#und-stats'), [
      { id: 'f', label: 'New fx, fy (px)' },
      { id: 'c', label: 'New cx, cy (px)' },
      { id: 'roi', label: 'roi = (x, y, w, h)', hot: true },
      { id: 'size', label: 'Image shown' }
    ]);
    $('#und-stats').style.display = 'grid';
    $('#und-stats').style.gridTemplateColumns = 'repeat(auto-fit, minmax(170px, 1fr))';
    $('#und-stats').style.columnGap = '16px';
    function show() {
      var P = PHOTOS[photo], v = P.V[variant];
      $('#und-orig').src = P.orig; $('#und-orig').alt = 'Original photo: ' + P.name;
      $('#und-orig-size').textContent = P.origSize;
      $('#und-title').textContent = 'Before and after · ' + P.name;
      img.src = v.src; img.style.width = v.w + '%';
      img.alt = P.name + ' undistorted: ' + v.cap;
      $('#und-cap').textContent = v.cap; $('#und-size').textContent = v.size;
      stats.f(v.K[0].toFixed(1) + ', ' + v.K[1].toFixed(1));
      stats.c(v.K[2].toFixed(1) + ', ' + v.K[3].toFixed(1));
      stats.roi(v.roi);
      stats.size(v.size);
      $$('#und-seg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.v === variant ? 'true' : 'false'); });
      $$('#und-photo button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.v === photo ? 'true' : 'false'); });
    }
    $$('#und-seg button').forEach(function (b) { b.addEventListener('click', function () { variant = b.dataset.v; show(); }); });
    $$('#und-photo button').forEach(function (b) { b.addEventListener('click', function () { photo = b.dataset.v; show(); }); });
    Object.keys(PHOTOS).forEach(function (p) { Object.keys(PHOTOS[p].V).forEach(function (k) { var i = new Image(); i.src = PHOTOS[p].V[k].src; }); });
    show();
  }

  /* ---------------------------------------------------------------- folded code */
  // Code walk-throughs start closed. Open them all for printing, and open the one a link points into.
  function setupFolds() {
    var folds = $$('details.fold');
    if (!folds.length) return;
    window.addEventListener('beforeprint', function () {
      folds.forEach(function (d) { d.dataset.wasOpen = d.open ? '1' : ''; d.open = true; });
    });
    window.addEventListener('afterprint', function () {
      folds.forEach(function (d) { if (d.dataset.wasOpen !== undefined) d.open = d.dataset.wasOpen === '1'; });
    });
    function openTarget() {
      var id = location.hash.slice(1);
      try { id = decodeURIComponent(id); } catch (e) { /* a cut-off link: use the hash as it is */ }
      var t = id && document.getElementById(id);
      var d = t && t.closest('details.fold');
      if (d && !d.open) { d.open = true; t.scrollIntoView(); }
    }
    window.addEventListener('hashchange', openTarget);
    openTarget();
  }

  function init() {
    setupCode();
    setupFolds();
    setupMath();
    setupNav();
    setupChecklist();
    drawBoard();
    drawCoverageSample();
    drawCoverageHQ();
    setupDistortion();
    setupResiduals();
    setupUndistort();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
