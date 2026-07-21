/* global CHARTS, EXAMPLES, LEADERBOARD */
(function () {
  "use strict";

  /* ---------- mobile nav ---------- */
  var navToggle = document.querySelector(".nav-toggle");
  var navLinks = document.querySelector(".nav-links");
  if (navToggle) {
    navToggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
    navLinks.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () { navLinks.classList.remove("open"); });
    });
  }

  /* ---------- inject precomputed charts ---------- */
  Object.keys(CHARTS).forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = CHARTS[id];
  });

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n).trim() + "\u2026" : s;
  }

  /* ---------- mirrored probe card builder ---------- */
  function mcard(ex, opts) {
    opts = opts || {};
    var full = !!opts.full;
    var verdictClass = ex.consistent ? "ok" : "bad";
    var verdictGlyph = ex.consistent ? "\u2713" : "\u2717";
    var respLenA = full ? 100000 : 150;
    var respLenB = full ? 100000 : 150;
    var reasonHtml = "";
    if (full && ex.score_reason) {
      reasonHtml = '<div class="reason">Scorer note: ' + esc(ex.score_reason) + "</div>";
    }
    return (
      '<div class="mcard" data-probe="' + esc(ex.probe_id) + '">' +
        '<div class="mcard-top">' +
          '<div class="mcard-tags">' +
            '<span class="tag" style="background:var(--f-' + ex.family + ')"><span class="fdot" style="background:#fff;opacity:.85"></span>' + esc(ex.family_label) + '</span>' +
            '<span class="tag outline">' + esc(ex.model_label) + '</span>' +
            '<span class="tag outline">' + esc(ex.domain) + " \u00b7 " + esc(ex.difficulty) + '</span>' +
          '</div>' +
          '<span class="mono">#' + esc(ex.probe_id) + '</span>' +
        '</div>' +
        '<div class="mcard-body">' +
          '<div class="mcard-panel">' +
            '<div class="plabel">Prompt A</div>' +
            '<div class="ptext">' + esc(ex.prompt_a) + '</div>' +
            '<div class="rtext"><b>Response A:</b> ' + esc(truncate(ex.response_a, respLenA)) + '</div>' +
          '</div>' +
          '<div class="mcard-seam"><div class="verdict ' + verdictClass + '">' + verdictGlyph + '</div></div>' +
          '<div class="mcard-panel">' +
            '<div class="plabel">Prompt B</div>' +
            '<div class="ptext">' + esc(ex.prompt_b) + '</div>' +
            '<div class="rtext"><b>Response B:</b> ' + esc(truncate(ex.response_b, respLenB)) + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="mcard-foot">' +
          '<span>' + (ex.consistent ? "Logically consistent" : "Inconsistent \u2014 constraint violated") + '</span>' +
          (ex.delta !== null && ex.delta !== undefined ? '<span class="mono">\u03b4=' + ex.delta + '</span>' : "") +
          '<span class="mono">' + esc(ex.score_method || "") + '</span>' +
          reasonHtml +
        '</div>' +
      '</div>'
    );
  }

  /* ---------- hero live demo ---------- */
  var heroSlot = document.getElementById("hero-demo-slot");
  var heroExamples = EXAMPLES.filter(function (e) { return !e.consistent; }).slice(0, 10);
  if (heroSlot && heroExamples.length) {
    var idx = 0;
    function renderHero() {
      heroSlot.style.opacity = 0;
      setTimeout(function () {
        heroSlot.innerHTML = mcard(heroExamples[idx], { full: false });
        heroSlot.style.opacity = 1;
        idx = (idx + 1) % heroExamples.length;
      }, 220);
    }
    heroSlot.style.transition = "opacity .22s ease";
    renderHero();
    setInterval(renderHero, 5200);
  }

  /* ---------- examples browser ---------- */
  var exGrid = document.getElementById("ex-grid");
  var exMeta = document.getElementById("ex-results-meta");
  var fFamily = document.getElementById("filter-family");
  var fModel = document.getElementById("filter-model");
  var fDomain = document.getElementById("filter-domain");
  var fVerdict = document.getElementById("filter-verdict");
  var shuffleBtn = document.getElementById("shuffle-btn");
  var resetBtn = document.getElementById("reset-btn");

  function populateSelect(sel, values, labelFn) {
    values.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v;
      opt.textContent = labelFn ? labelFn(v) : v;
      sel.appendChild(opt);
    });
  }

  if (exGrid) {
    var families = Array.from(new Set(EXAMPLES.map(function (e) { return e.family; }))).sort();
    var models = Array.from(new Set(EXAMPLES.map(function (e) { return e.model; })))
      .sort(function (a, b) {
        var la = EXAMPLES.find(function (e) { return e.model === a; }).model_label;
        var lb = EXAMPLES.find(function (e) { return e.model === b; }).model_label;
        return la.localeCompare(lb);
      });
    var domains = Array.from(new Set(EXAMPLES.map(function (e) { return e.domain; }))).sort();

    populateSelect(fFamily, families, function (v) {
      return EXAMPLES.find(function (e) { return e.family === v; }).family_label;
    });
    populateSelect(fModel, models, function (v) {
      return EXAMPLES.find(function (e) { return e.model === v; }).model_label;
    });
    populateSelect(fDomain, domains, function (v) { return v.charAt(0).toUpperCase() + v.slice(1); });

    var order = EXAMPLES.map(function (_, i) { return i; });

    function currentFiltered() {
      return order
        .map(function (i) { return EXAMPLES[i]; })
        .filter(function (e) {
          if (fFamily.value && e.family !== fFamily.value) return false;
          if (fModel.value && e.model !== fModel.value) return false;
          if (fDomain.value && e.domain !== fDomain.value) return false;
          if (fVerdict.value === "inconsistent" && e.consistent) return false;
          if (fVerdict.value === "consistent" && !e.consistent) return false;
          return true;
        });
    }

    function renderGrid() {
      var list = currentFiltered();
      exMeta.textContent = "Showing " + list.length + " of " + EXAMPLES.length + " probe pairs";
      if (!list.length) {
        exGrid.innerHTML = '<div class="no-results">No probe pairs match these filters. Try widening your selection.</div>';
        return;
      }
      exGrid.innerHTML = list.map(function (ex) {
        var verdictClass = ex.consistent ? "ok" : "bad";
        var verdictWord = ex.consistent ? "Consistent" : "Inconsistent";
        var famColor = "var(--f-" + ex.family + ")";
        return (
          '<div class="ex-card" data-probe="' + esc(ex.probe_id) + '" tabindex="0" role="button" aria-label="Open probe ' + esc(ex.probe_id) + '">' +
            '<div class="ex-card-top">' +
              '<span class="ex-card-fam" style="color:' + famColor + '"><span class="fdot ' + ex.family + '"></span>' + esc(ex.family_label) + '</span>' +
              '<span class="verdict ' + verdictClass + '" style="width:24px;height:24px;font-size:.75rem;">' + (ex.consistent ? "\u2713" : "\u2717") + '</span>' +
            '</div>' +
            '<div class="snippet"><b>A:</b> ' + esc(truncate(ex.prompt_a, 96)) + '</div>' +
            '<div class="snippet"><b>B:</b> ' + esc(truncate(ex.prompt_b, 96)) + '</div>' +
            '<div class="ex-card-meta">' +
              '<span class="pill">' + esc(ex.model_label) + '</span>' +
              '<span class="pill">' + esc(ex.domain) + '</span>' +
              '<span class="pill">' + esc(ex.difficulty) + '</span>' +
              '<span class="pill">' + verdictWord + '</span>' +
            '</div>' +
          '</div>'
        );
      }).join("");

      exGrid.querySelectorAll(".ex-card").forEach(function (card) {
        card.addEventListener("click", function () { openModal(card.dataset.probe, list); });
        card.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openModal(card.dataset.probe, list); }
        });
      });
    }

    [fFamily, fModel, fDomain, fVerdict].forEach(function (el) {
      el.addEventListener("change", renderGrid);
    });
    if (shuffleBtn) {
      shuffleBtn.addEventListener("click", function () {
        for (var i = order.length - 1; i > 0; i--) {
          var j = Math.floor(Math.random() * (i + 1));
          var tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        renderGrid();
      });
    }
    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        fFamily.value = ""; fModel.value = ""; fDomain.value = ""; fVerdict.value = "";
        renderGrid();
      });
    }

    renderGrid();

    /* ---------- modal ---------- */
    var backdrop = document.getElementById("modal-backdrop");
    var modalInner = document.getElementById("modal-inner");
    function openModal(probeId, list) {
      var ex = list.find(function (e) { return String(e.probe_id) === String(probeId); }) ||
               EXAMPLES.find(function (e) { return String(e.probe_id) === String(probeId); });
      if (!ex) return;
      modalInner.innerHTML = mcard(ex, { full: true });
      backdrop.classList.add("open");
      document.body.style.overflow = "hidden";
    }
    function closeModal() {
      backdrop.classList.remove("open");
      document.body.style.overflow = "";
    }
    document.getElementById("modal-close").addEventListener("click", closeModal);
    backdrop.addEventListener("click", function (e) { if (e.target === backdrop) closeModal(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeModal(); });
  }

  /* ---------- leaderboard ---------- */
  var lbBody = document.getElementById("lb-body");
  if (lbBody) {
    var data = LEADERBOARD.map(function (r) {
      return { name: r[0], paradigm: r[1] || "\u2014", mmlu: r[2], comp: r[3], rev: r[4], compl: r[5], ord_: r[6], equiv: r[7], overall: r[8], ccs: r[9] };
    });
    var sortKey = "overall", sortDir = 1;

    function heatColor(val, key) {
      var vals = data.map(function (d) { return d[key]; });
      var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
      var t = max > min ? (val - min) / (max - min) : 0;
      var r1 = 240, g1 = 238, b1 = 230; // paper-dim
      var r2 = 178, g2 = 58, b2 = 46;   // inconsistent
      var r = Math.round(r1 + (r2 - r1) * t);
      var g = Math.round(g1 + (g2 - g1) * t);
      var b = Math.round(b1 + (b2 - b1) * t);
      return "rgba(" + r + "," + g + "," + b + "," + (0.12 + t * 0.30) + ")";
    }

    function renderTable() {
      var rows = data.slice().sort(function (a, b) {
        var av = a[sortKey], bv = b[sortKey];
        if (typeof av === "string") return av.localeCompare(bv) * sortDir;
        return (av - bv) * sortDir;
      });
      lbBody.innerHTML = rows.map(function (r) {
        return (
          "<tr>" +
            '<td class="model">' + esc(r.name) + "</td>" +
            '<td class="paradigm">' + esc(r.paradigm) + "</td>" +
            '<td class="num" style="background:' + heatColor(r.overall, "overall") + '">' + r.overall.toFixed(1) + "%</td>" +
            '<td class="num" style="background:' + heatColor(r.comp, "comp") + '">' + r.comp.toFixed(1) + "</td>" +
            '<td class="num" style="background:' + heatColor(r.rev, "rev") + '">' + r.rev.toFixed(1) + "</td>" +
            '<td class="num" style="background:' + heatColor(r.compl, "compl") + '">' + r.compl.toFixed(1) + "</td>" +
            '<td class="num" style="background:' + heatColor(r.ord_, "ord_") + '">' + r.ord_.toFixed(1) + "</td>" +
            '<td class="num" style="background:' + heatColor(r.equiv, "equiv") + '">' + r.equiv.toFixed(1) + "</td>" +
            '<td class="num">' + r.ccs.toFixed(4) + "</td>" +
          "</tr>"
        );
      }).join("");
    }

    document.querySelectorAll("#lb-head th[data-key]").forEach(function (th) {
      th.addEventListener("click", function () {
        var key = th.dataset.key;
        if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = 1; }
        document.querySelectorAll("#lb-head th .arrow").forEach(function (a) { a.textContent = ""; });
        th.querySelector(".arrow").textContent = sortDir === 1 ? " \u2193" : " \u2191";
        renderTable();
      });
    });
    renderTable();
  }

  /* ---------- copy bibtex ---------- */
  var copyBtn = document.getElementById("copy-bibtex");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.getElementById("bibtex-text").textContent;
      navigator.clipboard.writeText(text).then(function () {
        copyBtn.textContent = "Copied";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600);
      }).catch(function () {});
    });
  }
})();
