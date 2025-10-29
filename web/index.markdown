---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

## Galaxy Clusters

{% assign sorted_clusters = site.galaxy_clusters | sort: 'redshift' %}

<!-- Timeline + always-visible snippets -->
<style>
  .timeline-wrapper {
    display:block;
    margin-bottom:1.5rem;
    position:relative; /* for overlay canvas / absolutely positioned snippets */
  }

  .timeline-column {
    width:100%;
    background:transparent;
    padding:0.5rem;
    box-sizing:border-box;
    position:relative;
    padding-top: 40px; /* shift the entire figure (canvas, overlay, snippets) down by 40px */
    padding-bottom: 100px; /* shift the entire figure (canvas, overlay, snippets) down by 40px */
}
  .timeline-column canvas.chart-canvas {
    width:100%;
    height:120px; /* short horizontal band */
    display:block;
  }

  /* overlay canvas spans the whole wrapper and is used to draw connector lines */
  .timeline-overlay {
    position:absolute;
    left:0;
    top:0;
    width:100%;
    height:100%;
    pointer-events:none;
    z-index:2; /* below snippets so snippets and links are interactive */
  }

  /* snippets overlay: absolute container used to place snippets above/below the arrow */
  .snippets-overlay {
    position:absolute;
    left:0;
    top:0;
    width:100%;
    height:100%;
    pointer-events:none; /* allow individual snippet pointer-events */
    z-index:3;
  }

  .cluster-snippet {
    position:absolute;
    transform: translateX(-50%);
    display:flex;
    gap:0.25rem;
    /* we'll stack name and image vertically; alignment depends on .top/.bottom */
    padding:0.25rem;
    background:rgba(255,255,255,0.02);
    border-radius:6px;
    transition: transform 160ms ease, box-shadow 160ms ease;
    pointer-events:auto; /* allow clicks on links/images */
    white-space:nowrap;
  }

  .cluster-snippet a {
    display:inline-block;
    text-decoration:none;
    color:inherit;
    text-align:center;
  }

  .cluster-snippet img {
    display:block;
    width:120px;
    height:72px;
    object-fit:cover;
    border-radius:4px;
    background:#0b1720;
  }

  /* For top snippets: name should appear above the image.
     For bottom snippets: name should appear below the image.
     Use column-reverse for top so the DOM order (img, name) results in name on top. */
  .cluster-snippet.top {
    flex-direction: column-reverse;
    align-items: center;
    transform-origin: center bottom;
  }
  .cluster-snippet.bottom {
    flex-direction: column;
    align-items: center;
    transform-origin: center top;
  }

  .cluster-snippet .name {
    font-weight:600;
    font-size:0.95rem;
    color:inherit;
    margin:0.25rem 0 0 0;
    line-height:1;
  }

  /* when highlighted, keep transform behavior but ensure stacked layout remains centered */
  .cluster-snippet.highlight {
    transform: translateX(-50%) scale(1.1);
    box-shadow: 0 8px 20px rgba(0,0,0,0.45);
    z-index:5;
  }

  @media (max-width:920px){
    .timeline-column canvas.chart-canvas { height:160px; }
    .cluster-snippet img { width:90px; height:56px; }
  }
</style>

<div class="timeline-wrapper" id="timelineWrapper">
  <div class="timeline-column">
    <canvas id="redshiftTimeline" class="chart-canvas" aria-label="Galaxy cluster redshift timeline" role="img"></canvas>
    <canvas id="timelineOverlay" class="timeline-overlay" aria-hidden="true"></canvas>
    <!-- snippets live inside wrapper so they can be positioned around the arrow -->
    <div class="snippets-overlay" id="snippetsOverlay">
      {% for g in sorted_clusters %}
      {% assign r = forloop.index0 | modulo: 2 %}
      <div class="cluster-snippet {% if r == 0 %}top{% else %}bottom{% endif %}" data-index="{{ forloop.index0 }}">
        <a class="thumb-link" href="{{ g.url_zenodo | default: g.url | relative_url }}" target="_blank" rel="noopener">
          <!-- get rid of the image for now <img src="{{ g.image | default: g.jwst_image | default: g.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}" alt="{{ g.name | escape }}"/> -->
        </a>
        <a class="name" href="{{ g.url_zenodo | default: g.url | relative_url }}" target="_blank" rel="noopener">{{ g.name | escape }}</a>
      </div>
      {% endfor %}
    </div>
  </div>
</div>

<!-- Data object for JS -->
<script>
  const clusters = [
{% for g in sorted_clusters %}
    {
      name: "{{ g.name | escape }}",
      redshift: "{{ g.redshift | default: '' }}",
      status: "{{ g.status | escape }}",
      image: "{{ g.image | default: g.jwst_image | default: g.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}",
      zenodo: "{{ g.url_zenodo | default: '' }}",
      permalink: "{{ g.url | absolute_url }}",
      index: {{ forloop.index0 }}
    }{% unless forloop.last %},{% endunless %}
{% endfor %}
  ];
</script>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
  (function(){
    const wrapper = document.getElementById('timelineWrapper');
    const chartCanvas = document.getElementById('redshiftTimeline');
    const overlay = document.getElementById('timelineOverlay');
    const snippetsOverlay = document.getElementById('snippetsOverlay');

    // prepare data points
    const points = clusters.map(c => {
      const z = parseFloat(String(c.redshift).replace(',', '.'));
      return { x: isFinite(z) ? z : null, y: 0, index: c.index, label: c.name, image: c.image, zenodo: c.zenodo, permalink: c.permalink, rawRedshift: c.redshift };
    });
    const plottable = points.filter(p => p.x !== null);

    // axis bounds
    const zs = plottable.map(p => p.x);
    let minZ = 0, maxZ = 1;
    if (zs.length) {
      const zmin = Math.min(...zs);
      const zmax = Math.max(...zs);
      if (Math.abs(zmax - zmin) < 1e-6) {
        minZ = Math.max(0, zmin - 0.5);
        maxZ = zmax + 0.5;
      } else {
        const pad = (zmax - zmin) * 0.08;
        minZ = Math.max(0, zmin - pad);
        maxZ = zmax + pad;
      }
    }

    chartCanvas.style.height = '120px';
    chartCanvas.height = 120;

    const ctx = chartCanvas.getContext('2d');

    const dataset = {
      label: 'Galaxy clusters (redshift)',
      data: plottable.map(p => ({ x: p.x, y: 0+28, index: p.index, label: p.label, image: p.image, permalink: p.permalink, rawRedshift: p.rawRedshift })),
      backgroundColor: 'black',
      borderColor: 'black',
      pointRadius: 3,
      pointHoverRadius: 10,
    };

    // arrow plugin (thin shaft)
    const arrowPlugin = {
      id: 'horizontalArrow',
      beforeDatasetsDraw(chart, args, options) {
        const {ctx, chartArea: ca} = chart;
        if (!ca) return;
        ctx.save();
        const shaftHeight = options.shaftHeight || 4;
        const headWidth = options.headWidth || 18;
        const headHeight = options.headHeight || 18;
        const y = Math.round((ca.top + ca.bottom) / 2) + 28;
        ctx.fillStyle = options.color || 'black';
        const shaftLeft = ca.left + headWidth;
        const shaftRight = ca.right;
        const shaftTop = y - shaftHeight/2;
        ctx.fillRect(shaftLeft, shaftTop, shaftRight - shaftLeft, shaftHeight);
        ctx.beginPath();
        ctx.moveTo(ca.left, y);
        ctx.lineTo(shaftLeft, y - headHeight/2);
        ctx.lineTo(shaftLeft, y + headHeight/2);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
    };

    const chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: [ dataset ] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        scales: {
          x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Redshift (z)' }, min: minZ, max: maxZ, grid: { display: false }, ticks: { color: '#000' } },
          y: { display: false, min: -1, max: 1 }
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false }, horizontalArrow: { color: 'black', shaftHeight: 4, headWidth: 18, headHeight: 18 } },
        onHover(evt, elements) {
          document.querySelectorAll('.cluster-snippet.highlight').forEach(el => el.classList.remove('highlight'));
          if (elements && elements.length) {
            const el = elements[0];
            const p = chart.data.datasets[el.datasetIndex].data[el.index];
            const snippet = document.querySelector(`.cluster-snippet[data-index="${p.index}"]`);
            if (snippet) snippet.classList.add('highlight');
          }
        },
        onClick(evt, elements) {
          if (elements && elements.length) {
            const el = elements[0];
            const p = chart.data.datasets[el.datasetIndex].data[el.index];
            const url = p.zenodo || p.permalink || '#';
            window.open(url, '_blank', 'noopener');
          }
        }
      },
      plugins: [ arrowPlugin ]
    });

    // position snippets around the chart and draw connectors
    function positionSnippetsAndDraw() {
      const ctx2 = overlay.getContext('2d');
      overlay.width = wrapper.clientWidth;
      overlay.height = wrapper.clientHeight;
      overlay.style.width = overlay.width + 'px';
      overlay.style.height = overlay.height + 'px';
      ctx2.clearRect(0, 0, overlay.width, overlay.height);

      const wrapperRect = wrapper.getBoundingClientRect();
      const chartRect = chartCanvas.getBoundingClientRect();

      // ensure chart elements positions are available
      chart.update();

      // helper: get pixel X inside wrapper for a redshift value; clamp to chart area
      function xForValue(val) {
        try {
          if (val === null) return Math.min(chartRect.right - wrapperRect.left - 20, chartRect.right - wrapperRect.left - 20);
          const px = chart.scales.x.getPixelForValue(val); // pixel relative to canvas
          return (chartRect.left - wrapperRect.left) + px;
        } catch (e) {
          return Math.min(chartRect.right - wrapperRect.left - 20, chartRect.right - wrapperRect.left - 20);
        }
      }

      // position each snippet element
      const snippetEls = snippetsOverlay.querySelectorAll('.cluster-snippet');
      // collect initial positions then resolve horizontal collisions so images don't overlap
      const topGroup = [];
      const bottomGroup = [];
      snippetEls.forEach(el => {
        const idx = Number(el.getAttribute('data-index'));
        const cluster = clusters[idx];
        // compute x pixel for cluster (use redshift if available)
        const z = parseFloat(String(cluster.redshift).replace(',', '.'));
        const hasZ = isFinite(z);
        let px = hasZ ? xForValue(z) : (chartRect.right - wrapperRect.left - 20);
        // clamp within wrapper
        px = Math.max(8, Math.min(px, wrapper.clientWidth - 8));
        // temporary set left to measure width
        el.style.left = px + 'px';
        const snippetRect = el.getBoundingClientRect();
        const sideTop = (idx % 2 === 0) || el.classList.contains('top');
        const topValue = sideTop
          ? (chartRect.top - wrapperRect.top) - snippetRect.height - 8 + (idx * 5)
          : (chartRect.bottom - wrapperRect.top) + 8 + (idx * 5);
        el.style.top = Math.max(-10, Math.min(topValue, wrapper.clientHeight - 6)) + 'px';
        // store for collision resolution
        const item = { el, left: px, width: snippetRect.width, idx, sideTop };
        if (sideTop) topGroup.push(item); else bottomGroup.push(item);
      });

      // simple collision resolver: push overlapping items to the right, then back-adjust if hitting bounds
      function resolveCollisions(group) {
        if (!group.length) return;
        const spacing = 8;
        // sort by proposed left
        group.sort((a,b) => a.left - b.left);
        // forward pass: ensure each starts after prev
        for (let i = 1; i < group.length; i++) {
          const prev = group[i-1];
          const cur = group[i];
          const minLeft = prev.left + prev.width + spacing;
          if (cur.left < minLeft) cur.left = minLeft;
        }
        // backward pass: if last overflows, pull previous leftwards
        const maxRight = wrapper.clientWidth - 8;
        if (group[group.length-1].left + group[group.length-1].width > maxRight) {
          group[group.length-1].left = Math.min(group[group.length-1].left, maxRight - group[group.length-1].width);
          for (let i = group.length - 2; i >= 0; i--) {
            const next = group[i+1];
            const cur = group[i];
            const desiredRight = next.left - spacing;
            if (cur.left + cur.width > desiredRight) {
              cur.left = Math.max(8, desiredRight - cur.width);
            }
          }
        }
        // apply final positions to elements
        group.forEach(item => {
          item.el.style.left = Math.max(8, Math.min(item.left, wrapper.clientWidth - item.width - 8)) + 'px';
        });
      }

      resolveCollisions(topGroup);
      resolveCollisions(bottomGroup);

      // compute arrow Y once (in wrapper coordinates) so connectors and points align exactly
      const ca = chart.chartArea;
      const arrowOffset = (chart.options && chart.options.plugins && chart.options.plugins.horizontalArrow && chart.options.plugins.horizontalArrow.offsetY) ? chart.options.plugins.horizontalArrow.offsetY : 0;
      const arrowMidInCanvas = Math.round((ca.top + ca.bottom) / 2) + arrowOffset;
      const arrowY = (chartRect.top - wrapperRect.top) + arrowMidInCanvas + 28;

      // use dataset points to draw connectors
      const ds = chart.data.datasets[0];
      const meta = chart.getDatasetMeta(0);
      ds.data.forEach(d => {
        // find rendered point element that matches this data index
        const pIndex = meta.data.findIndex(el => el && el.$context && el.$context.raw && el.$context.raw.index === d.index);
        if (pIndex === -1) return;
        const elPoint = meta.data[pIndex];
        if (!elPoint) return;
        const pointX = (chartRect.left - wrapperRect.left) + elPoint.x;

        const snippetEl = snippetsOverlay.querySelector(`.cluster-snippet[data-index="${d.index}"]`);
        if (!snippetEl) return;
        const snippetRect = snippetEl.getBoundingClientRect();
        const toX = snippetRect.left - wrapperRect.left + snippetRect.width / 2;

        // draw to bottom center if the snippet is 'top', otherwise draw to top center
        const isTopSnippet = snippetEl.classList.contains('top');
        const toY = isTopSnippet
          ? (snippetRect.top - wrapperRect.top + snippetRect.height) // bottom center of top snippet
          : (snippetRect.top - wrapperRect.top);                      // top center of bottom snippet

        ctx2.beginPath();
        ctx2.moveTo(pointX, arrowY);
        const midX = (pointX + toX) / 2;
        ctx2.bezierCurveTo(midX, arrowY, midX, toY, toX, toY);
        ctx2.stroke();

        ctx2.beginPath();
        ctx2.arc(toX, toY, 3, 0, Math.PI * 2);
        ctx2.fill();
      });

      // redraw points on overlay so they appear above connectors/arrow — align to arrowY
      ctx2.save();
      ctx2.fillStyle = 'black';
      ctx2.strokeStyle = 'white';
      ctx2.lineWidth = 1.5;
      meta.data.forEach(ptEl => {
        if (!ptEl) return;
        const px = (chartRect.left - wrapperRect.left) + ptEl.x;
        const py = arrowY; // align dot vertically with arrow
        const r = 6;
        ctx2.beginPath();
        ctx2.arc(px, py, r, 0, Math.PI * 2);
        ctx2.fill();
        ctx2.stroke();
      });
      ctx2.restore();
    }

    // clear highlight when not hovering chart area
    chartCanvas.addEventListener('mouseleave', () => {
      document.querySelectorAll('.cluster-snippet.highlight').forEach(el => el.classList.remove('highlight'));
    });

    // update layout on resize/scroll and after images load
    function resizeHandler() {
      overlay.width = wrapper.clientWidth;
      overlay.height = wrapper.clientHeight;
      chart.resize();
      setTimeout(positionSnippetsAndDraw, 60);
    }
    window.addEventListener('resize', resizeHandler);
    window.addEventListener('scroll', positionSnippetsAndDraw, true);

    // wait for images to settle then position
    const imgs = snippetsOverlay.querySelectorAll('img');
    let toLoad = imgs.length;
    if (toLoad === 0) setTimeout(positionSnippetsAndDraw, 80);
    else {
      imgs.forEach(img => {
        if (img.complete) { toLoad--; }
        else {
          img.addEventListener('load', () => { toLoad--; if (toLoad<=0) positionSnippetsAndDraw(); });
          img.addEventListener('error', () => { toLoad--; if (toLoad<=0) positionSnippetsAndDraw(); });
        }
      });
      if (toLoad<=0) setTimeout(positionSnippetsAndDraw, 80);
    }

    // initial draw
    setTimeout(positionSnippetsAndDraw, 120);
  })();
</script>

| Galaxy Cluster | Redshift | Status | Details | 
|---------------|----------|---------|---------|
{% for galaxy_cluster in sorted_clusters -%}
| {{ galaxy_cluster.name }} | {% if galaxy_cluster.redshift and galaxy_cluster.redshift != "" and galaxy_cluster.redshift != "---" %}{{ galaxy_cluster.redshift }}{% else %}N/A{% endif %} | {{ galaxy_cluster.status }}|[View Details]({{ galaxy_cluster.url_zenodo }}) | 
{% endfor %}