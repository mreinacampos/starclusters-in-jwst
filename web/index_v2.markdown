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
    display:flex;
    gap:1rem;
    align-items:flex-start;
    margin-bottom:1.5rem;
    position:relative; /* for overlay canvas */
  }

  .timeline-column {
    flex:1 1 640px;
    min-width:320px;
    max-width:860px;
    background:transparent;
    padding:0.5rem;
    box-sizing:border-box;
    position:relative;
  }

  .timeline-column canvas.chart-canvas {
    width:100%;
    height:160px; /* short horizontal band */
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
  }

  .snippets-column {
    width:320px;
    min-width:220px;
    max-width:320px;
    padding:0.5rem;
    box-sizing:border-box;
    display:flex;
    flex-direction:column;
    gap:0.75rem;
  }

  /* split top & bottom stacks */
  .snippets-top,
  .snippets-bottom {
    display:flex;
    flex-direction:column;
    gap:0.5rem;
  }

  .snippets-top { align-items:flex-start; margin-bottom:0.75rem; }
  .snippets-bottom { align-items:flex-start; margin-top:0.75rem; }

  .cluster-snippet {
    display:flex;
    gap:0.75rem;
    align-items:center;
    padding:0.45rem 0.5rem;
    background:rgba(255,255,255,0.02);
    border-radius:6px;
    min-height:40px;
    transition: transform 160ms ease, box-shadow 160ms ease;
    transform-origin: left center;
  }

  .cluster-snippet a.name {
    color:inherit;
    text-decoration:none;
    font-weight:600;
    font-size:0.95rem;
  }

  .cluster-snippet.highlight {
    transform: scale(1.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    z-index: 5;
  }

  /* we only show name (link) as requested */
  .cluster-snippet img { display:none; }
  .cluster-snippet .meta { display:block; }

  @media (max-width:920px){
    .timeline-wrapper{ flex-direction:column; }
    .snippets-column{ width:100%; flex-direction:row; gap:1rem; }
    .snippets-top, .snippets-bottom { flex:1; }
  }
</style>

<div class="timeline-wrapper" id="timelineWrapper">
  <div class="timeline-column">
    <canvas id="redshiftTimeline" class="chart-canvas" aria-label="Galaxy cluster redshift timeline" role="img"></canvas>
    <canvas id="timelineOverlay" class="timeline-overlay" aria-hidden="true"></canvas>
  </div>

  <div class="snippets-column" id="snippetsColumn">
    <div class="snippets-top" id="snippetsTop">
      {% for g in sorted_clusters %}
        {% if forloop.index0 | modulo: 2 == 0 %}
        <div class="cluster-snippet" data-index="{{ forloop.index0 }}">
          <div class="meta"><a class="name" href="{{ g.image | default: g.jwst_image | default: g.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}" target="_blank" rel="noopener">{{ g.name | escape }}</a></div>
        </div>
        {% endif %}
      {% endfor %}
    </div>

    <div class="snippets-bottom" id="snippetsBottom">
      {% for g in sorted_clusters %}
        {% if forloop.index0 | modulo: 2 != 0 %}
        <div class="cluster-snippet" data-index="{{ forloop.index0 }}">
          <div class="meta"><a class="name" href="{{ g.image | default: g.jwst_image | default: g.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}" target="_blank" rel="noopener">{{ g.name | escape }}</a></div>
        </div>
        {% endif %}
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
    // map clusters to plottable points: x = redshift, y = 0 (single horizontal axis)
    const points = clusters.map(c => {
      const z = parseFloat(String(c.redshift).replace(',', '.'));
      return { x: isFinite(z) ? z : null, y: 0, index: c.index, label: c.name, image: c.image, zenodo: c.zenodo, permalink: c.permalink, rawRedshift: c.redshift };
    });

    const plottable = points.filter(p => p.x !== null);

    // compute x axis bounds with padding (handle single-value case)
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

    const wrapper = document.getElementById('timelineWrapper');
    const chartCanvas = document.getElementById('redshiftTimeline');
    const overlay = document.getElementById('timelineOverlay');
    const snippetsColumn = document.getElementById('snippetsColumn');

    // create Chart.js scatter with all y=0 so dots lie on a single horizontal axis
    chartCanvas.style.height = '160px';
    chartCanvas.height = 160;

    const ctx = chartCanvas.getContext('2d');

    const dataset = {
      label: 'Galaxy clusters (redshift)',
      data: plottable.map(p => ({ x: p.x, y: 0, index: p.index, label: p.label, image: p.image, permalink: p.permalink, rawRedshift: p.rawRedshift })),
      backgroundColor: 'black',
      borderColor: 'black',
      pointRadius: 7,
      pointHoverRadius: 11,
    };

    const chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: [ dataset ] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'Redshift (z)' },
            min: minZ,
            max: maxZ
          },
          y: {
            display: false,
            min: -1,
            max: 1
          }
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        onHover(evt, elements) {
          // highlight snippet when hovering a dot
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
            const url = p.permalink || '#';
            window.open(url, '_blank', 'noopener');
          }
        }
      }
    });

    // overlay connectors: draw from point (on chart) to snippet center
    function drawConnectors() {
      const ctx2 = overlay.getContext('2d');
      overlay.width = wrapper.clientWidth;
      overlay.height = wrapper.clientHeight;
      overlay.style.width = overlay.width + 'px';
      overlay.style.height = overlay.height + 'px';
      ctx2.clearRect(0, 0, overlay.width, overlay.height);
      ctx2.lineWidth = 1;
      ctx2.strokeStyle = 'rgba(0,0,0,0.9)';
      ctx2.fillStyle = 'rgba(0,0,0,0.9)';

      const wrapperRect = wrapper.getBoundingClientRect();
      const chartRect = chartCanvas.getBoundingClientRect();

      chart.update(); // ensure elements positions available

      clusters.forEach((c) => {
        // find plotted point for this cluster index
        const ds = chart.data.datasets[0];
        const pIndex = ds.data.findIndex(d => d.index === c.index);
        if (pIndex === -1) return; // cluster not plotted (no numeric redshift)
        const el = chart.getDatasetMeta(0).data[pIndex];
        if (!el) return;

        // point location in wrapper coords
        const pointX = (chartRect.left - wrapperRect.left) + el.x;
        const pointY = (chartRect.top - wrapperRect.top) + el.y;

        // find snippet element
        const snippetEl = document.querySelector(`.cluster-snippet[data-index="${c.index}"]`);
        if (!snippetEl) return;
        const snippetRect = snippetEl.getBoundingClientRect();
        // target location: left edge of snippets column + small offset, vertical center of snippet
        const toX = snippetRect.left - wrapperRect.left + 8;
        const toY = snippetRect.top - wrapperRect.top + snippetRect.height / 2;

        // draw gentle curve
        ctx2.beginPath();
        ctx2.moveTo(pointX, pointY);
        const midX = (pointX + toX) / 2;
        ctx2.bezierCurveTo(midX, pointY, midX, toY, toX, toY);
        ctx2.stroke();

        // end dot
        ctx2.beginPath();
        ctx2.arc(toX, toY, 3, 0, Math.PI * 2);
        ctx2.fill();
      });
    }

    // clear highlight when not hovering chart area
    chartCanvas.addEventListener('mouseleave', () => {
      document.querySelectorAll('.cluster-snippet.highlight').forEach(el => el.classList.remove('highlight'));
    });

    // redraw connectors when layout changes
    function resizeHandler() {
      // sync overlay and redraw
      overlay.width = wrapper.clientWidth;
      overlay.height = wrapper.clientHeight;
      overlay.style.width = overlay.width + 'px';
      overlay.style.height = overlay.height + 'px';
      chart.resize();
      setTimeout(drawConnectors, 40);
    }

    window.addEventListener('resize', resizeHandler);
    window.addEventListener('scroll', drawConnectors, true);

    // redraw after images load (if any) and DOM settle
    setTimeout(drawConnectors, 120);
  })();
</script>

| Galaxy Cluster | Redshift | Status | Details | 
|---------------|----------|---------|---------|
{% for galaxy_cluster in sorted_clusters -%}
| {{ galaxy_cluster.name }} | {% if galaxy_cluster.redshift and galaxy_cluster.redshift != "" and galaxy_cluster.redshift != "---" %}{{ galaxy_cluster.redshift }}{% else %}N/A{% endif %} | {{ galaxy_cluster.status }}|[View Details]({{ galaxy_cluster.url_zenodo }}) | 
{% endfor %}