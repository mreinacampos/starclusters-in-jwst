---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

## Galaxy Clusters

{% assign sorted_clusters = site.galaxy_clusters | sort: 'redshift' %}

<!-- Timeline + preview -->
<style>
  /* small, self-contained styles — move to SCSS if desired */
  .timeline-wrapper { display:flex; gap:1.25rem; align-items:flex-start; margin-bottom:1.5rem; flex-wrap:wrap; }
  .timeline-canvas { flex:1 1 540px; min-width:320px; max-width:860px; background:transparent; padding:0.5rem; }
  .cluster-preview { width:240px; min-width:200px; background:rgba(255,255,255,0.02); border-radius:6px; padding:0.75rem; color:inherit; }
  .cluster-preview img { width:100%; height:auto; display:block; border-radius:4px; margin-bottom:0.5rem; object-fit:cover; background:#0b1720; }
  .cluster-meta { font-size:0.92rem; line-height:1.25; }
  .cluster-meta b { display:block; margin-bottom:0.25rem; }
  @media (max-width:720px){ .timeline-wrapper { flex-direction:column; } .cluster-preview{ width:100%; } }
</style>

<div class="timeline-wrapper">
  <div class="timeline-canvas">
    <canvas id="redshiftTimeline" aria-label="Galaxy cluster redshift timeline" role="img"></canvas>
  </div>

  <aside class="cluster-preview" id="clusterPreview">
    <img id="previewImage" src="{{ sorted_clusters.first.image | default: sorted_clusters.first.jwst_image | default: sorted_clusters.first.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}" alt="Cluster preview" />
    <div class="cluster-meta" id="previewMeta">
      <b id="previewTitle">{{ sorted_clusters.first.name }}</b>
      <div id="previewRedshift">Redshift: {% if sorted_clusters.first.redshift and sorted_clusters.first.redshift != "" and sorted_clusters.first.redshift != "---" %}{{ sorted_clusters.first.redshift }}{% else %}N/A{% endif %}</div>
      <div id="previewStatus">Status: {{ sorted_clusters.first.status }}</div>
      <div style="margin-top:.5rem;"><a id="previewLink" href="{{ sorted_clusters.first.url_zenodo | default: sorted_clusters.first.url | relative_url }}" target="_blank" rel="noopener">View details</a></div>
    </div>
  </aside>
</div>

<!-- Data object for JS: safe defaults; parse redshift in JS -->
<script>
  const clusters = [
{% for g in sorted_clusters %}
    {
      name: "{{ g.name | escape }}",
      redshift: "{{ g.redshift | default: '' }}",
      status: "{{ g.status | escape }}",
      image: "{{ g.image | default: g.jwst_image | default: g.thumbnail | default: '/assets/images/placeholder.png' | relative_url }}",
      zenodo: "{{ g.url_zenodo | default: '' }}",
      permalink: "{{ g.url | absolute_url }}"
    }{% unless forloop.last %},{% endunless %}
{% endfor %}
  ];
</script>

<!-- Chart.js from CDN (stable major) -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
  (function(){
    // Convert clusters to data points (skip invalid redshifts)
    const points = clusters.map((c, i) => {
      const z = parseFloat(String(c.redshift).replace(',', '.'));
      return { x: isFinite(z) ? z : null, y: 0, index: i, label: c.name, image: c.image, zenodo: c.url_zenodo, permalink: c.permalink, status: c.status, rawRedshift: c.redshift };
    }).filter(p => p.x !== null);

    // Find axis bounds
    const zs = points.map(p => p.x);
    const minZ = zs.length ? Math.min(...zs) : 0;
    const maxZ = zs.length ? Math.max(...zs) : 1;

    // Build Chart
    const ctx = document.getElementById('redshiftTimeline').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Galaxy clusters (redshift)',
          data: points,
          backgroundColor: '#f39c12',
          borderColor: '#ffffff55',
          pointRadius: 8,
          pointHoverRadius: 12,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 10, right: 10, bottom: 10, left: 10 } },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'Redshift (z)' },
            min: Math.max(0, minZ - (maxZ - minZ) * 0.08),
            max: maxZ + (maxZ - minZ) * 0.08
          },
          y: { display: false }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: false // we use an external hover preview
          }
        },
        onHover(evt, active) {
          // If hovering a point, update preview
          if (active && active.length) {
            const element = active[0];
            const datasetIndex = element.datasetIndex;
            const idx = element.index;
            const point = chart.data.datasets[datasetIndex].data[idx];
            updatePreview(point);
          }
        },
        onClick(evt, activeEls) {
          if (activeEls && activeEls.length) {
            const el = activeEls[0];
            const p = chart.data.datasets[el.datasetIndex].data[el.index];
            const url = p.url_zenodo || p.permalink || '#';
            window.open(url, '_blank', 'noopener');
          }
        }
      });

    // Find first cluster to initialize preview (prefer a numeric redshift)
    function initializePreview() {
      if (points.length) {
        updatePreview(points[0]);
      } else if (clusters.length) {
        // fallback: use first cluster from front matter
        const c = clusters[0];
        updatePreview({ label: c.name, image: c.image, zenodo: c.url_zenodo, permalink: c.permalink, rawRedshift: c.redshift, status: c.status });
      }
    }

    function updatePreview(point) {
      const title = document.getElementById('previewTitle');
      const img = document.getElementById('previewImage');
      const redshift = document.getElementById('previewRedshift');
      const status = document.getElementById('previewStatus');
      const link = document.getElementById('previewLink');

      title.textContent = point.label || 'Unknown';
      img.src = point.image || '/assets/images/placeholder.png';
      img.alt = (point.label || 'Cluster') + ' preview';
      redshift.textContent = 'Redshift: ' + (point.rawRedshift !== undefined && point.rawRedshift !== "" ? point.rawRedshift : 'N/A');
      status.textContent = 'Status: ' + (point.status || '—');
      link.href = point.url_zenodo || point.permalink || '#';
    }

    // init
    initializePreview();
  })();
</script>

| Galaxy Cluster | Redshift | Status | Details | 
|---------------|----------|---------|---------|
{% for galaxy_cluster in sorted_clusters -%}
| {{ galaxy_cluster.name }} | {% if galaxy_cluster.redshift and galaxy_cluster.redshift != "" and galaxy_cluster.redshift != "---" %}{{ galaxy_cluster.redshift }}{% else %}N/A{% endif %} | {{ galaxy_cluster.status }}|[View Details]({{ galaxy_cluster.url_zenodo }}) | 
{% endfor %}