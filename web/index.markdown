---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

This is the main page of the website

{% for galaxy_cluster in site.galaxy_clusters %}
  <h2>{{ galaxy_cluster.name }} - {{ galaxy_cluster.redshift }}</h2>
  <p>{{ galaxy_cluster.content | markdownify }}</p>
{% endfor %}