---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

## Galaxy Clusters

{% assign sorted_clusters = site.galaxy_clusters | sort: 'redshift' %}

| Galaxy Cluster | Redshift | Status | Details |
|---------------|----------|---------|---------|
{% for galaxy_cluster in sorted_clusters -%}
| {{ galaxy_cluster.name }} | {% if galaxy_cluster.redshift and galaxy_cluster.redshift != "" and galaxy_cluster.redshift != "---" %}{{ galaxy_cluster.redshift }}{% else %}N/A{% endif %} | {{ galaxy_cluster.status }}|[View Details]({{ site.baseurl }}{{ galaxy_cluster.url }}) |
{% endfor %}