---
name: "Bullet Cluster"
redshift: 0.296
status: "in progress"
layout: page
---

# Bullet Cluster

The Bullet Cluster (1E 0657-558) is a famous example of colliding galaxy clusters, providing strong evidence for dark matter.

Corresponding author: Bill Harris

## Basic Information
- **Redshift**: {{ page.redshift }}
- **Status**: {{ page.status }}
- **Alternative Names**: 1E 0657-558

## Public Data

- Chandra X-ray observations
- Hubble Space Telescope imaging
- Weak lensing mass maps

## Proprietary Data Access

<div id="password-section">
<p><strong>Proprietary research files require authentication.</strong></p>
<button onclick="checkPassword()" class="auth-button">Access Proprietary Files</button>
</div>

<div id="protected-content" style="display: none;">
<h3>🔒 Proprietary Files</h3>
<ul>
<li><a href="{{ site.baseurl }}/data/bullet_cluster/jwst_nircam_deep.fits" target="_blank">JWST NIRCam Deep Imaging</a></li>
<li><a href="{{ site.baseurl }}/data/bullet_cluster/dark_matter_reconstruction.fits" target="_blank">Dark Matter Mass Map</a></li>
<li><a href="{{ site.baseurl }}/data/bullet_cluster/collision_dynamics_model.py" target="_blank">Collision Dynamics Scripts</a></li>
<li><a href="{{ site.baseurl }}/data/bullet_cluster/shock_front_analysis.csv" target="_blank">Shock Front Analysis</a></li>
</ul>
<p><em>Note: These are placeholder links for demonstration. Actual files require <a href="{{ site.baseurl }}/request-access/">formal access request</a>.</em></p>
</div>

<script>
function checkPassword() {
    const password = prompt("Enter password to access proprietary files:");
    if (password === "MRC") {
        document.getElementById("protected-content").style.display = "block";
        document.getElementById("password-section").innerHTML = "<p><em>✅ Authentication successful. Proprietary files are now visible below.</em></p>";
    } else if (password !== null) {
        alert("Incorrect password. Please contact the research team for access.");
    }
}
</script>

<style>
.auth-button {
    background-color: #0366d6;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
}
.auth-button:hover {
    background-color: #0256cc;
}
#protected-content {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 16px;
    margin-top: 16px;
}
</style>

---

**Data Access**: For access to proprietary files, please [submit a request]({{ site.baseurl }}/request-access/).