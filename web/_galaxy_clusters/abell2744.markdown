---
name: "Abell 2744"
redshift: 0.308
status: "Available"
layout: page
url_zenodo: https://zenodo.org/records/17089402?preview=1
---

# Abell 2744

Corresponding author: Bill Harris

[Paper I](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.2696H/abstract)

[Paper II](https://ui.adsabs.harvard.edu/abs/2024ApJ...971..155H/abstract)

## Basic Information
- **Redshift**: {{ page.redshift }}
- **Status**: {{ page.status }}
- **References to be cited**:  [Harris & Reina-Campos (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.2696H/abstract) and [Harris & Reina-Campos (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...971..155H/abstract)

## Proprietary Data Access

<div id="password-section">
<p><strong>Proprietary research files require authentication.</strong></p>
<button onclick="checkPassword()" class="auth-button">Access Proprietary Files</button>
</div>

<div id="protected-content" style="display: none;">
<h3>🔒 Proprietary Files</h3>
<ul>
<li><a href="{{ site.baseurl }}/data/abell2744/jwst_raw_observations.fits" target="_blank">Photometric GC catalogues - v1</a></li>
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