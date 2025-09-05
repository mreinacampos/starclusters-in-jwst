---
name: "Abell 370"
redshift: "---"
status: "in progress"
layout: page
---

# Abell 370

Abell 370 is a massive galaxy cluster acting as a gravitational lens.

Corresponding author: Bill Harris

## Basic Information
- **Redshift**: Not available
- **Status**: {{ page.status }}

## Public Data

- Published lensing models
- Hubble Frontier Fields data

## Proprietary Data Access

<div id="password-section">
<p><strong>Proprietary research files require authentication.</strong></p>
<button onclick="checkPassword()" class="auth-button">Access Proprietary Files</button>
</div>

<div id="protected-content" style="display: none;">
<h3>🔒 Proprietary Files</h3>
<ul>
<li><a href="{{ site.baseurl }}/data/abell370/lensing_reconstruction.fits" target="_blank">Lensing Reconstruction Models</a></li>
<li><a href="{{ site.baseurl }}/data/abell370/background_galaxy_catalog.csv" target="_blank">Background Galaxy Catalog</a></li>
<li><a href="{{ site.baseurl }}/data/abell370/mass_profile_analysis.py" target="_blank">Mass Profile Scripts</a></li>
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