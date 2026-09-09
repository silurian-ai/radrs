---
hide:
  - navigation
  - toc
---

<section class="radrs-hero">
  <img src="assets/radrs-glyph.png" alt="" class="radrs-hero__glyph">
  <h1 class="radrs-hero__title">radrs</h1>
  <p class="radrs-hero__tagline">Fast NEXRAD Level 2 processing for Python.</p>
  <a href="guides/quickstart/" class="radrs-hero__cta">Read the docs</a>
  <div class="radrs-hero__badges">
    <a href="https://pypi.org/project/radrs/"><img src="https://img.shields.io/pypi/v/radrs?logo=pypi&logoColor=ffde57&label=pypi&style=for-the-badge&color=315452" alt="PyPI"></a>
    <a href="https://github.com/silurian-ai/radrs/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-315452?style=for-the-badge" alt="License"></a>
    <a href="https://github.com/silurian-ai/radrs"><img src="https://img.shields.io/github/stars/silurian-ai/radrs?style=for-the-badge&logo=github&color=315452" alt="GitHub stars"></a>
  </div>
  <p class="radrs-hero__attribution">An open source project from <a href="https://silurian.ai">Silurian AI</a></p>
</section>

<div class="radrs-terminal">
  <div class="radrs-terminal__chrome">
    <span class="radrs-terminal__dot radrs-terminal__dot--r"></span>
    <span class="radrs-terminal__dot radrs-terminal__dot--y"></span>
    <span class="radrs-terminal__dot radrs-terminal__dot--g"></span>
    <span class="radrs-terminal__title">quickstart.py</span>
  </div>
  <div class="radrs-terminal__body" markdown>

```python
import radrs.xradar as rxr
import radrs.raystack as rrs

src = "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06"

dt  = rxr.open_datatree(src)                  # xradar-compatible DataTree
rdt = rrs.open_datatree(src, fold_size=128)   # flat raystack DataTree
```

  </div>
</div>
