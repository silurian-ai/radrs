# Changelog

## [0.3.7](https://github.com/silurian-ai/radrs/compare/radrs-v0.3.6...radrs-v0.3.7) (2026-04-27)


### Bug Fixes

* Update source URL in quick start ([#39](https://github.com/silurian-ai/radrs/issues/39)) ([c4812d7](https://github.com/silurian-ai/radrs/commit/c4812d7633447031ef78e6ea780871d7f93d9f26))

## [0.3.6](https://github.com/silurian-ai/radrs/compare/radrs-v0.3.5...radrs-v0.3.6) (2026-04-27)


### Features

* accept gs:// and az:// URIs in open_datatree ([#31](https://github.com/silurian-ai/radrs/issues/31)) ([bd878ef](https://github.com/silurian-ai/radrs/commit/bd878ef1f4412a9266a7590237d64e8777811f51))

## [0.3.5](https://github.com/silurian-ai/radrs/compare/radrs-v0.3.4...radrs-v0.3.5) (2026-04-24)


### Features

* Add batched raystack loader and time-based iter ([1076bc0](https://github.com/silurian-ai/radrs/commit/1076bc0ec8e160c8b4562ec20f164669e3ba9169))
* add instrument_name ([49bacdc](https://github.com/silurian-ai/radrs/commit/49bacdc4fff976cfcd4851e9c3f581caffa5b25c))
* Add license and update publish action ([#21](https://github.com/silurian-ai/radrs/issues/21)) ([941ec3a](https://github.com/silurian-ai/radrs/commit/941ec3a0604486fa35d27deae87d0519685f782d))
* Add metadata-only batch functionality (with peek) ([#9](https://github.com/silurian-ai/radrs/issues/9)) ([6d6e87f](https://github.com/silurian-ai/radrs/commit/6d6e87fd487fc5dbed2fe380968fcd96b2e9be7d))
* add raystack quicklook anywidget app ([#13](https://github.com/silurian-ai/radrs/issues/13)) ([677fe84](https://github.com/silurian-ai/radrs/commit/677fe845dfa177bc07a96c6f091de826beb9e6e3))
* add root metadata and use azimuth as primary dimension ([#2](https://github.com/silurian-ai/radrs/issues/2)) ([6afb25d](https://github.com/silurian-ai/radrs/commit/6afb25d1240a2da88b6ece3822c05fc68bd1b06b))
* allow rust lib logging for debugging ([95e3e2c](https://github.com/silurian-ai/radrs/commit/95e3e2ce8181417b10bd0dc62f8a0ef373747c6a))
* be more robust against bad data in archive ([cf36a05](https://github.com/silurian-ai/radrs/commit/cf36a059ccab10fcca65247af0047db93cfe4dc0))
* faster filtering and other analysis tools ([#1](https://github.com/silurian-ai/radrs/issues/1)) ([ac5794d](https://github.com/silurian-ai/radrs/commit/ac5794db3b4e7c42f638e0dbd6dd5f50f3043ae3))
* Folding logic for batched raystacks ([#8](https://github.com/silurian-ai/radrs/issues/8)) ([c2516dd](https://github.com/silurian-ai/radrs/commit/c2516dd90ad9db963575e4d7e55747e792305979))
* high-performance S3 fetch with connection pooling and prefetch ([687d8b4](https://github.com/silurian-ai/radrs/commit/687d8b4051561951cc8cc28c18c85676e3851611))
* Make radrs compatible with current silurian library versions ([#14](https://github.com/silurian-ai/radrs/issues/14)) ([de2dc36](https://github.com/silurian-ai/radrs/commit/de2dc369c5d07eeba19a4d87a4bdb5076256149f))
* Modify Build/Package Setup ([#10](https://github.com/silurian-ai/radrs/issues/10)) ([4a06326](https://github.com/silurian-ai/radrs/commit/4a06326534181d73c86a80d43f24a9fa9ce0b6c1))
* prepare for release ([7b50583](https://github.com/silurian-ai/radrs/commit/7b505832a18af5839014fd244905293f0959264e))
* **qc:** add VRADH winding number for velocity dealiasing ([b044a57](https://github.com/silurian-ai/radrs/commit/b044a5705a094289f14a040200b3adca0b2ea8ac))
* some useful scripts and analysis notebooks ([#3](https://github.com/silurian-ai/radrs/issues/3)) ([192e7c1](https://github.com/silurian-ai/radrs/commit/192e7c16ec214037120aae0edefdf3bb2d444034))
* wire QC support into volume iterators ([7c8514d](https://github.com/silurian-ai/radrs/commit/7c8514d349ea8e01904646b809d0f9a521296f34))


### Bug Fixes

* adapt to upstream refactor ([#17](https://github.com/silurian-ai/radrs/issues/17)) ([b6c33af](https://github.com/silurian-ai/radrs/commit/b6c33af5fd5e9ee3fd81526ed50049cb0859c1cf))
* Add fix and test for exact-boundary edge case ([#16](https://github.com/silurian-ai/radrs/issues/16)) ([89986d3](https://github.com/silurian-ai/radrs/commit/89986d381f45ad4fb4d8c538aba34e564e047e80))
* bump version to 0.3.4 ([856ed57](https://github.com/silurian-ai/radrs/commit/856ed57be071b2837b274e91140754422b06771c))
* clearer function names ([a260ae1](https://github.com/silurian-ai/radrs/commit/a260ae15512bf633888d1ad0f2edf15865824ac8))
* correct rust setup ([dbbc013](https://github.com/silurian-ai/radrs/commit/dbbc013a97c05937b05573bfeaf18cef917b89e8))
* enough tests to know things are working decently ([ddf6054](https://github.com/silurian-ai/radrs/commit/ddf6054de7aac7dafb75fde8d7fd297af2a1db6d))
* introduce volumesource to allow later extension to alternative sources of radar data ([3c3acd1](https://github.com/silurian-ai/radrs/commit/3c3acd1c5a26fc8a42e919a1e376f4a2c170394c))
* issue with pyart parity testing ([ab63e6c](https://github.com/silurian-ai/radrs/commit/ab63e6c68225bf4b398509c0457442281e5273b7))
* keep ordering when in parallel ([5b5019e](https://github.com/silurian-ai/radrs/commit/5b5019e834c217530f34f750c44dbc00b362ecd7))
* update release config, add CI caching, and update nexrad dependencies ([#23](https://github.com/silurian-ai/radrs/issues/23)) ([02cfe65](https://github.com/silurian-ai/radrs/commit/02cfe65820e4614a95681ba4bda19b920717953a))
* use generic updater for Cargo.toml in release-please ([#25](https://github.com/silurian-ai/radrs/issues/25)) ([d963fd9](https://github.com/silurian-ai/radrs/commit/d963fd94212ca26346fc5ddd20e08ca8c72c97d1))
* use rust release type for release-please ([610f638](https://github.com/silurian-ai/radrs/commit/610f63822e98900ac639fb2875d67d40ec7fbcd5))


### Performance Improvements

* **qc:** parallelize vradh winding texture computation ([f76636b](https://github.com/silurian-ai/radrs/commit/f76636b832d57ea503829a39c5f11222c4dc7ac6))

## [0.3.5](https://github.com/silurian-ai/radrs/compare/v0.3.4...v0.3.5) (2026-04-24)


### Bug Fixes

* use generic updater for Cargo.toml in release-please ([#25](https://github.com/silurian-ai/radrs/issues/25)) ([d963fd9](https://github.com/silurian-ai/radrs/commit/d963fd94212ca26346fc5ddd20e08ca8c72c97d1))

## [0.3.4](https://github.com/silurian-ai/radrs/compare/v0.3.3...v0.3.4) (2026-04-24)


### Features

* Add license and update publish action ([#21](https://github.com/silurian-ai/radrs/issues/21)) ([941ec3a](https://github.com/silurian-ai/radrs/commit/941ec3a0604486fa35d27deae87d0519685f782d))


### Bug Fixes

* adapt to upstream refactor ([#17](https://github.com/silurian-ai/radrs/issues/17)) ([b6c33af](https://github.com/silurian-ai/radrs/commit/b6c33af5fd5e9ee3fd81526ed50049cb0859c1cf))
* update release config, add CI caching, and update nexrad dependencies ([#23](https://github.com/silurian-ai/radrs/issues/23)) ([02cfe65](https://github.com/silurian-ai/radrs/commit/02cfe65820e4614a95681ba4bda19b920717953a))
