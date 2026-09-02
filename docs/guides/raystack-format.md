# Raystack format

!!! note "Under construction"

    Reference for the flat raystack layout — `vcps`, `sweeps`, `returns`, and
    `activity` — is being written.

`fold_size` semantics are covered in
[Batch iteration and folding](batching.md#folding): how many returns a radial
becomes, why the `range` coordinate is a gate index rather than a distance, and
how to reconstruct physical range from `base_range` and `range_step`.

See the [`radrs.raystack` API reference](../reference/raystack.md) for the
current API surface.
