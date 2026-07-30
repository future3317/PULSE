# CrossPiezo Tensor Conventions

## Default internal representation

- **Quantity**: piezoelectric stress tensor `e`.
- **Shape**: full Cartesian `3 × 3 × 3`.
- **Symmetry**: last two indices symmetric (`e_{ijk} = e_{ikj}`).
- **Unit**: `C/m^2`.
- **Internal engineering Voigt order**: `xx, yy, zz, yz, xz, xy`.
- **Engineering shear**: off-diagonal Voigt components equal twice the
  tensor-shear component.

## Transformations recorded for every tensor

- source Voigt order;
- internal Voigt order;
- engineering shear / tensor shear convention;
- unit;
- stress (`e`) vs. strain (`d`) identity;
- Cartesian expansion;
- cell / lattice basis;
- atom mapping;
- point-group action and symmetry projection;
- source functional and code version.

## Rotation rule

For a matched pair, the MP Cartesian tensor is rotated into the JARVIS frame
using the polar-decomposition rotation obtained from the structure matcher:

```
e'_{ijk} = R_{il} R_{jm} R_{kn} e_{lmn}
```

This assumes the Cartesian tensor axes are tied to the CIF lattice axes.
Source-standard orientations (e.g. IEEE) may require additional alignment.
