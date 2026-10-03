"""Sanity checks against the published release numbers (need downloaded data)."""
import pytest

from flybrainlab.connectome import GroupResolver, load_connectome
from flybrainlab.connectome.loaders import sanity_check

from .conftest import have_data


@pytest.mark.data
@pytest.mark.parametrize("ds", ["malecns_v1.0", "flywire_v783"])
def test_release_counts(ds):
    if not have_data(ds):
        pytest.skip(f"{ds} not downloaded")
    c = load_connectome(ds)
    res = sanity_check(c)
    assert res["ok"], res["checks"]
    g = GroupResolver(c)
    for name in ("giant_fiber", "lplc2", "mn9", "sugar_grn", "dna02_L", "dna02_R", "jo_ce"):
        assert len(g.resolve(name)) > 0, name
