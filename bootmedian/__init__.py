"""Bootmedian package

Expose the public API from :mod:`bootmedian.main` for convenient imports:

```py
from bootmedian import bootmedian, bootfit
```
"""

__version__ = "1.0.0"
__author__ = "Alejandro S. Borlaff"
__author_email__ = "a.s.borlaff@nasa.gov"
__description__ = "A software to estimate the median using bootstrapping"
__url__ = "https://github.com/Borlaff/bootmedian"

from .bootmedian import (
	bootmedian,
	bootfit,
	boot_polyfit,
	bootstrap_resample,
	median_bootstrap,
	mean_bootstrap,
	sum_bootstrap,
	std_bootstrap,
)

__all__ = [
	"bootmedian",
	"bootfit",
	"boot_polyfit",
	"bootstrap_resample",
	"median_bootstrap",
	"mean_bootstrap",
	"sum_bootstrap",
	"std_bootstrap",
]
