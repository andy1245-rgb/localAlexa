"""Allow ``python -m localalexa`` to behave like the console script."""

from .cli import main

raise SystemExit(main())
