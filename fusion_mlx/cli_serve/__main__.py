# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m fusion_mlx.cli_serve``.

Delegates to the canonical argparse dispatcher in fusion_mlx.cli so
``python -m fusion_mlx.cli_serve <subcommand>`` works — this package only
hosts the serve/bench handlers, not the parser. One parser, not two.
"""

from fusion_mlx.cli import main

if __name__ == "__main__":
    main()
