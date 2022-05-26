"""python -m dataoverview ... entrypoint."""
import sys

from dataoverview import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
