
"""
Build script that will check some environment variables set by Github during build.
See: https://help.github.com/en/actions/configuring-and-managing-workflows/using-environment-variables#default-environment-variables
If the branch is master or dev, it will generate the appropriate content for version.txt. If it isn't, it returns nothing.  Note that the version.txt file is set by bash, not by this file.
"""

import os
from graspologic.version.version import __semver

ref_key = "GITHUB_REF"
run_id_key = "GITHUB_RUN_ID"
prerelease_ref = "refs/heads/dev"
release_ref = "refs/heads/main"
publish_refs = [prerelease_ref, release_ref]

environ = os.environ

if ref_key in environ and run_id_key in environ:
    ref = environ[ref_key]
    run_id = environ[run_id_key]
    if ref == prerelease_ref:
        print(f"{__semver}.dev{run_id}")
    elif ref == release_ref:
        print(f"{__semver}")
