# Continuous Integration for `graspologic`

Microsoft requires - and it's simply good policy - that we run continuous scans
of our builds checking for any accidental commits of credentials, as well as 
keeping track of all of our dependencies, looking for security issues as well 
as any legal concerns.

However, using Azure DevOps for our entire build restricts visibility into the
steps we require be completed for a viable build.  This makes processing
PRs a Microsoft-first process, instead of enabling our wonderful contributing
community to be able to see the actual build failures as they occur, instead of
relying on a Microsoft employee to convey the error conditions to them.

To achieve this, we broke our CI process into two primary processes:
1. Community build (via Github Actions)
2. Compliance and security build (via Azure DevOps)

We want to ensure we're up front about all our processes, so these build specification
files are committed into the repository for community inspection.

Of the steps included in `Compliance and Security` (Component Governance and Credential Scan),
the only part that should be possible to fail a build is Credential Scan. 

Unfortunately, this will require an approved Microsoft employee to determine the actual
feedback around it, but hopefully this isolation of black box failures to an individual
task will help mitigate most problems.
