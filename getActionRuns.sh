curl \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/Microsoft/graspologic/actions/workflows/publish.yml/runs" > "actionRuns.json"
