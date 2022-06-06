import subprocess
import json

subprocess.call(['sh', './getActionRuns.sh'])

# get PR titles
PR_title = []
with open('actionRuns.json', 'r') as f:
    data = json.load(f)["workflow_runs"]
    for workflow_run in data:
        if workflow_run is not None and workflow_run["conclusion"] == "success":
            PR_title.append(workflow_run["head_commit"]["message"])

# create release.rst
with open('release.rst', 'a+') as f:
    for title in PR_title:
        f.write(title + "\n")