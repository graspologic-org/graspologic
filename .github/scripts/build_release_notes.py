import requests
import re
import os

action_runs = requests.get("https://api.github.com/repos/Microsoft/graspologic/actions/workflows/publish.yml/runs").json()

# get PR titles
PR_title = []
data = action_runs["workflow_runs"]
for workflow_run in data:
    if workflow_run is not None and workflow_run["conclusion"] == "success":
        title = workflow_run["head_commit"]["message"].split("\n")[0]
        if re.findall("(#.*?)", title):
            PR_title.append(title)

previous_contents = ""

# Open a file: file
if os.path.exists("release.rst"):
    previous = open("release.rst",mode="r")
 
    # read all lines at once
    previous_contents = previous.read()

    # close the file
    previous.close()

    os.remove("release.rst")

# create release.rst
with open("release.rst", "a+") as f:
    for title in PR_title:
        f.write(title + "\n")
    f.write(previous_contents)
