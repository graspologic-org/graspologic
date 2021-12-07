# Copied and modified from https://stackoverflow.com/questions/15001888/conditional-toctree-in-sphinx

import re
from sphinx.directives.other import TocTree


def setup(app):
    app.add_config_value('toc_filter_exclude', [], 'html')
    app.add_directive('toctree-filt', TocTreeFilt)
    return {'version': '1.0.0'}

class TocTreeFilt(TocTree):
    """
    Directive to notify Sphinx about the hierarchical structure of the docs,
    and to include a table-of-contents like tree in the current document. This
    version filters the entries based on a list of prefixes. We simply filter
    the content of the directive and call the super's version of run. The
    list of exclusions is stored in the **toc_filter_exclusion** list. Any
    table of content entry prefixed by one of these strings will be excluded.
    If `toc_filter_exclusion=['secret','draft']` then all toc entries of the
    form `:secret:ultra-api` or `:draft:new-features` will be excuded from
    the final table of contents. Entries without a prefix are always included.
    """
    hasPat = re.compile('\s*(.*)$')

    # Remove any entries in the content that we dont want and strip
    # out any filter prefixes that we want but obviously don't want the
    # prefix to mess up the file name.
    def filter_entries(self, entries):
        excl = self.state.document.settings.env.config.toc_filter_exclude
        filtered = []
        for e in entries:
            m = self.hasPat.match(e)
            if m != None:
                if not m.groups()[0] in excl:
                    filtered.append(m.groups()[0])
            else:
                filtered.append(e)
        return filtered

    def run(self):
        # Remove all TOC entries that should not be on display
        self.content = self.filter_entries(self.content)
        return super().run()
