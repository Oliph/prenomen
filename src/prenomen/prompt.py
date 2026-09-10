"""Loading the gender-inference prompt template.

The template text is frozen: it differs from the prompt used for the published
results by whitespace only, and ``tests/fixtures/*.txt`` pins the rendered output
byte-for-byte. Change ``templates/gender_prompt.j2``, or the variables passed to
``render``, and those results are no longer comparable.

The render call takes exactly four variables:

    template.render(first_name=..., last_name=...,
                    dict_matches=[m for m in matches if m.found],
                    countries=<sorted upper-case tuple>)
"""

import os

from jinja2 import Environment, FileSystemLoader

TEMPLATE_NAME = "gender_prompt.j2"


def build_prompt_template(template_dir: str | os.PathLike | None = None):
    """Load the Jinja prompt template, from the package unless overridden."""
    if template_dir is None:
        template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
    env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
    )
    return env.get_template(TEMPLATE_NAME)
