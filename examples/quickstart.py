"""One name in, one result out.

    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

from prenomen import Prenomen

# Building this is the expensive part -- with the real dictionaries it costs minutes
# and gigabytes -- so build it once and reuse it for every name.
# `prenomen-dicts get all` downloads the dictionaries; without them every name goes
# to the model, which still works but is slower and less accurate.
p = Prenomen(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    dictionaries=[],  # e.g. ["wgnd"], or prenomen.ALL_DICTIONARIES
)

result = p.infer("Marie", "Curie", countries=["FR", "PL"])

print(f"{result.first_name} {result.last_name}: {result.gender.value}")
print(f"  confidence  {result.confidence:.2f}")
print(f"  decided by  {result.decided_by.value}")
print(f"  because     {result.reasoning_code.value if result.reasoning_code else '-'}")
for match in result.dict_matches:
    if match.found:
        print(f"  {match.source}: {match.gender.value} ({match.frequency:.2f})")
