"""
Load Claude-style skills (a SKILL.md with YAML frontmatter) so the same skill
can drive an agent on any provider.

    ---
    name: changelog
    description: Write a changelog from recent git history
    ---
    # Instructions ...
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    # Markdown body, frontmatter removed.
    body: str
    # Directory holding SKILL.md (and any helper scripts / reference files).
    path: Path
    metadata: dict[str, str] = field(default_factory=dict)

    def render(self, **variables: str) -> str:
        """Body with $NAME placeholders replaced (e.g. $ARGUMENTS)."""
        text = self.body
        for key, value in variables.items():
            text = text.replace(f"${key}", str(value))
        return text


def load_skill(path: str | Path) -> Skill:
    """Load a skill from a SKILL.md file or a directory containing one."""
    path = Path(path)
    skill_file = path / "SKILL.md" if path.is_dir() else path
    text = skill_file.read_text()

    metadata: dict[str, str] = {}
    body = text
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            # Flat "key: value" frontmatter is all skills use; no YAML dependency.
            for line in lines[1:end]:
                if ":" in line and not line.startswith((" ", "\t")):
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip().strip("\"'")
            body = "\n".join(lines[end + 1:]).lstrip("\n")

    return Skill(
        name=metadata.get("name", skill_file.parent.name),
        description=metadata.get("description", ""),
        body=body,
        path=skill_file.parent.resolve(),
        metadata=metadata,
    )
