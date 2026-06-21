from typer.testing import CliRunner

from argentine_deputies_discursive_distance import __version__
from argentine_deputies_discursive_distance.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout