import platformdirs as pd

from utils.constants import APP_NAME

config_dir = pd.user_config_path(
    APP_NAME,
    roaming=True,
    appauthor=False,
    ensure_exists=True,
)

calibration_file = config_dir / "calibration.ini"
