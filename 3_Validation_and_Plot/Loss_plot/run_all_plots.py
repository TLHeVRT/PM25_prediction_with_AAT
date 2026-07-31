from pathlib import Path
import subprocess
import sys


base_dir = Path(__file__).resolve().parent
loss_file = base_dir / "station_test_losses.csv"
plot_scripts = (
    "plot_loss_on_pm25_stats.py",
    "plot_loss_normalized_on_pm25_stats.py",
    "plot_normloss_vs_pm25_mean.py",
    "plot_loss_on_map.py",
)

for script in plot_scripts:
    subprocess.run(
        [sys.executable, str(base_dir / script), str(loss_file)],
        cwd=base_dir,
    )
