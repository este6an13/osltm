# Online Statistical Learning & Stochastic Modelling for TransMilenio

- [Click here to see TODO list](docs/todo/todo.md)
- [Click here to go to notebooks folder](src/scripts/notebooks/)

## Workflow Execution

```sh
# Run all steps
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps all

# Run only step 1
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1

# Run steps 1-2
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1-2
```

### Data Loader

```py
from src.workflow.data_loader import load_data

# Load all data from persistence (default behavior)
data = load_data()
checkins_df = data["checkins"]
checkouts_df = data["checkouts"]

# Load only check-ins
data = load_data(include_checkouts=False)
checkins_df = data["checkins"]

# Load only check-outs
data = load_data(include_checkins=False)
checkouts_df = data["checkouts"]

# Load specific dates and stations
data = load_data(
    dates=["20240625", "20240628"],
    station_codes=["03000", "05105"],
    include_checkins=True,
    include_checkouts=True
)
```

### Exploratory Analysis Scripts

```sh
uv run python -m src.workflow.fpca_per_station --stations 03000

uv run python -m src.workflow.within_between_distances --stations 03000

uv run python -m src.workflow.mean_envelope_plots --stations 03000
```