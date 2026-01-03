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