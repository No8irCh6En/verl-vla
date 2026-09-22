import hydra


@hydra.main(config_path="../../workflows/config", config_name="train/flow_grpo", version_base=None)
def main(config):
    from verl_vla.workflows.train.flow_grpo import run_flow_grpo

    run_flow_grpo(config)


if __name__ == "__main__":
    main()
