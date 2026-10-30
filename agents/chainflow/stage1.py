"""Device-only invocation adapter for the unchanged official stage1 model.

Feature builders, selected trajectory, loss and optimizer remain native.
"""


def trajectory_from_input(agent, agent_input):
    import torch
    from navsim.common.dataclasses import Trajectory

    agent.eval()
    device = next(agent.parameters()).device
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {key: value.unsqueeze(0).to(device) for key, value in features.items()}
    with torch.inference_mode():
        prediction = agent.forward(features)
    return Trajectory(prediction["trajectory"].squeeze(0).detach().cpu().numpy())
