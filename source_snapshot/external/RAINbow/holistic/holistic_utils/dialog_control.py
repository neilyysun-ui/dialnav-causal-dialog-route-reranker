def select_dialog_indices(ask, ended_before_step, ended_this_step, proposed_stop):
    """Select active episodes, including a newly proposed STOP for verification."""
    return [
        index
        for index, should_ask in enumerate(ask)
        if should_ask
        and not ended_before_step[index]
        and (not ended_this_step[index] or proposed_stop[index])
    ]


def select_goal_confirmation_indices(
    dialog_indices, localized_viewpoints, goals
):
    """Let the Guide confirm when its own location estimate is a target node."""
    return [
        index
        for index in dialog_indices
        if localized_viewpoints[index] in goals[index]
    ]
