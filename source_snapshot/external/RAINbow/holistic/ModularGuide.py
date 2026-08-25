from interface.Navigation import Navigation
from interface.WTA import WTA
from interface.AnswerGeneration  import AnswerGeneration
from interface.Localization import Localization
from interface.GuideAgent import GuideAgent
from holistic_utils.temporal_localization import TemporalLocalizationReranker
import numpy as np

class ModularGuide(GuideAgent):
    def __init__(self, args, answer_model: AnswerGeneration, localization_model: Localization, env_infos: dict):
        super().__init__(args, answer_model, localization_model)
        self.args = args
        self.answer_model = answer_model
        self.localization_model = localization_model
        self.localization_metadata = []
        self.temporal_localization = None

        self.shortest_distances = env_infos['shortest_distances']
        self.shortest_paths = env_infos['shortest_paths']
        self.goals_by_instr_id = env_infos['goals_by_instr_id']

    def get_goals(self, instruction_ids):
        return [
            self.goals_by_instr_id[str(instruction_id)]
            for instruction_id in instruction_ids
        ]

    def initialize_temporal_state(self, batch_size):
        self.temporal_localization = TemporalLocalizationReranker(
            batch_size,
            self.shortest_distances,
            weight=self.args.temporal_weight,
            top_k=self.args.temporal_top_k,
        )

    def _choose_path(self, scanId, viewpoint, goal_list):
        distances_all = []
        paths_all = []
        for g in goal_list:
            distances_all.append(self.shortest_distances[scanId][viewpoint][g])
            paths_all.append(self.shortest_paths[scanId][viewpoint][g])
        sorted_indices = np.argsort(distances_all)
        sorted_paths = [paths_all[i] for i in sorted_indices]
        sorted_distances = [distances_all[i] for i in sorted_indices]
        path = sorted_paths[0]
        return path

    ##### Answer Functions #####
    def answer(self, *args, **kwargs):
        if self.answer_model is None:
            raise ValueError("answer_model is not set")
        return self.answer_model.answer(*args, **kwargs)

    def confirm_goals(
        self,
        dialog_indices,
        localized_viewpoints,
        goals,
        answers,
        confirmation_text,
    ):
        """Encode a Guide-private target match as natural-language text."""
        for index in dialog_indices:
            if localized_viewpoints[index] in goals[index]:
                answers[index] = confirmation_text
        return answers

    def remember_answer_routes(self, step, dialog_indices, answer_seen_paths):
        if self.temporal_localization is None:
            raise ValueError("temporal localization state is not initialized")
        self.temporal_localization.update(
            step, dialog_indices, answer_seen_paths
        )
    
    ##### Localization Functions #####
    def localize(self, scanIds, questions):
        if self.localization_model is None:
            raise ValueError("localization_model is not set")
        predictions = self.localization_model.localize(scanIds, questions)
        self.localization_metadata = getattr(
            self.localization_model, 'last_metadata', []
        )
        return predictions

    def localize_temporally(self, scan_ids, questions, step, dialog_indices):
        if self.temporal_localization is None:
            raise ValueError("temporal localization state is not initialized")
        predictions = self.localize(scan_ids, questions)
        raw_predictions = list(predictions)
        if len(self.localization_metadata) != len(scan_ids):
            raise ValueError("GTL did not return per-item top-k metadata")

        expected_viewpoints = [None] * len(scan_ids)
        selected_probabilities = [None] * len(scan_ids)
        for index in dialog_indices:
            (
                predictions[index],
                expected_viewpoints[index],
                selected_probabilities[index],
            ) = self.temporal_localization.rerank(
                index,
                scan_ids[index],
                raw_predictions[index],
                self.localization_metadata[index],
                step,
            )
        return (
            predictions,
            raw_predictions,
            expected_viewpoints,
            selected_probabilities,
        )
