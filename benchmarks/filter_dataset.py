"""Small human-labeled workloads for conservative filter evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledCandidate:
    id: str
    text: str
    relevant: bool


@dataclass(frozen=True, slots=True)
class LabeledTask:
    name: str
    criterion: str
    candidates: tuple[LabeledCandidate, ...]


DATASET: tuple[LabeledTask, ...] = (
    LabeledTask(
        name="file-relevance",
        criterion="Fix player acceleration behavior",
        candidates=(
            LabeledCandidate("movement", "player_controller.gd handles acceleration, input, and jumping.", True),
            LabeledCandidate("animation", "player_animation.gd maps locomotion state to animations.", True),
            LabeledCandidate("movement-tests", "test_player_acceleration checks velocity after input.", True),
            LabeledCandidate("inventory", "inventory.gd manages item slots and equipment.", False),
            LabeledCandidate("save", "save_system.gd serializes player progress.", False),
            LabeledCandidate("audio", "audio_manager.gd mixes music and sound effects.", False),
            LabeledCandidate("network", "multiplayer_sync.gd replicates remote player state.", False),
        ),
    ),
    LabeledTask(
        name="search-result-relevance",
        criterion="Fix a null reference when the player jumps after landing on a slope",
        candidates=(
            LabeledCandidate("floor-snap", "CharacterBody2D movement and floor snapping after landing.", True),
            LabeledCandidate("jump-buffer", "Jump input buffering and coyote time in PlayerController.", True),
            LabeledCandidate("slope-shader", "Terrain slope shader normal calculation.", False),
            LabeledCandidate("serialization", "Save-game serialization of player inventory.", False),
            LabeledCandidate("editor-import", "Editor import settings for environment textures.", False),
            LabeledCandidate("collision", "Collision normal handling in the player movement loop.", True),
        ),
    ),
    LabeledTask(
        name="test-relevance",
        criterion="Review a change that fixes player acceleration and jump state transitions",
        candidates=(
            LabeledCandidate("acceleration-test", "test_player_acceleration_changes_velocity.", True),
            LabeledCandidate("jump-test", "test_jump_resets_after_landing.", True),
            LabeledCandidate("movement-regression", "test_player_state_transitions_after_jump.", True),
            LabeledCandidate("enemy-test", "test_enemy_pathfinding_avoids_walls.", False),
            LabeledCandidate("save-test", "test_save_slot_round_trip.", False),
            LabeledCandidate("menu-test", "test_main_menu_opens_settings.", False),
        ),
    ),
    LabeledTask(
        name="error-relevance",
        criterion="Diagnose player acceleration becoming zero after landing",
        candidates=(
            LabeledCandidate("null-player", "NullReference: PlayerController.velocity in _physics_process.", True),
            LabeledCandidate("floor-normal", "Warning: floor normal changed during player landing.", True),
            LabeledCandidate("audio-error", "AudioServer failed to load ambient_loop.ogg.", False),
            LabeledCandidate("network-timeout", "Network peer handshake timed out for lobby host.", False),
            LabeledCandidate("shader-warning", "Shader compilation warning in terrain_material.", False),
            LabeledCandidate("velocity-log", "Debug: player velocity reset to (0, 0) after move_and_slide.", True),
        ),
    ),
)
