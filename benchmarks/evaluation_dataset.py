"""Inspectable, human-labeled Milestone 5 evaluation data.

The development split is available for prompt/instruction experiments.  The
evaluation split is held out from those choices and is the primary basis for the
reported numbers.  ``possibly_relevant`` is treated as should-retain for the
asymmetric filtering analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RelevanceLabel = Literal["definitely_relevant", "possibly_relevant", "irrelevant"]
Split = Literal["dev", "eval"]


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    id: str
    text: str
    label: RelevanceLabel

    @property
    def should_retain(self) -> bool:
        return self.label != "irrelevant"


@dataclass(frozen=True, slots=True)
class RelevanceTask:
    id: str
    category: str
    criterion: str
    candidates: tuple[EvaluationCandidate, ...]
    split: Split


@dataclass(frozen=True, slots=True)
class ActionCase:
    id: str
    state: str
    question: str
    actions: tuple[str, ...]
    expected_action: str
    split: Split


@dataclass(frozen=True, slots=True)
class BoundedCase:
    id: str
    category: str
    context: str
    question: str
    options: tuple[str, ...] | None
    expected: bool | str
    split: Split


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    id: str
    label: RelevanceLabel
    variants: dict[str, str]

    @property
    def should_retain(self) -> bool:
        return self.label != "irrelevant"


@dataclass(frozen=True, slots=True)
class RepresentationCase:
    id: str
    criterion: str
    candidates: tuple[RepresentationCandidate, ...]
    split: Split


@dataclass(frozen=True, slots=True)
class ExtendedContextTask:
    id: str
    category: str
    criterion: str
    candidates: tuple[EvaluationCandidate, ...]


def _candidate_items(
    values: tuple[tuple[str, str, RelevanceLabel], ...],
) -> tuple[EvaluationCandidate, ...]:
    return tuple(EvaluationCandidate(*value) for value in values)


RELEVANCE_TASKS: tuple[RelevanceTask, ...] = (
    RelevanceTask(
        "file-acceleration-dev", "file_relevance", "Fix player acceleration after landing", _candidate_items((
            ("controller", "player_controller.gd applies input acceleration and move_and_slide.", "definitely_relevant"),
            ("state", "player_state.gd switches grounded and airborne movement modes.", "possibly_relevant"),
            ("camera", "camera_follow.gd smooths the view behind the player.", "irrelevant"),
            ("jump-test", "test_player_jump.gd checks jump input and landing reset.", "possibly_relevant"),
            ("save", "save_game.gd serializes checkpoints and settings.", "irrelevant"),
            ("input-map", "project.godot defines the move_left and move_right actions.", "possibly_relevant"),
        )), "dev",
    ),
    RelevanceTask(
        "file-auth-dev", "file_relevance", "Repair an authentication failure after token refresh", _candidate_items((
            ("session", "session_manager.py refreshes access tokens and updates cookies.", "definitely_relevant"),
            ("middleware", "auth_middleware.py attaches the current principal to requests.", "definitely_relevant"),
            ("login-test", "test_login_flow.py covers expired credentials and retry behavior.", "possibly_relevant"),
            ("profile", "profile_view.ts renders the signed-in user's avatar.", "irrelevant"),
            ("metrics", "metrics_exporter.py sends request counters to telemetry.", "irrelevant"),
            ("clock", "clock.py normalizes timestamps from external services.", "possibly_relevant"),
        )), "dev",
    ),
    RelevanceTask(
        "file-migration-dev", "file_relevance", "Change the database migration that adds an invoice status", _candidate_items((
            ("migration", "migrations/042_invoice_status.sql adds a nullable status column.", "definitely_relevant"),
            ("model", "invoice.py maps database rows to the Invoice domain object.", "possibly_relevant"),
            ("migration-test", "test_migrations.py applies migrations to a temporary database.", "possibly_relevant"),
            ("checkout", "checkout_controller.py validates card details before payment.", "irrelevant"),
            ("docs", "docs/billing.md explains how customers download receipts.", "possibly_relevant"),
            ("mailer", "mailer.py sends invoice emails after an order is paid.", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "file-payment-eval", "file_relevance", "Correct the payment total when a discount is applied", _candidate_items((
            ("totals", "payment_totals.ts combines line items, tax, discount, and currency rounding.", "definitely_relevant"),
            ("coupon", "coupon_service.ts validates promotion codes and expiration dates.", "possibly_relevant"),
            ("payment-test", "payment_totals.test.ts checks a discounted cart total.", "definitely_relevant"),
            ("receipt", "receipt_template.html formats the final amount for email.", "possibly_relevant"),
            ("shipping", "shipping_rates.ts selects a carrier based on destination.", "irrelevant"),
            ("login", "login_form.ts handles password validation and redirects.", "irrelevant"),
        )), "eval",
    ),
    RelevanceTask(
        "file-animation-eval", "file_relevance", "Fix an animation transition that sticks after a dash", _candidate_items((
            ("anim-tree", "character_animation.tres blends locomotion and dash states.", "definitely_relevant"),
            ("dash", "dash_ability.gd starts a dash and emits a movement state signal.", "definitely_relevant"),
            ("sprite", "sprite_sheet_import.gd configures frame regions for characters.", "possibly_relevant"),
            ("combat-test", "test_combat_damage.gd checks hit points after an attack.", "irrelevant"),
            ("camera-shake", "camera_shake.gd responds to impacts with a screen offset.", "possibly_relevant"),
            ("audio", "footstep_audio.gd selects a sound for each surface material.", "possibly_relevant"),
        )), "eval",
    ),
    RelevanceTask(
        "file-timeout-eval", "file_relevance", "Investigate an API timeout while loading account settings", _candidate_items((
            ("client", "settings_client.go calls the account preferences endpoint with a context deadline.", "definitely_relevant"),
            ("retry", "retry_policy.go applies exponential backoff to idempotent HTTP requests.", "possibly_relevant"),
            ("handler", "settings_handler.go returns preferences to the web application.", "possibly_relevant"),
            ("db", "postgres_pool.go creates the database connection pool for billing.", "possibly_relevant"),
            ("avatar", "avatar_upload.go streams image bytes to object storage.", "irrelevant"),
            ("timeout-test", "settings_client_test.go simulates a slow upstream response.", "definitely_relevant"),
        )), "eval",
    ),
    RelevanceTask(
        "search-slope-dev", "search_relevance", "Fix a null reference when the player jumps after landing on a slope", _candidate_items((
            ("hit-1", "player.gd: velocity = move_and_slide(velocity, floor_normal)", "definitely_relevant"),
            ("hit-2", "terrain_shader.gd: normal_map changes lighting on steep slopes", "irrelevant"),
            ("hit-3", "movement_test.gd: jump is buffered while floor_snap_length is nonzero", "definitely_relevant"),
            ("hit-4", "collision.gd: stores the last contact normal for moving bodies", "possibly_relevant"),
            ("hit-5", "save.gd: writes the current scene name before quitting", "irrelevant"),
            ("hit-6", "player_camera.gd: follows the target with a vertical dead zone", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "search-token-dev", "search_relevance", "Repair invalid authentication after a refresh token rotation", _candidate_items((
            ("hit-1", "oauth.py: rotate_refresh_token(old_token) persists the replacement", "definitely_relevant"),
            ("hit-2", "template.html: displays the word authentication in a help paragraph", "irrelevant"),
            ("hit-3", "request_context.py: reads the bearer token before dispatch", "possibly_relevant"),
            ("hit-4", "test_oauth.py: rejects reuse of a rotated refresh token", "definitely_relevant"),
            ("hit-5", "jwt_tools.py: decodes claims without validating the signature", "possibly_relevant"),
            ("hit-6", "billing.py: retries a failed charge against the payment gateway", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "search-index-eval", "search_relevance", "Fix a slow query after adding a customer status index", _candidate_items((
            ("hit-1", "customer_repo.rb: WHERE status = ? scans the customers relation", "definitely_relevant"),
            ("hit-2", "migration.sql: CREATE INDEX customers_status_idx ON customers(status)", "definitely_relevant"),
            ("hit-3", "query_log.txt: SELECT status FROM orders took 4ms", "possibly_relevant"),
            ("hit-4", "search_controller.rb: tokenizes free-text product queries", "irrelevant"),
            ("hit-5", "db_metrics.rb: records query duration and row counts", "possibly_relevant"),
            ("hit-6", "customer_view.erb: renders the status badge", "irrelevant"),
        )), "eval",
    ),
    RelevanceTask(
        "search-cache-eval", "search_relevance", "Find why cached API responses contain stale permissions", _candidate_items((
            ("hit-1", "cache_key.go: key includes user ID but omits permission version", "definitely_relevant"),
            ("hit-2", "permissions.go: computes roles from the current organization", "definitely_relevant"),
            ("hit-3", "redis_client.go: sets a 60-second expiration on response keys", "possibly_relevant"),
            ("hit-4", "docs/cache.md: explains cache invalidation terminology", "possibly_relevant"),
            ("hit-5", "image_cache.go: caches resized profile pictures", "irrelevant"),
            ("hit-6", "audit_log.go: records role changes for compliance", "possibly_relevant"),
        )), "eval",
    ),
    RelevanceTask(
        "search-payment-dev", "search_relevance", "Correct tax rounding in the payment calculation", _candidate_items((
            ("hit-1", "money.ts: roundTax(amount, currency) applies bankers rounding", "definitely_relevant"),
            ("hit-2", "cart.ts: subtotal() sums item prices before discounts", "possibly_relevant"),
            ("hit-3", "tax_test.ts: expects a one-cent difference for three items", "definitely_relevant"),
            ("hit-4", "payment_gateway.ts: sends the final amount to the processor", "possibly_relevant"),
            ("hit-5", "analytics.ts: reports conversion rate by campaign", "irrelevant"),
            ("hit-6", "invoice.css: aligns currency symbols in the PDF", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "test-movement-dev", "test_relevance", "Fix player acceleration and jump state transitions", _candidate_items((
            ("t1", "test_player_acceleration_changes_velocity", "definitely_relevant"),
            ("t2", "test_jump_resets_after_landing", "definitely_relevant"),
            ("t3", "test_player_scene_loads_with_default_input_map", "possibly_relevant"),
            ("t4", "test_enemy_pathfinding_avoids_walls", "irrelevant"),
            ("t5", "test_save_slot_round_trip", "irrelevant"),
            ("t6", "test_camera_smoothing_after_respawn", "possibly_relevant"),
        )), "dev",
    ),
    RelevanceTask(
        "test-auth-dev", "test_relevance", "Repair authentication after refresh token rotation", _candidate_items((
            ("t1", "test_refresh_token_rotation_rejects_old_token", "definitely_relevant"),
            ("t2", "test_login_redirects_after_valid_credentials", "possibly_relevant"),
            ("t3", "test_request_middleware_attaches_user", "definitely_relevant"),
            ("t4", "test_invoice_email_is_sent_after_payment", "irrelevant"),
            ("t5", "test_password_reset_token_expires", "possibly_relevant"),
            ("t6", "test_avatar_upload_resizes_images", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "test-migration-eval", "test_relevance", "Change the migration that adds invoice status", _candidate_items((
            ("t1", "test_migration_042_adds_invoice_status", "definitely_relevant"),
            ("t2", "test_invoice_model_reads_status", "definitely_relevant"),
            ("t3", "test_migration_is_idempotent", "possibly_relevant"),
            ("t4", "test_checkout_declines_expired_card", "irrelevant"),
            ("t5", "test_invoice_pdf_contains_total", "possibly_relevant"),
            ("t6", "test_database_pool_reconnects", "possibly_relevant"),
        )), "eval",
    ),
    RelevanceTask(
        "test-timeout-eval", "test_relevance", "Investigate an API timeout while loading settings", _candidate_items((
            ("t1", "test_settings_client_times_out", "definitely_relevant"),
            ("t2", "test_settings_request_retries_once", "definitely_relevant"),
            ("t3", "test_preferences_handler_returns_200", "possibly_relevant"),
            ("t4", "test_websocket_reconnects_after_disconnect", "possibly_relevant"),
            ("t5", "test_avatar_upload_timeout", "irrelevant"),
            ("t6", "test_metrics_exporter_flushes_counters", "irrelevant"),
        )), "eval",
    ),
    RelevanceTask(
        "test-animation-dev", "test_relevance", "Fix an animation transition that sticks after a dash", _candidate_items((
            ("t1", "test_dash_returns_to_run_animation", "definitely_relevant"),
            ("t2", "test_animation_tree_blends_locomotion", "definitely_relevant"),
            ("t3", "test_footstep_sound_changes_on_surface", "possibly_relevant"),
            ("t4", "test_damage_flash_ends_after_timer", "irrelevant"),
            ("t5", "test_camera_shake_on_dash", "possibly_relevant"),
            ("t6", "test_inventory_equipment_persists", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "error-movement-dev", "error_relevance", "Diagnose player acceleration becoming zero after landing", _candidate_items((
            ("e1", "NullReference: PlayerController.velocity in _physics_process", "definitely_relevant"),
            ("e2", "Debug: player velocity reset to (0, 0) after move_and_slide", "definitely_relevant"),
            ("e3", "Warning: floor normal changed during player landing", "possibly_relevant"),
            ("e4", "AudioServer failed to load ambient_loop.ogg", "irrelevant"),
            ("e5", "Network peer handshake timed out for lobby host", "irrelevant"),
            ("e6", "Shader compilation warning in terrain_material", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "error-auth-dev", "error_relevance", "Diagnose invalid authentication after token refresh", _candidate_items((
            ("e1", "401: refresh token has already been rotated", "definitely_relevant"),
            ("e2", "Warning: bearer token missing from request context", "definitely_relevant"),
            ("e3", "INFO: login page rendered in 42ms", "possibly_relevant"),
            ("e4", "Payment gateway returned card_declined", "irrelevant"),
            ("e5", "Redis connection pool reached its limit", "possibly_relevant"),
            ("e6", "Image decoder rejected an unsupported PNG profile", "irrelevant"),
        )), "dev",
    ),
    RelevanceTask(
        "error-payment-eval", "error_relevance", "Diagnose a payment total that differs by one cent", _candidate_items((
            ("e1", "AssertionError: expected 10.01, received 10.00 after tax rounding", "definitely_relevant"),
            ("e2", "Payment processor rejected amount format with currency EUR", "possibly_relevant"),
            ("e3", "Trace: discount applied before tax calculation", "definitely_relevant"),
            ("e4", "Timeout while fetching shipping rates", "possibly_relevant"),
            ("e5", "Login attempt failed for unknown user", "irrelevant"),
            ("e6", "PDF renderer could not load the invoice font", "irrelevant"),
        )), "eval",
    ),
    RelevanceTask(
        "error-cache-eval", "error_relevance", "Diagnose stale permissions in cached API responses", _candidate_items((
            ("e1", "Audit: role changed, but cached permission version stayed at 12", "definitely_relevant"),
            ("e2", "Cache miss for user 42 after organization switch", "possibly_relevant"),
            ("e3", "403: permission denied on a newly granted endpoint", "definitely_relevant"),
            ("e4", "Avatar thumbnail cache returned a corrupt image", "irrelevant"),
            ("e5", "Redis eviction count increased during deploy", "possibly_relevant"),
            ("e6", "Database migration checksum mismatch", "irrelevant"),
        )), "eval",
    ),
    RelevanceTask(
        "error-timeout-dev", "error_relevance", "Diagnose an API timeout while loading account settings", _candidate_items((
            ("e1", "context deadline exceeded calling /account/settings", "definitely_relevant"),
            ("e2", "Retry budget exhausted after three upstream timeouts", "definitely_relevant"),
            ("e3", "Settings response decoded with an unknown field", "possibly_relevant"),
            ("e4", "Web client failed to load the dashboard CSS", "irrelevant"),
            ("e5", "Database pool wait exceeded 200ms", "possibly_relevant"),
            ("e6", "Email delivery provider returned a 5xx", "irrelevant"),
        )), "dev",
    ),
)


ACTION_CASES: tuple[ActionCase, ...] = (
    ActionCase("route-1", "A reported bug names a likely module but no code has been inspected yet.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "inspect_file", "dev"),
    ActionCase("route-2", "The symbol name is unknown and the repository may contain several implementations.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "search_code", "dev"),
    ActionCase("route-3", "A small patch is complete and a focused test command is available.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "run_tests", "eval"),
    ActionCase("route-4", "The requested change is implemented and its checks are green.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "continue_implementation", "eval"),
    ActionCase("route-5", "The requirement is ambiguous and two materially different behaviors are plausible.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "ask_user", "eval"),
    ActionCase("route-6", "The decision affects credentials and the available evidence is contradictory.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "escalate_reasoning"), "escalate_reasoning", "eval"),
    ActionCase("route-7", "A stack trace identifies the exact failing function and the change request is precise.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "inspect_file", "dev"),
    ActionCase("route-8", "A regression test fails after a change, but the failure output is complete.", "What should happen next?", ("inspect_file", "search_code", "run_tests", "continue_implementation", "ask_user"), "inspect_file", "eval"),
)


BOUNDED_CASES: tuple[BoundedCase, ...] = (
    BoundedCase("bool-1", "bounded_binary", "The build has 12 passing tests and 0 failures.", "Are all tests passing?", None, True, "dev"),
    BoundedCase("bool-2", "bounded_binary", "The request returned HTTP status 404.", "Did the request succeed?", None, False, "dev"),
    BoundedCase("bool-3", "bounded_binary", "The retry limit is configured as 3 attempts.", "Is the retry limit greater than 2?", None, True, "dev"),
    BoundedCase("bool-4", "bounded_binary", "The feature flag named new_checkout is disabled.", "Is new_checkout enabled?", None, False, "eval"),
    BoundedCase("bool-5", "bounded_binary", "The cache entry expires after 60 seconds.", "Does the cache expire within one minute?", None, True, "eval"),
    BoundedCase("bool-6", "bounded_binary", "The migration is marked reversible in the manifest.", "Can this migration be rolled back?", None, True, "eval"),
    BoundedCase("choice-1", "bounded_choice", "The request is read-only and the cache is cold.", "Which operation is appropriate?", ("read_cache", "write_cache", "delete_cache"), "read_cache", "dev"),
    BoundedCase("choice-2", "bounded_choice", "The user entered an invalid one-time code.", "Which outcome should be shown?", ("accept", "retry", "lock_account"), "retry", "dev"),
    BoundedCase("choice-3", "bounded_choice", "The database schema is one version behind and the migration is available.", "What should happen?", ("apply_migration", "drop_database", "ignore"), "apply_migration", "eval"),
    BoundedCase("choice-4", "bounded_choice", "The API returned a temporary 503 and the request is idempotent.", "Which policy applies?", ("retry", "surface_error", "duplicate_request"), "retry", "eval"),
    BoundedCase("choice-5", "bounded_choice", "The supplied token is expired but the refresh token is valid.", "Which action is correct?", ("refresh", "logout", "grant_access"), "refresh", "eval"),
    BoundedCase("choice-6", "bounded_choice", "The input contains a malformed email address.", "Which validation result applies?", ("valid", "invalid", "unknown"), "invalid", "dev"),
)


REPRESENTATION_CASES: tuple[RepresentationCase, ...] = (
    RepresentationCase(
        "repr-acceleration",
        "Fix player acceleration after landing",
        (
            RepresentationCandidate("controller", "definitely_relevant", {
                "path_only": "player_controller.gd",
                "path_short": "player_controller.gd — movement controller",
                "path_rich": "player_controller.gd — applies input acceleration and move_and_slide after landing",
                "short_snippet": "player_controller.gd: velocity is updated from input and floor state",
                "long_snippet": "player_controller.gd: after a landing, the movement loop reads floor state, applies horizontal input acceleration, and calls move_and_slide before storing the new velocity",
            }),
            RepresentationCandidate("animation", "possibly_relevant", {
                "path_only": "player_animation.gd",
                "path_short": "player_animation.gd — locomotion animation",
                "path_rich": "player_animation.gd — transitions between grounded, falling, and dash animations",
                "short_snippet": "player_animation.gd: animation state follows grounded and airborne movement",
                "long_snippet": "player_animation.gd: animation transitions consume grounded, falling, and dash signals emitted by the player movement state machine",
            }),
            RepresentationCandidate("inventory", "irrelevant", {
                "path_only": "inventory.gd",
                "path_short": "inventory.gd — item slots",
                "path_rich": "inventory.gd — adds, removes, and persists item stacks",
                "short_snippet": "inventory.gd: item slots and equipment persistence",
                "long_snippet": "inventory.gd: manages item stacks, equipment slots, and serialization of the player's collected resources",
            }),
            RepresentationCandidate("input-map", "possibly_relevant", {
                "path_only": "project.godot",
                "path_short": "project.godot — input actions",
                "path_rich": "project.godot — defines move_left and move_right actions",
                "short_snippet": "project.godot: movement actions are mapped to keyboard and controller input",
                "long_snippet": "project.godot: input actions for move_left and move_right are defined here and consumed by the player controller",
            }),
        ),
        "eval",
    ),
    RepresentationCase(
        "repr-auth",
        "Repair authentication after refresh token rotation",
        (
            RepresentationCandidate("oauth", "definitely_relevant", {
                "path_only": "oauth.py",
                "path_short": "oauth.py — refresh token handling",
                "path_rich": "oauth.py — rotates refresh tokens and persists replacements",
                "short_snippet": "oauth.py: old refresh tokens are rejected after rotation",
                "long_snippet": "oauth.py: refresh_token rotates the credential, persists its replacement, and invalidates the previous token before returning",
            }),
            RepresentationCandidate("middleware", "possibly_relevant", {
                "path_only": "request_context.py",
                "path_short": "request_context.py — request authentication",
                "path_rich": "request_context.py — reads bearer credentials before dispatch",
                "short_snippet": "request_context.py: bearer token is read before the request handler runs",
                "long_snippet": "request_context.py: request middleware extracts the bearer credential, resolves the principal, and attaches authentication state to the context",
            }),
            RepresentationCandidate("profile", "irrelevant", {
                "path_only": "profile_view.ts",
                "path_short": "profile_view.ts — avatar rendering",
                "path_rich": "profile_view.ts — renders the signed-in user's avatar and display name",
                "short_snippet": "profile_view.ts: renders account avatar and display name",
                "long_snippet": "profile_view.ts: presents the signed-in user's profile details and avatar after authentication has already succeeded",
            }),
            RepresentationCandidate("clock", "possibly_relevant", {
                "path_only": "clock.py",
                "path_short": "clock.py — timestamp normalization",
                "path_rich": "clock.py — normalizes timestamps from external services",
                "short_snippet": "clock.py: normalizes token expiry timestamps from external services",
                "long_snippet": "clock.py: converts external timestamp formats before expiry checks compare the current time with token claims",
            }),
        ),
        "eval",
    ),
    RepresentationCase(
        "repr-index",
        "Fix a slow query after adding a customer status index",
        (
            RepresentationCandidate("repo", "definitely_relevant", {
                "path_only": "customer_repo.rb",
                "path_short": "customer_repo.rb — customer queries",
                "path_rich": "customer_repo.rb — queries customers by status",
                "short_snippet": "customer_repo.rb: WHERE status = ? scans the customers relation",
                "long_snippet": "customer_repo.rb: the list_by_status query filters the customers table by status and currently scans the relation instead of using the new index",
            }),
            RepresentationCandidate("migration", "definitely_relevant", {
                "path_only": "migration.sql",
                "path_short": "migration.sql — customer status index",
                "path_rich": "migration.sql — creates customers_status_idx on status",
                "short_snippet": "migration.sql: CREATE INDEX customers_status_idx ON customers(status)",
                "long_snippet": "migration.sql: migration 18 creates customers_status_idx on the status column used by the repository's slow list query",
            }),
            RepresentationCandidate("metrics", "possibly_relevant", {
                "path_only": "db_metrics.rb",
                "path_short": "db_metrics.rb — query timings",
                "path_rich": "db_metrics.rb — records query duration and row counts",
                "short_snippet": "db_metrics.rb: records duration and rows for slow database queries",
                "long_snippet": "db_metrics.rb: query instrumentation records duration, rows scanned, and the normalized SQL label used in performance investigations",
            }),
            RepresentationCandidate("view", "irrelevant", {
                "path_only": "customer_view.erb",
                "path_short": "customer_view.erb — status badge",
                "path_rich": "customer_view.erb — renders a customer status badge",
                "short_snippet": "customer_view.erb: renders status text in the customer page",
                "long_snippet": "customer_view.erb: presents the status value returned by the controller and adds CSS classes for the visual badge",
            }),
        ),
        "eval",
    ),
    RepresentationCase(
        "repr-timeout",
        "Investigate an API timeout while loading account settings",
        (
            RepresentationCandidate("client", "definitely_relevant", {
                "path_only": "settings_client.go",
                "path_short": "settings_client.go — account settings request",
                "path_rich": "settings_client.go — calls account preferences with a context deadline",
                "short_snippet": "settings_client.go: GET /account/settings uses a context deadline",
                "long_snippet": "settings_client.go: the account preferences request sets a context deadline, waits for the upstream response, and decodes settings for the caller",
            }),
            RepresentationCandidate("retry", "possibly_relevant", {
                "path_only": "retry_policy.go",
                "path_short": "retry_policy.go — HTTP retry policy",
                "path_rich": "retry_policy.go — backs off on idempotent upstream timeouts",
                "short_snippet": "retry_policy.go: exponential backoff applies after idempotent HTTP timeouts",
                "long_snippet": "retry_policy.go: idempotent requests retry with exponential backoff after timeout and temporary upstream failures, subject to a retry budget",
            }),
            RepresentationCandidate("pool", "possibly_relevant", {
                "path_only": "postgres_pool.go",
                "path_short": "postgres_pool.go — database connection pool",
                "path_rich": "postgres_pool.go — creates the account database connection pool",
                "short_snippet": "postgres_pool.go: pool wait time can delay account settings queries",
                "long_snippet": "postgres_pool.go: the database pool controls connection acquisition and can add latency before the account settings handler executes its query",
            }),
            RepresentationCandidate("avatar", "irrelevant", {
                "path_only": "avatar_upload.go",
                "path_short": "avatar_upload.go — image uploads",
                "path_rich": "avatar_upload.go — streams profile images to object storage",
                "short_snippet": "avatar_upload.go: streams image bytes to object storage",
                "long_snippet": "avatar_upload.go: uploads profile image bytes to object storage and reports image processing errors independently of account settings",
            }),
        ),
        "eval",
    ),
)


EXTENDED_CONTEXT_TASKS: tuple[ExtendedContextTask, ...] = (
    ExtendedContextTask(
        "extended-file-payment",
        "file_relevance",
        "Correct the payment total when a discount is applied",
        _candidate_items((
            (
                "totals",
                "payment_totals.ts is the module that combines line items, tax, discounts, currency conversion, and rounding. It receives the cart subtotal, applies a promotion adjustment, calculates tax according to the destination, and formats the amount sent to the payment gateway. The reported bug is a one-cent discrepancy after a discount, so this module's arithmetic and ordering are directly involved.",
                "definitely_relevant",
            ),
            (
                "receipt",
                "receipt_template.html renders the final amount in an email. It selects a currency symbol, formats decimal places, inserts the merchant address, and lays out the receipt using HTML and CSS. It does not calculate the amount or communicate with the payment processor; it only displays values already supplied by the server.",
                "possibly_relevant",
            ),
            (
                "shipping",
                "shipping_rates.ts chooses a carrier and delivery estimate from a destination, parcel dimensions, and service level. It can change the shipping line item before checkout, but it does not apply discounts, calculate tax, or round the payment total. Its tests focus on postal zones and unavailable services.",
                "irrelevant",
            ),
        )),
    ),
    ExtendedContextTask(
        "extended-search-cache",
        "search_relevance",
        "Find why cached API responses contain stale permissions",
        _candidate_items((
            (
                "cache-key",
                "cache_key.go builds the key used for authorization-sensitive API responses. The current key contains the user identifier and endpoint but not the organization permission version. When a role changes, the old response remains addressable under the same key until its time-to-live expires, which can expose stale permissions even though the permission resolver returns the new role.",
                "definitely_relevant",
            ),
            (
                "audit",
                "audit_log.go appends an entry whenever an organization role changes. It records actor, target, previous role, new role, and timestamp for compliance review. It does not read or write response-cache keys, although its events could be used as a source for invalidation if the system connected them.",
                "possibly_relevant",
            ),
            (
                "images",
                "image_cache.go stores resized profile pictures under keys derived from image hashes and requested dimensions. It has no user permission data, organization roles, API response bodies, or authorization checks. Evicting these thumbnails would not change the stale permissions returned by an API endpoint.",
                "irrelevant",
            ),
        )),
    ),
    ExtendedContextTask(
        "extended-test-timeout",
        "test_relevance",
        "Investigate an API timeout while loading account settings",
        _candidate_items((
            (
                "client-timeout",
                "settings_client_test.go starts a local upstream server that intentionally delays the account preferences response beyond the configured context deadline. It asserts that the client returns a timeout, records the upstream URL and elapsed time, and does not retry a request that has already exceeded its caller-provided deadline. This directly exercises the reported failure.",
                "definitely_relevant",
            ),
            (
                "handler-success",
                "settings_handler_test.go sends a normal authenticated request to the settings endpoint and checks that the handler returns status 200 with a JSON preference object. The fixture uses an in-memory repository and responds immediately, so it does not exercise upstream delays or client retry behavior.",
                "possibly_relevant",
            ),
            (
                "avatar-timeout",
                "avatar_upload_test.go streams a large image to object storage and verifies that an upload timeout is reported. The test uses a separate storage client and endpoint; it never calls account settings and does not share the HTTP timeout policy under investigation.",
                "irrelevant",
            ),
        )),
    ),
    ExtendedContextTask(
        "extended-error-auth",
        "error_relevance",
        "Diagnose invalid authentication after refresh token rotation",
        _candidate_items((
            (
                "rotation",
                "401: refresh token has already been rotated. The request came from a session that obtained an access token before the latest refresh operation. The authentication service stores a replacement token and invalidates the previous one, so a retry with the old credential is expected to fail unless the client updates its session state.",
                "definitely_relevant",
            ),
            (
                "redis-limit",
                "Redis connection pool reached its limit during the deploy window. The affected process also uses Redis for short-lived session metadata and rate limits, but the log does not identify a failed token lookup or a missing session record. It could contribute indirectly through availability, but there is no direct authentication evidence.",
                "possibly_relevant",
            ),
            (
                "image-profile",
                "Image decoder rejected an unsupported PNG color profile while a user uploaded a profile picture. The exception is isolated to image processing and occurred after the request was authorized; it contains no token, session, refresh, or permission information.",
                "irrelevant",
            ),
        )),
    ),
)


ALL_RELEVANCE_TASKS = RELEVANCE_TASKS
ALL_EVALUATION_CASES = BOUNDED_CASES
TOTAL_RELEVANCE_CANDIDATES = sum(len(task.candidates) for task in RELEVANCE_TASKS)
TOTAL_EVALUATION_DECISIONS = (
    TOTAL_RELEVANCE_CANDIDATES
    + len(ACTION_CASES)
    + len(BOUNDED_CASES)
    + sum(len(case.candidates) * len(next(iter(case.candidates)).variants) for case in REPRESENTATION_CASES)
)
