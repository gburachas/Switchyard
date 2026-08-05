// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Follow-up routing: re-decide the target on fresh user turns, hold it across tool chains.
//!
//! Task-level classification decides once, at the start of a conversation, and the
//! session then rides that verdict. In long agentic sessions the task moves underneath
//! it — a conversation that opened trivially becomes a hard debugging session ten turns
//! later, and the opening verdict is stale but still binding.
//!
//! This module re-classifies when the *human* asks something new, and pins the result
//! across the tool-call turns that follow, so a tool chain never flips target mid-flight.
//!
//! The pin lives in [`State::extra`] under [`PIN_KEY`], the same mechanism the escalation
//! router uses for its streak latch. It is deliberately *not* the [`affinity`] map:
//! that latch is first-wins and process-local, so it can never be refreshed.
//!
//! [`affinity`]: super::affinity

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use jsonptr::PointerBuf;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, Message, Request, Response, Role};

use super::affinity::first_user_message_hash;
use super::llm_judge::JudgePolicy;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::{State, StateValue};
use crate::observability;
use crate::{LibsyError, Result};

/// Session-state key holding the conversation's pinned target.
pub(crate) const PIN_KEY: &str = "followup_pinned_tier";

/// Prompt contract giving a re-classifying judge a referent for "the current tier".
const INCUMBENT_TIER_PREFIX: &str = "[current_tier]";

/// When a pinned target stops being reused.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RefreshPolicy {
    /// Classify once and hold that target for the whole conversation.
    PinOnce,
    /// Re-classify whenever the client appends a genuinely new user message.
    #[default]
    OnUserTurn,
}

/// Tier served when the judge abstains and there is nothing to fall back on.
///
/// Capability mode only: it names one of that mode's two tiers. Custom mode routes among
/// arbitrary labels and falls open to the route's `default_target` instead.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailOpenTier {
    /// Prefer correctness over cost — the safe default.
    #[default]
    Capable,
    /// Prefer cost over correctness.
    Efficient,
}

/// The tuning surface for follow-up routing.
///
/// Deliberately small. Thresholds are not repeated here: `followup` is a sub-table on the
/// existing `llm_classifier` route, so it reuses that route's `base_threshold` and
/// `min_confidence` rather than shadowing them.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FollowupConfig {
    /// When the pin is refreshed.
    pub refresh: RefreshPolicy,
    /// Trailing turns the judge sees on top of the client instructions and the task.
    pub recent_turn_window: Option<usize>,
    /// Tier served when the judge abstains with nothing to hold. Capability mode only.
    pub fail_open_tier: Option<FailOpenTier>,
    /// JSON Pointer to a boolean abstain flag in a custom verdict. Custom mode only.
    ///
    /// Custom schemas are user-supplied, so the abstain field has no fixed location the way
    /// it does in the capability verdict. Without this, an abstaining custom judge is
    /// indistinguishable from an unusable one and holds the incumbent instead of failing open.
    pub abstain_selector: Option<String>,
    /// Renders the judge payload as one condensed block capped at this many characters.
    ///
    /// Unset sends the conversation as a message array. Setting it reproduces the
    /// pre-migration Python payload contract, which the judge was tuned against: framing
    /// and tool results are collapsed to `[role]` lines under a budget that sheds the
    /// oldest window turn first. Verdicts are not comparable across the two shapes.
    pub judge_char_budget: Option<usize>,
    /// Keys the pin on the first user message when a request carries no session id.
    ///
    /// The pin normally lives in per-session state, which needs a session id to persist.
    /// Clients that send none — benchmark harnesses typically do not — would otherwise
    /// re-classify every turn, silently collapsing `on_user_turn` and `pin_once` into the
    /// same behaviour. Mirrors `message_hash_fallback` on the route itself.
    pub message_hash_fallback: bool,
}

impl FollowupConfig {
    /// Resolves the configured fail-open target, defaulting to the route's own fallback.
    ///
    /// `default_target` is already "what this route serves when the judge produces nothing
    /// usable", which is exactly the fail-open case; `fail_open_tier` only overrides it for
    /// capability mode, where the two tiers are named.
    pub(crate) fn fail_open_target(
        &self,
        default_target: &str,
        efficient: Option<&str>,
        capable: Option<&str>,
    ) -> String {
        match (self.fail_open_tier, efficient, capable) {
            (Some(FailOpenTier::Efficient), Some(efficient), _) => efficient.to_string(),
            (Some(FailOpenTier::Capable), _, Some(capable)) => capable.to_string(),
            _ => default_target.to_string(),
        }
    }

    /// Rejects settings that would leave the judge with nothing useful to read.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.recent_turn_window == Some(0) {
            return Err(LibsyError::AlgorithmError {
                message: "followup recent_turn_window must be at least 1".to_string(),
            });
        }
        Ok(())
    }

    /// Validates against capability mode, which reads abstain from the typed verdict.
    ///
    /// # Errors
    ///
    /// Returns an error for `abstain_selector`, which would silently do nothing here.
    pub(crate) fn validate_capability(&self) -> Result<()> {
        self.validate()?;
        if self.abstain_selector.is_some() {
            return Err(LibsyError::AlgorithmError {
                message: "followup abstain_selector applies to mode custom only; the \
                          capability verdict carries its own abstain field"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Validates against custom mode, which routes among labels rather than two tiers.
    ///
    /// # Errors
    ///
    /// Returns an error for `fail_open_tier`, which names tiers custom mode does not have.
    pub(crate) fn validate_custom(&self) -> Result<()> {
        self.validate()?;
        if self.fail_open_tier.is_some() {
            return Err(LibsyError::AlgorithmError {
                message: "followup fail_open_tier applies to mode capability only; mode \
                          custom fails open to default_target"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Parses `abstain_selector` into a JSON Pointer, if one is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the selector is not a valid JSON Pointer, or points at the
    /// verdict root rather than a field within it.
    pub(crate) fn abstain_pointer(&self) -> Result<Option<PointerBuf>> {
        let Some(selector) = self.abstain_selector.as_ref() else {
            return Ok(None);
        };
        let pointer = PointerBuf::parse(selector).map_err(|error| LibsyError::AlgorithmError {
            message: format!("followup abstain_selector is not a valid JSON Pointer: {error}"),
        })?;
        if pointer.is_root() {
            return Err(LibsyError::AlgorithmError {
                message: "followup abstain_selector must identify a response field".to_string(),
            });
        }
        Ok(Some(pointer))
    }
}

/// Whether `request` ends in a genuinely new user message, as opposed to a tool result
/// being fed back.
///
/// This is the trigger for re-classification: re-decide when the human asked something
/// new, keep tool-chain turns on the pinned target.
///
/// libsy receives messages already normalized by `switchyard-protocol`, so unlike the
/// wire-format-aware Python equivalent this needs no per-format branching — an Anthropic
/// `tool_result` block and an OpenAI `function_call_output` both arrive as
/// [`ContentBlock::ToolResult`]. An empty conversation counts as a new turn, so an
/// unrecognized shape re-classifies rather than riding a stale pin.
pub(crate) fn is_new_user_turn(request: &Request) -> bool {
    let Some(last) = request.llm_request.messages.last() else {
        return true;
    };
    if last.role != Role::User {
        return false;
    }
    !last
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult(_)))
}

/// The target pinned to this conversation, regardless of refresh policy.
///
/// This is the *incumbent* — a fresh user turn hides the pin from routing but not from
/// here, because a re-classification still wants to know what it is replacing.
pub(crate) fn pinned_tier(state: &State) -> Option<String> {
    match state.extra.get(PIN_KEY) {
        Some(StateValue::String(tier)) => Some(tier.clone()),
        _ => None,
    }
}

/// Pins `tier` to this conversation, returning the target it replaced.
///
/// The previous value is what a caller counts tier flips from.
pub(crate) fn set_pinned_tier(state: &mut State, tier: &str) -> Option<String> {
    let previous = pinned_tier(state);
    state
        .extra
        .insert(PIN_KEY.to_string(), StateValue::String(tier.to_string()));
    previous
}

/// A single decisive score for `target`.
pub(crate) fn decisive(target: &str) -> Classification {
    Classification::Scores(vec![Score {
        target: target.to_string(),
        confidence: 1.0,
    }])
}

/// Selects the messages a re-classifying judge sees, plus the incumbent-tier line.
///
/// Mirrors the capability classifier's view and appends the incumbent so the prompt can
/// express "stay on the current tier unless…" — the judge is stateless, so hysteresis has
/// to be carried in the prompt.
pub(crate) struct FollowupInput<I> {
    /// The mode's own message view, reused rather than reimplemented.
    pub(crate) inner: I,
}

impl<I: super::llm_judge::ClassifierInput> super::llm_judge::ClassifierInput for FollowupInput<I> {
    fn build_messages(&self, state: &State, request: &Request) -> Vec<Message> {
        let mut messages = self.inner.build_messages(state, request);
        // Read from state, not from the caller: the judge is consulted through
        // `JudgeClassifier`, which owns the call and passes state through.
        if let Some(incumbent) = pinned_tier(state) {
            messages.push(Message::text(
                Role::User,
                format!("{INCUMBENT_TIER_PREFIX} {incumbent}"),
            ));
        }
        messages
    }
}

/// Routes an explicit abstain in a custom verdict to the fail-open target.
///
/// Custom mode's [`TargetSelectorPolicy`](super::target_selector::TargetSelectorPolicy) reads
/// only the routing field, so an abstaining verdict reaches it as an unmapped label and
/// becomes [`Classification::Ambiguous`] — indistinguishable from a malformed one. This
/// reads the abstain flag first so asymmetry 1 holds: a judge that explicitly declines to
/// decide overrides the incumbent rather than being held under it.
pub(crate) struct AbstainSelectorPolicy<P> {
    pub(crate) inner: P,
    pub(crate) abstain: PointerBuf,
    pub(crate) fail_open: String,
}

impl<P: JudgePolicy<Verdict = Value>> JudgePolicy for AbstainSelectorPolicy<P> {
    type Verdict = Value;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        let abstained = verdict
            .and_then(|verdict| self.abstain.resolve(verdict).ok())
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if abstained {
            return decisive(&self.fail_open);
        }
        self.inner.to_classification(verdict)
    }
}

/// Re-classifies on fresh user turns and pins the result across the tool-call turns that
/// follow, so a tool chain never flips target mid-flight.
///
/// Wraps the mode's own classifier, so capability and custom modes share one implementation.
pub(crate) struct FollowupClassifier {
    /// The mode's classifier, consulted only when the pin is not being reused.
    pub(crate) inner: Arc<dyn Classifier<State>>,
    pub(crate) refresh: RefreshPolicy,
    /// Target served when the judge produces nothing usable and nothing is pinned.
    pub(crate) fail_open: String,
    /// Algorithm name used to label tier-flip metrics.
    pub(crate) algorithm: &'static str,
    /// Pins for conversations that arrive without a session id, keyed by a hash of the
    /// first user message. `None` unless `message_hash_fallback` is configured.
    ///
    /// Session-keyed state cannot serve these: it needs a session id to persist, so every
    /// turn would arrive with a fresh `State` and re-classify.
    pub(crate) hashed_pins: Option<Mutex<HashMap<String, String>>>,
}

/// Upper bound on hash-keyed pins, keeping the process-local map from growing without
/// limit; one arbitrary entry is evicted once the bound is reached.
const MAX_HASHED_PINS: usize = 4096;

impl FollowupClassifier {
    /// The hash key for a request, when the pin cannot live in session state.
    ///
    /// Returns `None` whenever session state will do — a request carrying a session id is
    /// already keyed correctly, and reading it from two places would let the two disagree.
    fn hash_key(&self, request: &Request) -> Option<String> {
        self.hashed_pins.as_ref()?;
        if session_id(request).is_some() {
            return None;
        }
        first_user_message_hash(request)
    }

    /// The pinned target for this conversation, from whichever store holds it.
    fn pin(&self, state: &State, request: &Request) -> Option<String> {
        match self.hash_key(request) {
            Some(key) => self.hashed_pins.as_ref()?.lock().get(&key).cloned(),
            None => pinned_tier(state),
        }
    }

    /// Pins `tier`, returning the target it replaced.
    fn set_pin(&self, state: &mut State, request: &Request, tier: &str) -> Option<String> {
        let Some(key) = self.hash_key(request) else {
            return set_pinned_tier(state, tier);
        };
        let Some(pins) = self.hashed_pins.as_ref() else {
            return set_pinned_tier(state, tier);
        };
        let mut pins = pins.lock();
        if pins.len() >= MAX_HASHED_PINS
            && !pins.contains_key(&key)
            && let Some(evicted) = pins.keys().next().cloned()
        {
            pins.remove(&evicted);
        }
        pins.insert(key, tier.to_string())
    }
}

/// The request's session id, if it carries a non-empty one.
fn session_id(request: &Request) -> Option<&str> {
    request
        .metadata
        .as_ref()?
        .session_id
        .as_deref()
        .filter(|id| !id.is_empty())
}

#[async_trait]
impl Classifier<State> for FollowupClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        self.inner.routing_tier(selected_model)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        // The incumbent is read regardless of policy: a fresh turn hides the pin from
        // routing, but a re-classification still wants to know what it is replacing.
        let incumbent = self.pin(state, request);

        // `pin_once` reuses the pin forever; `on_user_turn` reuses it only while the
        // client is feeding tool results back.
        let reuse_pin = match self.refresh {
            RefreshPolicy::PinOnce => true,
            RefreshPolicy::OnUserTurn => !is_new_user_turn(request),
        };
        if reuse_pin && let Some(tier) = &incumbent {
            return Ok((decisive(tier), None));
        }

        // Asymmetry 3: one judge call, whatever it returns. The judge is consulted once
        // and its outcome interpreted; neither branch below calls it again.
        let (classification, response) = self.inner.score(state, request, driver).await?;
        // Asymmetry 2: an ambiguous verdict is "no evidence", which is not evidence of
        // change — hold the incumbent, and fail open only with nothing to hold.
        // Asymmetry 1 is resolved before this point, by the mode's abstain-aware policy
        // turning an explicit abstain into a decisive fail-open score.
        let tier = match classification.argmax(false)? {
            Some(score) => score.target,
            None => incumbent.unwrap_or_else(|| self.fail_open.clone()),
        };
        // A re-pin to a different target is a flip; the first pin of a conversation is
        // not, since there was nothing to move away from.
        if let Some(previous) = self.set_pin(state, request, &tier)
            && previous != tier
        {
            observability::record_tier_flip(self.algorithm, &previous, &tier);
        }
        Ok((decisive(&tier), response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{LlmRequest, ToolCall, ToolResult};

    fn request_ending_in(message: Message) -> Request {
        Request {
            llm_request: LlmRequest {
                messages: vec![Message::text(Role::User, "the original task"), message],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn a_plain_user_message_is_a_new_turn() {
        let request = request_ending_in(Message::text(Role::User, "and now do this"));
        assert!(is_new_user_turn(&request));
    }

    #[test]
    fn a_tool_result_carried_as_a_user_message_is_not_a_new_turn() {
        // The Anthropic shape: a tool result transported with role=user. Treating it
        // as a new turn would re-classify mid tool-chain and flip tiers.
        let request = request_ending_in(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "exit 0".to_string(),
                }],
                is_error: None,
            })],
        });
        assert!(!is_new_user_turn(&request));
    }

    #[test]
    fn an_assistant_turn_is_not_a_new_user_turn() {
        let request = request_ending_in(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "bash".to_string(),
                arguments: json!({"cmd": "ls"}),
            })],
        });
        assert!(!is_new_user_turn(&request));
    }

    #[test]
    fn an_empty_conversation_re_classifies_rather_than_riding_a_stale_pin() {
        let request = Request {
            llm_request: LlmRequest::default(),
            raw_request: None,
            metadata: None,
        };
        assert!(is_new_user_turn(&request));
    }

    #[test]
    fn the_pin_round_trips_and_reports_what_it_replaced() {
        let mut state = State::default();
        assert_eq!(pinned_tier(&state), None);

        assert_eq!(set_pinned_tier(&mut state, "weak"), None);
        assert_eq!(pinned_tier(&state), Some("weak".to_string()));

        // The previous value is what a flip counter reads.
        assert_eq!(
            set_pinned_tier(&mut state, "strong"),
            Some("weak".to_string())
        );
        assert_eq!(pinned_tier(&state), Some("strong".to_string()));
    }

    #[test]
    fn a_zero_window_is_rejected() {
        let config = FollowupConfig {
            recent_turn_window: Some(0),
            ..FollowupConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(FollowupConfig::default().validate().is_ok());
    }

    #[test]
    fn the_default_refresh_policy_is_on_user_turn() {
        // The whole point of the router; a `pin_once` default would silently make it
        // behave like stock sticky routing.
        assert_eq!(FollowupConfig::default().refresh, RefreshPolicy::OnUserTurn);
    }

    #[test]
    fn fail_open_defaults_to_the_route_fallback_and_is_overridden_by_tier() {
        // Custom mode names no tiers, so the route's own fallback is the fail-open target.
        let config = FollowupConfig::default();
        assert_eq!(config.fail_open_target("default", None, None), "default");

        // Capability mode names both, and an explicit tier selects between them.
        let efficient = FollowupConfig {
            fail_open_tier: Some(FailOpenTier::Efficient),
            ..FollowupConfig::default()
        };
        assert_eq!(
            efficient.fail_open_target("strong", Some("weak"), Some("strong")),
            "weak"
        );
        let capable = FollowupConfig {
            fail_open_tier: Some(FailOpenTier::Capable),
            ..FollowupConfig::default()
        };
        assert_eq!(
            capable.fail_open_target("weak", Some("weak"), Some("strong")),
            "strong"
        );
    }

    #[test]
    fn an_abstain_selector_must_be_a_pointer_to_a_field() -> Result<()> {
        assert!(FollowupConfig::default().abstain_pointer()?.is_none());

        let valid = FollowupConfig {
            abstain_selector: Some("/abstain".to_string()),
            ..FollowupConfig::default()
        };
        assert!(valid.abstain_pointer()?.is_some());

        // The root would resolve to the whole verdict, which is never a boolean flag.
        let root = FollowupConfig {
            abstain_selector: Some("".to_string()),
            ..FollowupConfig::default()
        };
        assert!(root.abstain_pointer().is_err());

        let malformed = FollowupConfig {
            abstain_selector: Some("abstain".to_string()),
            ..FollowupConfig::default()
        };
        assert!(malformed.abstain_pointer().is_err());
        Ok(())
    }
}

// ── Condensed judge payload ────────────────────────────────────────────────

/// Per-message cap for system/developer framing, which carries no task signal.
const SYSTEM_CLIP_CAP: usize = 1_500;

/// Cap for the first user message — the task statement, so it gets a wider share.
const TASK_ANCHOR_CLIP_CAP: usize = 2_500;

/// The newest turn is never clipped below this: a verdict on a question reduced to
/// nothing is worse than a slightly over-budget payload.
const NEWEST_TURN_FLOOR: usize = 500;

/// Marks a message clipped from its tail.
const TRUNCATION_SUFFIX: &str = "...<truncated>";

/// Marks a message clipped from its head, keeping the tail.
const TAIL_KEPT_PREFIX: &str = "<truncated>...";

/// Characters reserved for a truncation marker when clipping.
///
/// Deliberately 16 rather than the markers' own 14: the payload contract this reproduces
/// reserved a fixed 16, so a clipped line lands two characters under the cap. Matching the
/// arithmetic — not just the intent — is what keeps verdicts comparable across the port.
const CLIP_RESERVE: usize = 16;

/// Flattens one message's content to plain text, the way the judge payload renders it.
///
/// Tool results and images collapse to placeholders: the classifier routes on the shape
/// of the conversation, and a build log pasted whole would crowd out the actual question.
fn text_of(message: &Message) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } => parts.push(text),
            ContentBlock::ToolResult(_) => parts.push("<tool_result>"),
            ContentBlock::Image { .. } => parts.push("<image>"),
            _ => {}
        }
    }
    parts.join(" ")
}

/// Clips from the tail, marking the cut.
fn clip_head(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let keep = limit.saturating_sub(CLIP_RESERVE);
    text.chars().take(keep).collect::<String>() + TRUNCATION_SUFFIX
}

/// Clips from the head, keeping the tail — used for the newest turn, whose end holds
/// the fresh question or result.
fn clip_tail(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let keep = limit.saturating_sub(CLIP_RESERVE);
    let skip = count.saturating_sub(keep);
    TAIL_KEPT_PREFIX.to_string() + &text.chars().skip(skip).collect::<String>()
}

/// Renders the anchor-plus-window view of a conversation as one text block.
///
/// The judge sees a single condensed message rather than the conversation itself:
/// `[system]` framing, the `[task]` anchor, then the trailing window as `[role]` lines.
/// When the result exceeds `budget` it sheds the oldest window turn first, so the newest
/// turn — the one the verdict is about — is the last thing to go.
pub(crate) fn condense(messages: &[Message], window: usize, budget: usize) -> String {
    let is_instruction = |m: &Message| matches!(m.role, Role::System | Role::Developer);
    let system_parts: Vec<String> = messages
        .iter()
        .filter(|m| is_instruction(m))
        .map(text_of)
        .collect();
    let first_user = messages.iter().position(|m| m.role == Role::User);
    let tail: Vec<&Message> = messages
        .iter()
        .enumerate()
        .filter(|(index, m)| !is_instruction(m) && Some(*index) != first_user)
        .map(|(_, m)| m)
        .collect();

    let window = if window > 0 { window } else { 1 };
    let window_msgs = &tail[tail.len().saturating_sub(window)..];

    let mut lines: Vec<String> = Vec::new();
    if !system_parts.is_empty() {
        lines.push(clip_head(
            &format!("[system] {}", system_parts.join(" ")),
            SYSTEM_CLIP_CAP.min(budget / 5),
        ));
    }
    if let Some(index) = first_user {
        lines.push(clip_head(
            &format!("[task] {}", text_of(&messages[index])),
            TASK_ANCHOR_CLIP_CAP.min(budget / 3),
        ));
    }
    let mut win_lines: Vec<String> = window_msgs
        .iter()
        .map(|m| format!("[{}] {}", role_label(m.role), text_of(m)))
        .collect();

    while !win_lines.is_empty() {
        let text = lines
            .iter()
            .chain(win_lines.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if text.chars().count() <= budget {
            return text;
        }
        if win_lines.len() == 1 {
            // One oversized newest turn: keep its tail rather than drop the question.
            let head = lines.join("\n");
            let room = budget
                .saturating_sub(head.chars().count() + 1)
                .max(NEWEST_TURN_FLOOR);
            return head + "\n" + &clip_tail(&win_lines[0], room);
        }
        win_lines.remove(0);
    }
    lines.join("\n").chars().take(budget).collect()
}

/// The lowercase role name used in a condensed `[role]` line.
fn role_label(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Developer => "developer",
        Role::Tool => "tool",
    }
}

/// Presents the judge with one condensed message instead of the conversation itself.
///
/// This is the payload contract the pre-migration Python classifier used, and matching it
/// matters: a judge shown a raw message array with full tool-result payloads reads a
/// different conversation than one shown a budgeted `[task]`/`[role]` digest, and routes
/// differently. The incumbent tier is appended into the same block, not sent separately.
pub(crate) struct CondensedInput {
    pub(crate) recent_turn_window: usize,
    pub(crate) budget: usize,
    /// Appended as its own trailing message, keeping the routing instruction last.
    pub(crate) reinforcement: &'static str,
}

impl super::llm_judge::ClassifierInput for CondensedInput {
    fn build_messages(&self, state: &State, request: &Request) -> Vec<Message> {
        let mut blob = condense(
            &request.llm_request.messages,
            self.recent_turn_window,
            self.budget,
        );
        if let Some(incumbent) = pinned_tier(state) {
            blob.push('\n');
            blob.push_str(&format!("{INCUMBENT_TIER_PREFIX} {incumbent}"));
        }
        vec![
            Message::text(Role::User, blob),
            Message::text(Role::User, self.reinforcement.to_string()),
        ]
    }
}

#[cfg(test)]
mod condense_tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, ToolResult};

    fn msg(role: Role, text: &str) -> Message {
        Message::text(role, text.to_string())
    }

    #[test]
    fn renders_the_anchor_and_window_as_labelled_lines() {
        let messages = vec![
            msg(Role::System, "you are support"),
            msg(Role::User, "my data is down"),
            msg(Role::Assistant, "checking"),
            msg(Role::User, "still down"),
        ];
        // Window of 2 keeps the two newest non-anchor turns; the anchor always survives.
        assert_eq!(
            condense(&messages, 2, 8_000),
            "[system] you are support\n[task] my data is down\n[assistant] checking\n[user] still down"
        );
    }

    #[test]
    fn drops_the_oldest_window_turn_before_the_newest() {
        let messages = vec![
            msg(Role::User, "task"),
            msg(Role::Assistant, "x".repeat(200).as_str()),
            msg(Role::User, "the fresh question"),
        ];
        let out = condense(&messages, 2, 60);
        // The newest turn is what the verdict is about, so it is the last thing shed.
        assert!(out.contains("the fresh question"), "{out}");
        assert!(!out.contains(&"x".repeat(200)), "{out}");
    }

    #[test]
    fn keeps_the_tail_of_a_single_oversized_newest_turn() {
        let messages = vec![
            msg(Role::User, "task"),
            msg(Role::User, &("y".repeat(2_000) + " THE QUESTION")),
        ];
        // The budget must exceed NEWEST_TURN_FLOOR for clipping to engage at all: below
        // it the floor wins and the turn is passed through whole, deliberately.
        let out = condense(&messages, 1, 800);
        // Head-clipping would delete the question; the tail is kept instead.
        assert!(out.contains("THE QUESTION"), "{out}");
        assert!(out.contains(TAIL_KEPT_PREFIX), "{out}");
    }

    #[test]
    fn tool_results_collapse_to_a_placeholder() {
        // A build log pasted whole would crowd out the question it is evidence for.
        let tool_turn = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "c1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "x".repeat(5_000),
                }],
                is_error: None,
            })],
        };
        let out = condense(&[msg(Role::User, "task"), tool_turn], 1, 8_000);
        assert!(out.ends_with("[user] <tool_result>"), "{out}");
    }

    #[test]
    fn the_judge_sees_one_block_plus_the_reinforcement() {
        use super::super::llm_judge::ClassifierInput;
        let input = CondensedInput {
            recent_turn_window: 4,
            budget: 8_000,
            reinforcement: "ROUTE IT",
        };
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![msg(Role::User, "task"), msg(Role::User, "newest")],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        let mut state = State::default();
        set_pinned_tier(&mut state, "weak");

        let built = input.build_messages(&state, &request);
        // Exactly two: the condensed block, then the routing instruction last.
        assert_eq!(built.len(), 2);
        let blob = built[0].text_content("").unwrap_or_default();
        // The incumbent rides inside the block, not as a separate message.
        assert!(blob.ends_with("[current_tier] weak"), "{blob}");
        assert_eq!(built[1].text_content("").as_deref(), Some("ROUTE IT"));
    }

    /// Emits condensed payloads for fixtures so they can be diffed against the Python
    /// implementation this port replicates. Inert unless `CONDENSE_PARITY_IN` is set, so
    /// it costs nothing in a normal run but keeps the comparison reproducible.
    #[test]
    fn emit_condensed_payloads_for_parity() {
        let Ok(input) = std::env::var("CONDENSE_PARITY_IN") else {
            return;
        };
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&input).unwrap_or_default())
                .unwrap_or(serde_json::Value::Null);
        let mut out: Vec<String> = Vec::new();
        for case in cases.as_array().map(Vec::as_slice).unwrap_or_default() {
            let messages: Vec<Message> = case["messages"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .map(|m| {
                    let role = match m["role"].as_str().unwrap_or("user") {
                        "system" => Role::System,
                        "developer" => Role::Developer,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::User,
                    };
                    Message::text(role, m["content"].as_str().unwrap_or_default().to_string())
                })
                .collect();
            out.push(condense(
                &messages,
                case["window"].as_u64().unwrap_or(4) as usize,
                case["budget"].as_u64().unwrap_or(8_000) as usize,
            ));
        }
        if let Ok(path) = std::env::var("CONDENSE_PARITY_OUT") {
            let _ = std::fs::write(path, serde_json::to_string(&out).unwrap_or_default());
        }
    }
}
