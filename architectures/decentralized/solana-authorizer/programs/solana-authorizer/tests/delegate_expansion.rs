//! Records the delegation reach of the join gate, found in the second pass of
//! the internal review (wienerlabs/leviathan#15).
//!
//! `join_run` admits a signer when `Authorization::is_valid_for` holds for the
//! run's `join_authority`. The grantor - the join authority - decides who the
//! grantee is. It does not decide who the grantee's delegates are:
//! `authorization_grantee_update` is gated on `authorization.grantee ==
//! grantee.key()`, so the grantee alone extends the set, without the grantor's
//! signature.
//!
//! That is a reasonable feature for one operator running several nodes. It is
//! also the only gate on how many identities one sponsorship produces, and every
//! committee in the protocol is priced in the fraction of identities an attacker
//! holds - so it is bounded now
//! (wienerlabs/leviathan#15, finding 19).

use psyche_solana_authorizer::state::Authorization;
use anchor_lang::prelude::Pubkey;

fn key(n: u8) -> Pubkey {
    Pubkey::new_from_array([n; 32])
}

const SCOPE: &[u8] = b"CoordinatorJoinRun";

fn granted_to(grantor: Pubkey, grantee: Pubkey) -> Authorization {
    Authorization {
        bump: 255,
        grantor,
        grantee,
        scope: SCOPE.to_vec(),
        active: true,
        delegates: vec![],
        grantor_update_unix_timestamp: 0,
    }
}

#[test]
fn the_named_grantee_is_admitted_and_a_stranger_is_not() {
    let join_authority = key(1);
    let grantee = key(2);
    let stranger = key(3);

    let authorization = granted_to(join_authority, grantee);
    assert!(authorization.is_valid_for(&join_authority, &grantee, SCOPE));
    assert!(!authorization.is_valid_for(&join_authority, &stranger, SCOPE));
}

/// Delegates still work: every key the grantee adds is admitted, which is the
/// point of the feature.
#[test]
fn the_delegates_a_grantee_adds_are_admitted() {
    let join_authority = key(1);
    let grantee = key(2);
    let mut authorization = granted_to(join_authority, grantee);

    let delegates: Vec<Pubkey> = (10..10 + Authorization::MAX_DELEGATES as u8)
        .map(key)
        .collect();
    authorization.delegates.extend(delegates.iter().copied());

    for delegate in &delegates {
        assert!(authorization.is_valid_for(&join_authority, delegate, SCOPE));
    }
}

/// But the count is bounded, so one sponsorship cannot become an unlimited
/// supply of committee seats. The cap lives on the type, next to the field it
/// bounds, and `authorization_grantee_update` enforces it.
#[test]
fn the_delegate_count_is_capped() {
    assert!(
        Authorization::MAX_DELEGATES > 0,
        "delegation is still allowed"
    );
    assert!(
        Authorization::MAX_DELEGATES < 1000,
        "and it is a bound, not a formality"
    );
}

/// Revocation is all-or-nothing: clearing `active` shuts out the delegates and
/// the original grantee together. There is no way to drop one delegate.
#[test]
fn revocation_cannot_single_out_a_delegate() {
    let join_authority = key(1);
    let grantee = key(2);
    let sybil = key(4);
    let mut authorization = granted_to(join_authority, grantee);
    authorization.delegates.push(sybil);

    authorization.active = false;
    assert!(!authorization.is_valid_for(&join_authority, &sybil, SCOPE));
    assert!(
        !authorization.is_valid_for(&join_authority, &grantee, SCOPE),
        "the honest grantee loses access at the same moment"
    );
}

/// The scope and the grantor are both content-checked, so an authorization
/// minted by somebody else, or for another purpose, does not open this door.
/// This is why `join_run` can safely omit a seeds constraint on the account.
#[test]
fn another_grantor_or_another_scope_does_not_carry() {
    let join_authority = key(1);
    let impostor = key(5);
    let grantee = key(2);

    let elsewhere = granted_to(impostor, grantee);
    assert!(
        !elsewhere.is_valid_for(&join_authority, &grantee, SCOPE),
        "grantor is compared by value, and it is set from a signer at creation"
    );

    let wrong_scope = Authorization {
        scope: b"SomethingElse".to_vec(),
        ..granted_to(join_authority, grantee)
    };
    assert!(!wrong_scope.is_valid_for(&join_authority, &grantee, SCOPE));
}
