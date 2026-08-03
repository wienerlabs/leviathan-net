//! Records the delegation reach of the join gate, found in the second pass of
//! the internal review (wienerlabs/leviathan#15).
//!
//! `join_run` admits a signer when `Authorization::is_valid_for` holds for the
//! run's `join_authority`. The grantor - the join authority - decides who the
//! grantee is. It does not decide who the grantee's delegates are:
//! `authorization_grantee_update` is gated on `authorization.grantee ==
//! grantee.key()`, so the grantee alone extends the set, without a signature or
//! a cap.
//!
//! That is a reasonable feature for one operator running several nodes. It is
//! also the sybil gate for every committee in the protocol, and the two
//! readings should not be confused: sponsoring one participant sponsors as many
//! identities as that participant cares to create.

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

/// Every key the grantee adds is admitted on the same authorization. The join
/// authority signed once, for one key, and gets as many participants as the
/// grantee wants.
#[test]
fn every_key_the_grantee_adds_is_admitted_too() {
    let join_authority = key(1);
    let grantee = key(2);
    let mut authorization = granted_to(join_authority, grantee);

    let sybils: Vec<Pubkey> = (10..74u8).map(key).collect();
    // `authorization_grantee_update` is exactly this, gated only on the grantee
    // signing: no grantor approval, no cap on the count.
    authorization.delegates.extend(sybils.iter().copied());

    for sybil in &sybils {
        assert!(
            authorization.is_valid_for(&join_authority, sybil, SCOPE),
            "a delegate the join authority never saw passes the gate"
        );
    }
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
