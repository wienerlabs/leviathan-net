use psyche_solana_treasurer::ID;

#[cfg(not(feature = "mainnet"))]
#[test]
fn a_default_build_carries_the_devnet_id() {
    assert_eq!(ID.to_string(), "9A1kc8Dr9dFJW9t1npAk7EHrADm6TAyFeVLH27CDdvv8");
}

#[cfg(feature = "mainnet")]
#[test]
fn a_mainnet_build_carries_the_mainnet_id() {
    assert_eq!(ID.to_string(), "A6Z8jZeKi81zUaozR7X7SGXtY8EyXm1YyTeFMuFgXEkW");
}

/// The treasurer calls the coordinator and the authorizer. If the mainnet
/// feature failed to reach them, a mainnet treasurer would sign CPIs against
/// devnet deployments, which is the kind of mistake that only surfaces once
/// real money is on the line.
#[cfg(feature = "mainnet")]
#[test]
fn the_mainnet_feature_reaches_every_program_this_one_calls() {
    assert_eq!(
        psyche_solana_coordinator::ID.to_string(),
        "9Sid2EWErkyMBKoqy9vzruRq6qJV2TUy9grp6NiieWN7",
    );
    assert_eq!(
        psyche_solana_authorizer::ID.to_string(),
        "2QXAd9g31vKFGSyxZC2wcjJdCZ4bjCdzrXA95H6Ft2eU",
    );
}
