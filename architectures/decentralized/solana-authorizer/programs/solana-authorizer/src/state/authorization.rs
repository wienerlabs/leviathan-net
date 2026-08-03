use anchor_lang::prelude::*;

#[account()]
#[derive(Debug)]
pub struct Authorization {
    pub bump: u8,

    pub grantor: Pubkey,
    pub grantee: Pubkey,
    pub scope: Vec<u8>,

    pub active: bool,
    pub delegates: Vec<Pubkey>,

    pub grantor_update_unix_timestamp: i64,
}

impl Authorization {
    pub const SEEDS_PREFIX: &'static [u8] = b"Authorization";

    /// How many keys a grantee may add to its own authorization.
    ///
    /// Delegation is a real convenience - one operator, several nodes - but it
    /// is also the only gate on how many identities a single sponsorship
    /// produces, and every committee in the protocol is priced in the fraction
    /// of identities an attacker holds. Uncapped, the join authority approved
    /// one key and got as many as that key cared to create
    /// (wienerlabs/leviathan#15, finding 19).
    ///
    /// The number is deliberately generous - the same 64 the verdict and appeal
    /// voter lists use, and well above what `memnet_authorizer_full_cycle`
    /// exercises - because this is a bound on unbounded growth, not a redesign
    /// of what delegation is for. Before it, the only limit was Solana's 10 MB
    /// account ceiling, which is some three hundred thousand keys.
    ///
    /// Whether 64 is the right number for a given run's *economics* is the
    /// operator's call and belongs with the committee sizing in
    /// `docs/COMMITTEE_ECONOMICS.md`. This only stops the count being unbounded.
    pub const MAX_DELEGATES: usize = 64;

    pub fn space_with_discriminator(
        scope_len: usize,
        delegates_len: usize,
    ) -> usize {
        8 + std::mem::size_of::<bool>()
            + std::mem::size_of::<Pubkey>()
            + std::mem::size_of::<Pubkey>()
            + (4 + scope_len * std::mem::size_of::<u8>())
            + std::mem::size_of::<bool>()
            + (4 + delegates_len * std::mem::size_of::<Pubkey>())
            + std::mem::size_of::<i64>()
    }

    pub fn is_valid_for(
        &self,
        grantor: &Pubkey,
        grantee: &Pubkey,
        scope: &[u8],
    ) -> bool {
        if !self.active {
            return false;
        }
        if !self.grantor.eq(grantor) {
            return false;
        }
        if !self.scope.eq(scope) {
            return false;
        }
        self.grantee == Pubkey::default()
            || self.grantee.eq(grantee)
            || self.delegates.contains(grantee)
    }
}
