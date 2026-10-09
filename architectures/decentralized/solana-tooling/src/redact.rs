use std::sync::OnceLock;

static GUARDED_URL: OnceLock<String> = OnceLock::new();

/// The scheme and host of an RPC URL, with any userinfo, path and query
/// replaced by `***`.
///
/// Hosted RPC providers put the API key in the URL, as a query parameter
/// (`?api-key=...`) or a path segment (`/v2/<key>`), so printing the URL a
/// tool connects to prints the key with it, and demo output ends up in screen
/// recordings. The host is enough to tell which provider a run used and never
/// enough to reuse its key. A moniker such as `devnet` passes through as is.
pub fn redact_rpc_url(url: &str) -> String {
    let (prefix, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (format!("{scheme}://"), rest),
        None => (String::new(), url),
    };
    let end = rest
        .find(|c| matches!(c, '/' | '?' | '#'))
        .unwrap_or(rest.len());
    let authority = &rest[..end];
    let remainder = &rest[end..];
    // Userinfo (`user:password@host`) is a credential too.
    let (host, had_userinfo) = match authority.rsplit_once('@') {
        Some((_, host)) => (host, true),
        None => (authority, false),
    };
    let hidden = had_userinfo || !(remainder.is_empty() || remainder == "/");
    if hidden {
        format!("{prefix}{host}/***")
    } else {
        format!("{prefix}{host}")
    }
}

/// Keeps `url` out of panic messages from here on.
///
/// An RPC error quotes the URL it failed to reach, key included, and the demos
/// unwrap their RPC results, so one failed call would print the key while the
/// process goes down. This installs a panic hook that scrubs the message. The
/// error a demo returns from `main` goes through [`scrub_rpc_url`] instead.
pub fn guard_rpc_url(url: &str) {
    if url.is_empty() || GUARDED_URL.set(url.to_string()).is_err() {
        return;
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        let Some(message) = message else {
            return default_hook(info);
        };
        let scrubbed = scrub_rpc_url(message);
        if scrubbed == message {
            return default_hook(info);
        }
        let location =
            info.location().map(ToString::to_string).unwrap_or_default();
        let thread = std::thread::current();
        eprintln!(
            "thread '{}' panicked at {location}:\n{scrubbed}",
            thread.name().unwrap_or("<unnamed>")
        );
    }));
}

/// `text` with the URL given to [`guard_rpc_url`] and its credentials
/// redacted. Unchanged when no URL is guarded.
pub fn scrub_rpc_url(text: &str) -> String {
    match GUARDED_URL.get() {
        Some(url) => scrub(text, url),
        None => text.to_string(),
    }
}

fn scrub(text: &str, url: &str) -> String {
    let mut out = text.replace(url, &redact_rpc_url(url));
    // Clients normalise a URL before quoting it (adding the `/` before the
    // query, for one), so the credential parts are scrubbed on their own too.
    for secret in credentials(url) {
        out = out.replace(secret, "***");
    }
    out
}

/// The parts of `url` that can carry a credential, as they appear in it: the
/// userinfo and whatever follows the host. Short parts are skipped, since they
/// are not keys and replacing them would garble the surrounding text.
fn credentials(url: &str) -> Vec<&str> {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let end = rest
        .find(|c| matches!(c, '/' | '?' | '#'))
        .unwrap_or(rest.len());
    let userinfo = rest[..end].rsplit_once('@').map(|(userinfo, _)| userinfo);
    let tail = rest[end..].trim_start_matches('/');
    userinfo
        .into_iter()
        .chain(Some(tail))
        .filter(|part| part.len() >= 8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::redact_rpc_url;
    use super::scrub;

    #[test]
    fn hides_a_key_in_the_query() {
        assert_eq!(
            redact_rpc_url(
                "https://devnet.helius-rpc.com/?api-key=0000-1111-2222"
            ),
            "https://devnet.helius-rpc.com/***"
        );
    }

    #[test]
    fn hides_a_key_in_the_path_or_userinfo() {
        assert_eq!(
            redact_rpc_url(
                "https://example.solana-devnet.quiknode.pro/abc123/"
            ),
            "https://example.solana-devnet.quiknode.pro/***"
        );
        assert_eq!(
            redact_rpc_url("https://user:secret@rpc.example.com:8899"),
            "https://rpc.example.com:8899/***"
        );
    }

    #[test]
    fn leaves_keyless_endpoints_readable() {
        assert_eq!(
            redact_rpc_url("https://api.devnet.solana.com"),
            "https://api.devnet.solana.com"
        );
        assert_eq!(
            redact_rpc_url("https://api.devnet.solana.com/"),
            "https://api.devnet.solana.com"
        );
        assert_eq!(redact_rpc_url("devnet"), "devnet");
    }

    #[test]
    fn scrubs_a_url_the_client_normalised() {
        // The env value has no `/` before the query; the client adds one
        // when it quotes the URL in an error.
        let url = "https://devnet.helius-rpc.com?api-key=0000-1111-2222";
        let error = "error sending request for url \
                     (https://devnet.helius-rpc.com/?api-key=0000-1111-2222)";
        assert_eq!(
            scrub(error, url),
            "error sending request for url (https://devnet.helius-rpc.com/***)"
        );
    }

    #[test]
    fn scrubs_userinfo() {
        let url = "https://user:secret-password@rpc.example.com:8899";
        let text = format!("cannot connect to {url}");
        let out = scrub(&text, url);
        assert!(!out.contains("secret-password"));
        assert_eq!(out, "cannot connect to https://rpc.example.com:8899/***");
    }

    #[test]
    fn leaves_text_alone_for_a_keyless_url() {
        let text =
            "error sending request for url (https://api.devnet.solana.com/)";
        assert_eq!(scrub(text, "https://api.devnet.solana.com"), text);
    }
}
