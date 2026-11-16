#![cfg(feature = "server")]

use std::sync::Arc;

use leviathan_telemetry::exposition::CONTENT_TYPE;
use leviathan_telemetry::server::serve;
use leviathan_telemetry::verifier::PassReport;
use leviathan_telemetry::verifier::VerifierMetrics;
use prometheus_client::registry::Registry;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

/// One HTTP/1.1 request over a plain socket, as Prometheus makes it, and the
/// whole response back.
async fn get(address: std::net::SocketAddr, path: &str) -> String {
    let mut stream = TcpStream::connect(address).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    response
}

#[tokio::test]
async fn the_endpoint_serves_what_the_registry_holds_as_it_changes() {
    let mut registry = Registry::default();
    let metrics = VerifierMetrics::register(&mut registry);
    let registry = Arc::new(registry);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve(listener, registry.clone()));

    metrics.record(
        "run-a",
        &PassReport {
            fraud_found: 1,
            submission_failures: 1,
            ..PassReport::default()
        },
    );
    let first = get(address, "/metrics").await;
    assert!(first.starts_with("HTTP/1.1 200"), "{first}");
    assert!(
        first
            .to_ascii_lowercase()
            .contains(&format!("content-type: {CONTENT_TYPE}")),
        "{first}"
    );
    assert!(
        first.contains(r#"leviathan_verifier_submission_failures_total{run_id="run-a"} 1"#),
        "{first}"
    );

    // A scrape reads the registry as it stands, not as it stood at startup.
    metrics.pass_failed("run-a");
    let second = get(address, "/metrics").await;
    assert!(
        second.contains(r#"leviathan_verifier_pass_errors_total{run_id="run-a"} 1"#),
        "{second}"
    );

    let missing = get(address, "/elsewhere").await;
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");

    server.abort();
}
