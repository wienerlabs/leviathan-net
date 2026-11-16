use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;

use crate::exposition::render;
use crate::exposition::CONTENT_TYPE;

/// Serves `registry` at `/metrics` on `listener`, for Prometheus to scrape.
/// Returns only if the server fails.
pub async fn serve(listener: TcpListener, registry: Arc<Registry>) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/metrics", get(scrape))
        .with_state(registry);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn scrape(State(registry): State<Arc<Registry>>) -> Response {
    match render(&registry) {
        Ok(body) => (StatusCode::OK, [(header::CONTENT_TYPE, CONTENT_TYPE)], body).into_response(),
        // A scrape that cannot be encoded fails loudly: an empty 200 would
        // read as "every series vanished" rather than as an exporter fault.
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("the metrics could not be encoded: {err}"),
        )
            .into_response(),
    }
}
