use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use leviathan_multitude::{Contribution, ErrorReveal};
use serde::Serialize;
use tokio::net::TcpListener;

use crate::{Reject, Relay, Status};

pub type Shared = Arc<Mutex<Relay>>;
type Failure = (StatusCode, String);

const CLOSE_CHECK: Duration = Duration::from_millis(200);

fn lock(relay: &Shared) -> MutexGuard<'_, Relay> {
    relay.lock().unwrap_or_else(PoisonError::into_inner)
}

fn encode<T: Serialize + ?Sized>(value: &T) -> Vec<u8> {
    postcard::to_stdvec(value).expect("relay messages always serialize")
}

fn rejected(reject: Reject) -> Failure {
    let status = match reject {
        Reject::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
        Reject::Convicted(_) => StatusCode::FORBIDDEN,
        Reject::Full(_) => StatusCode::SERVICE_UNAVAILABLE,
        Reject::NoAudit(_) => StatusCode::NOT_FOUND,
    };
    (status, reject.to_string())
}

fn undecodable(err: impl std::fmt::Display) -> Failure {
    (StatusCode::BAD_REQUEST, format!("could not decode: {err}"))
}

fn unpublished(round: u32) -> Failure {
    (
        StatusCode::NOT_FOUND,
        format!("round {round} is not published"),
    )
}

pub fn router(relay: Shared) -> Router {
    let body_limit = lock(&relay).params().1.len() * 4 + 64 * 1024;
    Router::new()
        .route("/v1/run", get(run))
        .route("/v1/corpus", get(corpus))
        .route("/v1/round", get(round))
        .route("/v1/params", get(params))
        .route("/v1/contribute", post(contribute))
        .route("/v1/rounds/{round}", get(result))
        .route("/v1/rounds/{round}/contributions", get(contributions))
        .route("/v1/reveal", post(reveal))
        .route("/v1/status", get(status))
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(relay)
}

async fn run(State(relay): State<Shared>) -> Vec<u8> {
    encode(lock(&relay).info())
}

async fn corpus(State(relay): State<Shared>) -> Vec<u8> {
    lock(&relay).corpus().to_vec()
}

async fn round(State(relay): State<Shared>) -> Vec<u8> {
    encode(&lock(&relay).round(Instant::now()))
}

async fn params(State(relay): State<Shared>) -> Vec<u8> {
    encode(&lock(&relay).params())
}

async fn contribute(State(relay): State<Shared>, body: Bytes) -> Result<StatusCode, Failure> {
    let contribution = Contribution::from_bytes(&body).map_err(undecodable)?;
    lock(&relay).submit(contribution).map_err(rejected)?;
    Ok(StatusCode::ACCEPTED)
}

async fn result(State(relay): State<Shared>, Path(round): Path<u32>) -> Result<Vec<u8>, Failure> {
    lock(&relay)
        .result(round)
        .map(encode)
        .ok_or_else(|| unpublished(round))
}

async fn contributions(
    State(relay): State<Shared>,
    Path(round): Path<u32>,
) -> Result<Vec<u8>, Failure> {
    lock(&relay)
        .contributions(round)
        .map(encode)
        .ok_or_else(|| unpublished(round))
}

async fn reveal(State(relay): State<Shared>, body: Bytes) -> Result<String, Failure> {
    let reveal: ErrorReveal = postcard::from_bytes(&body).map_err(undecodable)?;
    let line = tokio::task::spawn_blocking(move || lock(&relay).reveal(reveal))
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .map_err(rejected)?;
    println!("{line}");
    Ok(line)
}

async fn status(State(relay): State<Shared>) -> Json<Status> {
    Json(lock(&relay).status())
}

async fn close_rounds(relay: Shared) {
    loop {
        tokio::time::sleep(CLOSE_CHECK).await;
        let shared = relay.clone();
        match tokio::task::spawn_blocking(move || lock(&shared).close_if_due(Instant::now())).await
        {
            Ok(Ok(lines)) => lines.iter().for_each(|line| println!("{line}")),
            Ok(Err(err)) => eprintln!("closing the round failed: {err}"),
            Err(err) => eprintln!("closing the round panicked: {err}"),
        }
    }
}

pub async fn serve(listener: TcpListener, relay: Relay) -> std::io::Result<()> {
    let shared: Shared = Arc::new(Mutex::new(relay));
    tokio::spawn(close_rounds(shared.clone()));
    axum::serve(listener, router(shared)).await
}
