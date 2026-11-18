use crate::errors::UploadError;
use crate::hub::model::HubRepo;
use hf_hub::{
    Cache, Repo, RepoType,
    api::{
        Siblings,
        tokio::{ApiError, UploadSource},
    },
};
use psyche_coordinator::model;
use psyche_core::FixedString;
use std::{path::PathBuf, time::Instant};
use tokio::sync::mpsc;
use tracing::{error, info};

const MODEL_EXTENSIONS: [&str; 3] = [".safetensors", ".json", ".py"];
const DATASET_EXTENSIONS: [&str; 1] = [".parquet"];

/// Strip leading/trailing whitespace and control characters from a repo identifier.
/// TODO: Remove once https://github.com/PsycheFoundation/nousnet/pull/636 is merged
fn sanitize_repo_id(raw: &str) -> String {
    raw.trim_matches(|c: char| c.is_whitespace() || c.is_control())
        .to_string()
}

fn check_extensions(sibling: &Siblings, extensions: &[&'static str]) -> bool {
    match extensions.is_empty() {
        true => true,
        false => {
            for ext in extensions {
                if sibling.rfilename.ends_with(ext) {
                    return true;
                }
            }
            false
        }
    }
}

async fn download_repo_async(
    repo: Repo,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
    extensions: &[&'static str],
) -> Result<Vec<PathBuf>, ApiError> {
    let builder = hf_hub::api::tokio::ApiBuilder::new();
    let cache = match cache {
        Some(cache) => Cache::new(cache),
        None => Cache::default(),
    };
    let api = builder
        .with_cache_dir(cache.path().clone())
        .with_token(token.or(cache.token()))
        .with_progress(progress_bar)
        .build()?
        .repo(repo);
    let siblings = api
        .info()
        .await?
        .siblings
        .into_iter()
        .filter(|x| check_extensions(x, extensions))
        .collect::<Vec<_>>();
    let mut ret: Vec<PathBuf> = Vec::new();
    for chunk in siblings.chunks(max_concurrent_downloads.unwrap_or(siblings.len())) {
        let futures = chunk
            .iter()
            .map(|x| async {
                let start_time = Instant::now();
                tracing::debug!(filename = x.rfilename, "Starting file download from hub");
                let res =
                    retry_lost_link_race_async(|| api.get(&x.rfilename), tokio_io_error).await;
                if res.is_ok() {
                    let duration_secs = (Instant::now() - start_time).as_secs_f32();
                    tracing::info!(
                        filename = x.rfilename,
                        duration_secs = duration_secs,
                        "Finished downloading file from hub"
                    );
                }
                res
            })
            .collect::<Vec<_>>();
        for future in futures {
            ret.push(future.await?);
        }
    }
    Ok(ret)
}

pub async fn download_model_repo_async(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_async(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Model, revision),
            None => Repo::model(repo_id),
        },
        cache,
        token,
        max_concurrent_downloads,
        progress_bar,
        &MODEL_EXTENSIONS,
    )
    .await
}

pub async fn download_dataset_repo_async(
    repo_id: String,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, ApiError> {
    let repo_id = sanitize_repo_id(&repo_id);
    download_repo_async(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Dataset, revision),
            None => Repo::new(repo_id, RepoType::Dataset),
        },
        cache,
        token,
        max_concurrent_downloads,
        progress_bar,
        &DATASET_EXTENSIONS,
    )
    .await
}

/// Whether `error` is the `AlreadyExists` that a concurrent download of the
/// same file into the same cache leaves behind.
///
/// hf-hub moves a finished file into the cache, then links it into the
/// revision's snapshot after checking that no link is there yet. Two downloads
/// that both pass that check race to create the link, and the one that loses
/// fails although the file is in place. Test binaries downloading one model
/// from parallel tests hit it on a fresh cache, and so can clients sharing a
/// machine's cache (wienerlabs/leviathan-net#87).
fn lost_link_race(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::AlreadyExists
}

fn sync_io_error(error: &hf_hub::api::sync::ApiError) -> Option<&std::io::Error> {
    match error {
        hf_hub::api::sync::ApiError::IoError(error) => Some(error),
        _ => None,
    }
}

fn tokio_io_error(error: &ApiError) -> Option<&std::io::Error> {
    match error {
        ApiError::IoError(error) => Some(error),
        _ => None,
    }
}

/// Runs `get`, and once more if it lost the race to link its file: the winner
/// has linked it by then, so the second run finds it in the cache. Any other
/// failure, or a second clash, is returned as it is.
fn retry_lost_link_race<T, E>(
    mut get: impl FnMut() -> Result<T, E>,
    io_error: impl Fn(&E) -> Option<&std::io::Error>,
) -> Result<T, E> {
    match get() {
        Err(error) if io_error(&error).is_some_and(lost_link_race) => get(),
        result => result,
    }
}

/// [`retry_lost_link_race`], for the async download.
async fn retry_lost_link_race_async<T, E, F, Fut>(
    mut get: F,
    io_error: impl Fn(&E) -> Option<&std::io::Error>,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    match get().await {
        Err(error) if io_error(&error).is_some_and(lost_link_race) => get().await,
        result => result,
    }
}

fn download_repo_sync(
    repo: Repo,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
    extensions: &[&'static str],
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let builder = hf_hub::api::sync::ApiBuilder::new();
    let cache = match cache {
        Some(cache) => Cache::new(cache),
        None => Cache::default(),
    };
    let api = builder
        .with_cache_dir(cache.path().clone())
        .with_token(token.or(cache.token()))
        .with_progress(progress_bar)
        .build()?
        .repo(repo);
    let res: Result<Vec<PathBuf>, _> = api
        .info()?
        .siblings
        .into_iter()
        .filter(|x| check_extensions(x, extensions))
        .map(|x| retry_lost_link_race(|| api.get(&x.rfilename), sync_io_error))
        .collect();

    res
}

pub fn download_model_repo_sync(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_sync(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Model, revision),
            None => Repo::model(repo_id),
        },
        cache,
        token,
        progress_bar,
        &MODEL_EXTENSIONS,
    )
}

pub fn download_dataset_repo_sync(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_sync(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Dataset, revision),
            None => Repo::new(repo_id, RepoType::Dataset),
        },
        cache,
        token,
        progress_bar,
        &DATASET_EXTENSIONS,
    )
}

#[derive(Debug, Clone)]
pub struct HubUploadInfo {
    pub hub_repo: String,
    pub hub_token: String,
}

pub async fn upload_to_hub(
    hub_info: HubUploadInfo,
    local: Vec<PathBuf>,
    step: u64,
    tx_checkpoint: mpsc::UnboundedSender<model::Checkpoint>,
) -> Result<(), UploadError> {
    let HubUploadInfo {
        hub_repo,
        hub_token,
    } = hub_info;

    info!(repo = hub_repo, "Uploading checkpoint to HuggingFace");

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_token(Some(hub_token.clone()))
        .build()?;
    let repo = Repo::model(hub_repo.clone());
    let api_repo = api.repo(repo);

    let files: Result<Vec<(UploadSource, String)>, _> = local
        .into_iter()
        .map(|path| {
            path.file_name()
                .ok_or(UploadError::NotAFile(path.clone()))
                .and_then(|name| {
                    name.to_str()
                        .ok_or(UploadError::InvalidFilename(path.clone()))
                        .map(|s| s.to_string())
                })
                .map(|name| (path.into(), name))
        })
        .collect();

    let files = files?;

    let commit_info = api_repo
        .upload_files(files, Some(format!("step {step}")), None, false)
        .await
        .map_err(|e| {
            error!(
                repo = hub_repo,
                error = ?e,
                "Failed to upload files to HuggingFace"
            );
            e
        })?;

    let revision = commit_info.oid;

    info!(
        repo = hub_repo,
        revision = revision,
        "Upload to HuggingFace complete"
    );

    tx_checkpoint
        .send(model::Checkpoint::Hub(HubRepo {
            repo_id: FixedString::from_str_truncated(&hub_repo),
            revision: Some(FixedString::from_str_truncated(&revision)),
        }))
        .map_err(|_| UploadError::SendCheckpoint)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::{Error, ErrorKind};

    fn sync_error(kind: ErrorKind) -> hf_hub::api::sync::ApiError {
        hf_hub::api::sync::ApiError::IoError(Error::from(kind))
    }

    fn kind(error: &hf_hub::api::sync::ApiError) -> Option<ErrorKind> {
        sync_io_error(error).map(Error::kind)
    }

    #[test]
    fn a_get_that_lost_the_link_race_is_run_once_more_and_finds_the_file() {
        let calls = Cell::new(0);
        let got = retry_lost_link_race(
            || {
                calls.set(calls.get() + 1);
                match calls.get() {
                    1 => Err(sync_error(ErrorKind::AlreadyExists)),
                    _ => Ok(PathBuf::from("config.json")),
                }
            },
            sync_io_error,
        );
        assert_eq!(got.unwrap(), PathBuf::from("config.json"));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn any_other_failure_is_returned_as_it_is() {
        let calls = Cell::new(0);
        let got: Result<PathBuf, _> = retry_lost_link_race(
            || {
                calls.set(calls.get() + 1);
                Err(sync_error(ErrorKind::NotFound))
            },
            sync_io_error,
        );
        assert_eq!(kind(&got.unwrap_err()), Some(ErrorKind::NotFound));
        assert_eq!(calls.get(), 1);

        let calls = Cell::new(0);
        let got: Result<PathBuf, _> = retry_lost_link_race(
            || {
                calls.set(calls.get() + 1);
                Err(hf_hub::api::sync::ApiError::TooManyRetries(Box::new(
                    sync_error(ErrorKind::AlreadyExists),
                )))
            },
            sync_io_error,
        );
        assert!(kind(&got.unwrap_err()).is_none());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_link_that_still_clashes_on_the_second_run_is_an_error_not_a_loop() {
        let calls = Cell::new(0);
        let got: Result<PathBuf, _> = retry_lost_link_race(
            || {
                calls.set(calls.get() + 1);
                Err(sync_error(ErrorKind::AlreadyExists))
            },
            sync_io_error,
        );
        assert_eq!(kind(&got.unwrap_err()), Some(ErrorKind::AlreadyExists));
        assert_eq!(calls.get(), 2);
    }

    #[tokio::test]
    async fn the_async_download_runs_a_lost_get_once_more_too() {
        let calls = Cell::new(0);
        let got = retry_lost_link_race_async(
            || {
                calls.set(calls.get() + 1);
                let attempt = calls.get();
                async move {
                    match attempt {
                        1 => Err(ApiError::IoError(Error::from(ErrorKind::AlreadyExists))),
                        _ => Ok(PathBuf::from("model.safetensors")),
                    }
                }
            },
            tokio_io_error,
        )
        .await;
        assert_eq!(got.unwrap(), PathBuf::from("model.safetensors"));
        assert_eq!(calls.get(), 2);

        let calls = Cell::new(0);
        let got: Result<PathBuf, _> = retry_lost_link_race_async(
            || {
                calls.set(calls.get() + 1);
                async { Err(ApiError::IoError(Error::from(ErrorKind::PermissionDenied))) }
            },
            tokio_io_error,
        )
        .await;
        assert!(matches!(
            got.unwrap_err(),
            ApiError::IoError(error) if error.kind() == ErrorKind::PermissionDenied
        ));
        assert_eq!(calls.get(), 1);
    }
}
