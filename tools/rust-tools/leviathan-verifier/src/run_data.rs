//! A run's training data, read the way a node reads it.

use std::path::Path;

use anyhow::anyhow;
use anyhow::Result;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLM;
use psyche_core::BatchId;
use psyche_data_provider::http::FileURLs;
use psyche_data_provider::http::HttpDataProvider;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedData;
use psyche_data_provider::TokenizedDataProvider;

use crate::replay::replay_data_layout;

/// A run's data: a local copy, or the run's own HTTP location.
pub enum RunData {
    Local(LocalDataProvider),
    Http(HttpDataProvider),
}

impl RunData {
    /// The data `llm` trains on, read from `data_dir` in the run's layout if
    /// one is given, else from the run's HTTP location, as a node reads it.
    pub async fn open(llm: &LLM, data_dir: Option<&Path>) -> Result<Self> {
        let (token_size, shuffle) = replay_data_layout(llm)?;
        let sequence_length = llm.max_seq_len;
        match data_dir {
            Some(dir) => Ok(Self::Local(LocalDataProvider::new_from_directory(
                dir,
                token_size,
                sequence_length as usize,
                shuffle,
            )?)),
            None => {
                let LLMTrainingDataLocation::Http(http) = llm.data_location else {
                    return Err(anyhow!("the run's data is not at an HTTP location"));
                };
                let urls = FileURLs::from_location(&http.location).await?;
                Ok(Self::Http(HttpDataProvider::new(
                    urls,
                    token_size,
                    sequence_length,
                    shuffle,
                )?))
            }
        }
    }
}

impl TokenizedDataProvider for RunData {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        match self {
            Self::Local(provider) => provider.get_samples(data_ids).await,
            Self::Http(provider) => provider.get_samples(data_ids).await,
        }
    }
}
