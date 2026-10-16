use anyhow::{Result, anyhow, bail};
use psyche_core::{BatchId, ClosedInterval, Shuffle, TokenSize};
use std::fs;
use tracing::info;

use crate::{
    TokenizedData,
    file_extensions::DATA_FILE_EXTENSIONS,
    http::{SequencePointer, sequence_pointers},
    traits::{LengthKnownDataProvider, TokenizedDataProvider},
};

fn is_truthy_env_bool(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "1" | "true" | "yes")
}

fn mmap_file(p: &std::path::PathBuf) -> Result<Box<dyn AsRef<[u8]> + Send>> {
    let file = std::fs::File::open(p)?;

    // try to mmap first, only falling back to read if allowed
    match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(mmap) => Ok(Box::new(mmap)),
        Err(e)
            if e.raw_os_error() == Some(22)
                && std::env::var("ALLOW_FAIL_MMAP")
                    .map(|v| is_truthy_env_bool(&v))
                    .unwrap_or(false) =>
        {
            eprintln!("mmap failed (likely under Valgrind), falling back to file read");
            let data = std::fs::read(p)?;
            Ok(Box::new(data))
        }
        Err(e) => Err(e.into()),
    }
}

pub struct LocalDataProvider {
    data_files: Vec<Box<dyn AsRef<[u8]> + Send>>,
    sequences: Vec<SequencePointer>,
    seq_len: usize,
    token_size_in_bytes: TokenSize,
}

impl LengthKnownDataProvider for LocalDataProvider {
    fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
}

impl LocalDataProvider {
    pub fn new_from_directory(
        dir: impl AsRef<std::path::Path>,
        token_size_in_bytes: TokenSize,
        num_tokens_per_sequence: usize, // num tokens per sequence
        shuffle: Shuffle,
    ) -> Result<Self> {
        let dir = std::fs::canonicalize(&dir)
            .map_err(|e| anyhow!("Failed to open data directory {:?}: {e}", dir.as_ref()))?;
        let mut bin_files = vec![];
        for file in std::fs::read_dir(&dir)
            .map_err(|e| anyhow!("couldn't load training data from {}: {e}", dir.display()))?
            .flatten()
        {
            let file = file.path();
            if let Some(extension) = file.extension().and_then(|s| s.to_str()) {
                if DATA_FILE_EXTENSIONS.contains(&extension) {
                    bin_files.push(file);
                }
            }
        }
        bin_files.sort();
        let data_files = bin_files
            .iter()
            .map(mmap_file)
            .collect::<Result<Vec<_>>>()?;

        if data_files.is_empty() {
            bail!("No training data files in directory {:?}", dir);
        }

        info!(
            "Loaded {} files ({}) of training data from directory {}",
            bin_files.len(),
            bin_files
                .iter()
                .map(|f| fs::metadata(f).unwrap().len())
                .sum::<u64>(),
            dir.display()
        );

        let sequences = sequence_pointers(
            bin_files.iter().map(|path| path.display()).zip(
                data_files
                    .iter()
                    .map(|file| file.as_ref().as_ref().len() as u64),
            ),
            token_size_in_bytes,
            num_tokens_per_sequence as u64,
            shuffle,
        )?;

        Ok(Self {
            data_files,
            sequences,
            seq_len: num_tokens_per_sequence,
            token_size_in_bytes,
        })
    }

    fn internal_get_samples(&self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        let mut ret: Vec<_> = Vec::new();
        for data_id in data_ids.iter() {
            let SequencePointer {
                byte_offset,
                file_index,
            } = self.sequences.get(data_id as usize).ok_or_else(|| {
                anyhow!(
                    "index {data_id} is out of bounds, we only have {} samples.",
                    self.sequences.len()
                )
            })?;

            let file = &self.data_files[*file_index];
            let data_len = usize::from(self.token_size_in_bytes) * self.seq_len;
            let data = &file.as_ref().as_ref()[*byte_offset..*byte_offset + data_len];

            let tokens: Vec<i32> = data
                .chunks(self.token_size_in_bytes.into())
                .map(|t| {
                    use TokenSize::*;
                    match self.token_size_in_bytes {
                        TwoBytes => u16::from_le_bytes(t.try_into().unwrap()) as i32,
                        FourBytes => u32::from_le_bytes(t.try_into().unwrap()) as i32,
                    }
                })
                .collect();
            ret.push(TokenizedData::from_input_ids(tokens));
        }
        Ok(ret)
    }
}

impl TokenizedDataProvider for LocalDataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        self.internal_get_samples(data_ids)
    }
}

pub struct LocalDataProviderIter {
    provider: LocalDataProvider,
    current_index: u64,
}

impl Iterator for LocalDataProviderIter {
    type Item = TokenizedData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.provider.num_sequences() as u64 {
            let result = self
                .provider
                .internal_get_samples(BatchId(ClosedInterval::new(
                    self.current_index,
                    self.current_index,
                )))
                .unwrap()
                .pop()
                .unwrap();
            self.current_index += 1;
            Some(result)
        } else {
            None
        }
    }
}

impl IntoIterator for LocalDataProvider {
    type Item = TokenizedData;
    type IntoIter = LocalDataProviderIter;

    fn into_iter(self) -> Self::IntoIter {
        LocalDataProviderIter {
            provider: self,
            current_index: 0,
        }
    }
}
