//! The production path the DisTrO peer-result suites share: the genesis
//! rehearsal config's model, loaded the way a node loads it, and real results
//! from a real step, carried through the network serialization.

#![allow(dead_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::OnceLock;

use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::model::{Checkpoint, Model};
use psyche_core::OptimizerDefinition;
use psyche_data_provider::download_model_repo_sync;
use psyche_modeling::{
    CausalLM, Distro, DistroResult, Trainer, auto_model_for_causal_lm_from_pretrained,
};
use psyche_network::SerializedDistroResult;
use tch::{Device, Kind, Tensor};

#[derive(serde::Deserialize)]
pub struct RunConfig {
    #[allow(dead_code)]
    config: CoordinatorConfig,
    model: Model,
}

/// The model and the DisTrO settings the genesis rehearsal runs with.
#[derive(Clone)]
pub struct Recipe {
    pub repo_files: Vec<PathBuf>,
    pub max_seq_len: usize,
    pub compression_decay: f64,
    pub compression_chunk: i64,
    pub compression_topk: i64,
    pub quantize_1bit: bool,
    pub robust_aggregation: bool,
    pub lr: f64,
}

impl Recipe {
    /// The same run, aggregating by `robust_aggregation` instead.
    pub fn aggregating(&self, robust_aggregation: bool) -> Recipe {
        Recipe {
            robust_aggregation,
            ..self.clone()
        }
    }
}

/// The genesis rehearsal's recipe, read and downloaded once per test binary.
/// The tests of one binary run in parallel, and each downloading the model
/// into an empty cache used to race (wienerlabs/leviathan-net#87). A test
/// that asks while the first download runs waits for it.
pub fn genesis_recipe() -> Recipe {
    static RECIPE: OnceLock<Recipe> = OnceLock::new();
    RECIPE.get_or_init(read_genesis_recipe).clone()
}

fn read_genesis_recipe() -> Recipe {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/leviathan/genesis-rehearsal.toml"
    );
    let run: RunConfig = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let Model::LLM(llm) = run.model;
    let Checkpoint::Hub(hub) = llm.checkpoint else {
        panic!("the genesis rehearsal starts from a Hub checkpoint");
    };
    let OptimizerDefinition::Distro {
        compression_decay,
        compression_chunk,
        compression_topk,
        quantize_1bit,
        robust_aggregation,
        ..
    } = llm.optimizer
    else {
        panic!("the genesis rehearsal trains with DisTrO");
    };
    let repo_files = download_model_repo_sync(
        &hub.repo_id.to_string(),
        hub.revision.map(|revision| revision.to_string()),
        None,
        None,
        false,
    )
    .expect("the genesis rehearsal model is downloadable");
    Recipe {
        repo_files,
        max_seq_len: llm.max_seq_len as usize,
        compression_decay: compression_decay as f64,
        compression_chunk: compression_chunk as i64,
        compression_topk: compression_topk as i64,
        quantize_1bit,
        robust_aggregation,
        lr: llm.lr_schedule.get_lr(llm.lr_schedule.get_warmup_steps()),
    }
}

pub fn load(recipe: &Recipe) -> Box<dyn CausalLM> {
    let model = auto_model_for_causal_lm_from_pretrained(
        recipe.repo_files.clone(),
        Some(Kind::Float),
        None,
        Some(Device::Cpu),
        None,
        Some(recipe.max_seq_len),
    )
    .unwrap();
    model.prepare_for_training();
    model
}

pub fn distro(recipe: &Recipe, model: &dyn CausalLM) -> Distro {
    Distro::new(
        model,
        recipe.compression_decay,
        recipe.compression_chunk,
        recipe.compression_topk,
        0.0,
        recipe.robust_aggregation,
        recipe.quantize_1bit,
    )
}

/// What a peer broadcasts for one batch: a real step on `tokens`, as received.
pub fn peer_results(recipe: &Recipe, tokens: &[i64]) -> Vec<DistroResult> {
    peer_results_quantized(recipe, tokens, recipe.quantize_1bit)
}

pub fn peer_results_quantized(
    recipe: &Recipe,
    tokens: &[i64],
    quantize: bool,
) -> Vec<DistroResult> {
    let model = load(recipe);
    let mut distro = distro(recipe, &*model);
    let input = Tensor::from_slice(tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    let results = distro.generate(&*model, &[], 0.0, recipe.lr, false);
    let results = match quantize {
        true => Trainer::quantize_results(&results),
        false => results,
    };
    results
        .iter()
        .map(|result| {
            DistroResult::try_from(&SerializedDistroResult::try_from(result).unwrap()).unwrap()
        })
        .collect()
}

/// A batch of the model's own vocabulary, a different one per `seed`.
pub fn batch(recipe: &Recipe, seed: i64) -> Vec<i64> {
    let model = load(recipe);
    let vocab = model
        .variables()
        .find(|variable| variable.name().ends_with("embed_tokens.weight"))
        .expect("the model has a token embedding")
        .full_tensor_shape()[0];
    (0..recipe.max_seq_len as i64)
        .map(|i| (i * 7 + seed) % vocab)
        .collect()
}

/// A node as it stands when it applies a round's results: it has trained its
/// own batch, so its gradients exist, and its optimizer starts from zero.
/// Training leaves the parameters where they were loaded.
pub fn trained_node(recipe: &Recipe) -> (Box<dyn CausalLM>, Distro) {
    let model = load(recipe);
    let tokens = batch(recipe, NODE_BATCH_SEED);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    let distro = distro(recipe, &*model);
    (model, distro)
}

/// The batch the applying node trains on itself, apart from its peers'.
pub const NODE_BATCH_SEED: i64 = 5;

/// The model's parameters after a trained node applies `peers`, or the panic
/// `apply` raised.
pub fn applied(recipe: &Recipe, peers: &[Vec<DistroResult>]) -> Result<Vec<Tensor>, String> {
    let (model, mut distro) = trained_node(recipe);
    catch_unwind(AssertUnwindSafe(|| distro.apply(&*model, peers, recipe.lr))).map_err(
        |panic| {
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        },
    )?;
    Ok(model
        .variables()
        .map(|variable| variable.logical_tensor().copy())
        .collect())
}

pub fn fresh(recipe: &Recipe) -> Vec<Tensor> {
    load(recipe)
        .variables()
        .map(|variable| variable.logical_tensor().copy())
        .collect()
}

pub fn same(a: &[Tensor], b: &[Tensor]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.equal(b))
}

pub fn rebuilt(result: &DistroResult, sparse_idx: Tensor, sparse_val: Tensor) -> DistroResult {
    DistroResult {
        sparse_idx,
        sparse_val,
        xshape: result.xshape.clone(),
        totalk: result.totalk,
        stats: None,
    }
}
