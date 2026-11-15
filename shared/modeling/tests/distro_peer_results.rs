//! A peer's DisTrO results are applied only if they fit the local model
//! (wienerlabs/leviathan-net#41).
//!
//! Everything here is the production path: the genesis rehearsal config's model
//! loaded the way a node loads it, a real forward and backward pass, results
//! from `Distro::generate`, quantized as the config says, and carried through
//! the network serialization. The malformed payloads are what a trainer could
//! commit and sign: its honest results, cut or rearranged.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use psyche_coordinator::model::{Checkpoint, Model};
use psyche_coordinator::CoordinatorConfig;
use psyche_core::OptimizerDefinition;
use psyche_data_provider::download_model_repo_sync;
use psyche_modeling::{
    auto_model_for_causal_lm_from_pretrained, CausalLM, CompressDCT, Distro, DistroResult,
    DistroResultMismatch, Trainer, TransformDCT,
};
use psyche_network::SerializedDistroResult;
use tch::{Device, Kind, Tensor};

#[derive(serde::Deserialize)]
struct RunConfig {
    #[allow(dead_code)]
    config: CoordinatorConfig,
    model: Model,
}

/// The model and the DisTrO settings the genesis rehearsal runs with.
struct Recipe {
    repo_files: Vec<PathBuf>,
    max_seq_len: usize,
    compression_decay: f64,
    compression_chunk: i64,
    compression_topk: i64,
    quantize_1bit: bool,
    lr: f64,
}

fn genesis_recipe() -> Recipe {
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
        lr: llm.lr_schedule.get_lr(llm.lr_schedule.get_warmup_steps()),
    }
}

fn load(recipe: &Recipe) -> Box<dyn CausalLM> {
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

fn distro(recipe: &Recipe, model: &dyn CausalLM) -> Distro {
    Distro::new(
        model,
        recipe.compression_decay,
        recipe.compression_chunk,
        recipe.compression_topk,
        0.0,
    )
}

/// What a peer broadcasts for one batch: a real step on `tokens`, as received.
fn peer_results(recipe: &Recipe, tokens: &[i64]) -> Vec<DistroResult> {
    peer_results_quantized(recipe, tokens, recipe.quantize_1bit)
}

fn peer_results_quantized(recipe: &Recipe, tokens: &[i64], quantize: bool) -> Vec<DistroResult> {
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
fn batch(recipe: &Recipe, seed: i64) -> Vec<i64> {
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
fn trained_node(recipe: &Recipe) -> (Box<dyn CausalLM>, Distro) {
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
const NODE_BATCH_SEED: i64 = 5;

/// The model's parameters after a trained node applies `peers`, or the panic
/// `apply` raised.
fn applied(recipe: &Recipe, peers: &[Vec<DistroResult>]) -> Result<Vec<Tensor>, String> {
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

fn fresh(recipe: &Recipe) -> Vec<Tensor> {
    load(recipe)
        .variables()
        .map(|variable| variable.logical_tensor().copy())
        .collect()
}

fn same(a: &[Tensor], b: &[Tensor]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.equal(b))
}

fn rebuilt(result: &DistroResult, sparse_idx: Tensor, sparse_val: Tensor) -> DistroResult {
    DistroResult {
        sparse_idx,
        sparse_val,
        xshape: result.xshape.clone(),
        totalk: result.totalk,
        stats: None,
    }
}

/// One result short.
fn fewer_results(honest: &[DistroResult]) -> Vec<DistroResult> {
    honest[..honest.len() - 1].to_vec()
}

/// Two variables of different shapes trade results.
fn swapped_shapes(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let other = (1..results.len())
        .find(|i| results[*i].xshape != results[0].xshape)
        .expect("the model has variables of different shapes");
    results.swap(0, other);
    results
}

/// Every index of one variable points one past the end of its encoding, in
/// the 8-bit wire encoding a `totalk` below 256 is sent in.
fn indices_outside(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let target = results
        .iter()
        .position(|r| r.totalk < 256)
        .expect("the model has a variable encoded with 8-bit indices");
    let result = &results[target];
    assert_eq!(result.sparse_idx.kind(), Kind::Uint8);
    let outside = Tensor::full(
        result.sparse_idx.size(),
        result.totalk,
        (Kind::Uint8, Device::Cpu),
    );
    results[target] = rebuilt(result, outside, result.sparse_val.copy());
    results
}

/// One variable carries one value fewer than it has indices.
fn counts_differ(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let result = &results[0];
    let last = *result.sparse_val.size().last().unwrap();
    let short = result.sparse_val.narrow(-1, 0, last - 1);
    results[0] = rebuilt(result, result.sparse_idx.copy(), short);
    results
}

#[test]
fn a_peer_whose_results_do_not_fit_the_model_is_skipped() {
    let recipe = genesis_recipe();
    let honest = peer_results(&recipe, &batch(&recipe, 3));
    let baseline = applied(&recipe, std::slice::from_ref(&honest)).expect("honest results apply");

    for (name, malformed) in [
        ("fewer results than variables", fewer_results(&honest)),
        ("another variable's shape", swapped_shapes(&honest)),
        ("indices outside totalk", indices_outside(&honest)),
        ("index and value counts that differ", counts_differ(&honest)),
    ] {
        for peers in [
            vec![honest.clone(), malformed.clone()],
            vec![malformed.clone(), honest.clone()],
        ] {
            let after = applied(&recipe, &peers)
                .unwrap_or_else(|panic| panic!("a peer with {name} crashed apply: {panic}"));
            assert!(
                same(&after, &baseline),
                "a peer with {name} changed the model"
            );
        }
        let alone = applied(&recipe, &[malformed])
            .unwrap_or_else(|panic| panic!("a peer with {name} crashed apply alone: {panic}"));
        assert!(
            same(&alone, &fresh(&recipe)),
            "a lone peer with {name} moved the model"
        );
    }
}

#[test]
fn a_second_valid_peer_is_still_applied() {
    let recipe = genesis_recipe();
    let first = peer_results(&recipe, &batch(&recipe, 3));
    let second = peer_results(&recipe, &batch(&recipe, 11));
    let one = applied(&recipe, std::slice::from_ref(&first)).unwrap();
    let two = applied(&recipe, &[first, second]).unwrap();
    assert!(
        !same(&one, &two),
        "a second valid peer's results were not applied"
    );
}

/// Each malformed payload is refused for its own reason, and the honest one is
/// accepted, so the check is not merely refusing everything unusual.
#[test]
fn each_payload_that_does_not_fit_is_named_for_why() {
    let recipe = genesis_recipe();
    let honest = peer_results(&recipe, &batch(&recipe, 3));
    let (model, distro) = trained_node(&recipe);
    assert_eq!(distro.check_results(&*model, &honest), Ok(()));

    assert!(matches!(
        distro.check_results(&*model, &fewer_results(&honest)),
        Err(DistroResultMismatch::Count { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &swapped_shapes(&honest)),
        Err(DistroResultMismatch::Shape { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &indices_outside(&honest)),
        Err(DistroResultMismatch::Indices { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &counts_differ(&honest)),
        Err(DistroResultMismatch::Values { .. })
    ));
}

/// Without 1-bit quantization the values travel as floating point, and a peer
/// that sends a NaN would turn every replica's parameters to NaN through the
/// sign step. It is skipped like any other payload that does not fit.
#[test]
fn a_peer_with_non_finite_values_is_skipped() {
    let recipe = genesis_recipe();
    let honest = peer_results_quantized(&recipe, &batch(&recipe, 3), false);
    let mut poisoned = honest.clone();
    let target = &poisoned[0];
    let nan = target.sparse_val.full_like(f64::NAN);
    poisoned[0] = rebuilt(target, target.sparse_idx.copy(), nan);

    let (model, distro) = trained_node(&recipe);
    assert_eq!(distro.check_results(&*model, &honest), Ok(()));
    assert!(matches!(
        distro.check_results(&*model, &poisoned),
        Err(DistroResultMismatch::Values { .. })
    ));

    let baseline = applied(&recipe, std::slice::from_ref(&honest)).unwrap();
    let after = applied(&recipe, &[honest, poisoned]).unwrap();
    assert!(same(&after, &baseline), "a NaN peer changed the model");
}

/// The encoding the check expects for each variable is exactly what `encode`
/// and `compress` produce for it, on every variable of the real model.
#[test]
fn the_expected_encoding_is_the_real_one() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut transform = TransformDCT::new(model.variables(), recipe.compression_chunk);
    for variable in model.variables() {
        let shape = variable.logical_tensor().size();
        let expected = transform
            .encoded_shape(&shape)
            .unwrap_or_else(|| panic!("{} has no encoding", variable.name()));
        let encoded =
            transform.encode(&Tensor::zeros(shape.as_slice(), (Kind::Float, Device::Cpu)));
        assert_eq!(encoded.size(), expected, "{}", variable.name());
        let (_, _, xshape, totalk) = CompressDCT::compress(&encoded, recipe.compression_topk);
        assert_eq!(xshape, expected, "{}", variable.name());
        let flat_last = match expected.len() {
            n if n > 2 => expected[n - 2] * expected[n - 1],
            n => expected[n - 1],
        };
        assert_eq!(totalk, flat_last, "{}", variable.name());
    }
}
