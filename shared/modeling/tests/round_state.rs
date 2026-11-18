//! The state a node trains a round from, and the digest it commits to
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`), on the genesis
//! rehearsal's model.

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use common::{Recipe, batch, distro, genesis_recipe, load, load_in, round_trainer, train_round};
use psyche_modeling::{
    CausalLM, Distro, ROUND_STATE_DOMAIN, keep_round_state, round_state_digest, round_state_file,
    round_state_len, write_round_state,
};
use tch::{Kind, Tensor};

/// A step of `distro` on the batch for `seed`: what fills the buffer.
fn step(recipe: &Recipe, model: &dyn CausalLM, distro: &mut Distro, seed: i64) {
    let tokens = batch(recipe, seed);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    distro.generate(model, &[], 0.0, recipe.lr, false);
}

/// Reads the tensors of one section of an encoding at `at`: their names and
/// the number of bytes each holds.
fn section(bytes: &[u8], at: &mut usize) -> Vec<(String, usize)> {
    let mut read = |len: usize| {
        let slice = &bytes[*at..*at + len];
        *at += len;
        slice
    };
    let count = u32::from_le_bytes(read(4).try_into().unwrap());
    (0..count)
        .map(|_| {
            let name_len = u32::from_le_bytes(read(4).try_into().unwrap()) as usize;
            let name = String::from_utf8(read(name_len).to_vec()).unwrap();
            let _tag = read(1)[0];
            let dims = u32::from_le_bytes(read(4).try_into().unwrap());
            let shape: Vec<i64> = (0..dims)
                .map(|_| i64::from_le_bytes(read(8).try_into().unwrap()))
                .collect();
            let len = u64::from_le_bytes(read(8).try_into().unwrap()) as usize;
            let _data = read(len);
            let numel: i64 = shape.iter().product();
            assert_eq!(len % numel.max(1) as usize, 0, "{name}");
            (name, len)
        })
        .collect()
}

/// The digest is the BLAKE3 hash of the encoding, which holds every variable
/// of the model, whole, and then the buffer of each, and nothing else.
#[test]
fn the_digest_is_the_hash_of_an_encoding_of_every_variable_and_buffer() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);
    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &node).unwrap();
    assert!(bytes.starts_with(ROUND_STATE_DOMAIN));
    assert_eq!(
        round_state_digest(&*model, &node).unwrap(),
        *blake3::hash(&bytes).as_bytes()
    );

    let mut at = ROUND_STATE_DOMAIN.len();
    let weights = section(&bytes, &mut at);
    let buffers = section(&bytes, &mut at);
    assert_eq!(at, bytes.len(), "the encoding holds nothing else");
    let variables: Vec<_> = model.variables().collect();
    assert_eq!(weights.len(), variables.len());
    assert_eq!(buffers.len(), variables.len());
    for ((variable, (name, len)), (buffer_name, buffer_len)) in
        variables.iter().zip(&weights).zip(&buffers)
    {
        assert_eq!(name, variable.name());
        assert_eq!(buffer_name, &format!("{}.delta", variable.name()));
        let full: i64 = variable.full_tensor_shape().iter().product();
        let tensor = variable.logical_tensor();
        assert_eq!(*len, full as usize * tensor.kind().elt_size_in_bytes());
        assert_eq!(buffer_len, len);
    }
}

/// Two nodes in one state commit to one digest, and a node's digest follows
/// its weights and its buffer.
#[test]
fn the_digest_names_the_state_and_follows_it() {
    let recipe = genesis_recipe();
    let one = load(&recipe);
    let other = load(&recipe);
    let mut one_distro = distro(&recipe, &*one);
    let other_distro = distro(&recipe, &*other);
    let fresh = round_state_digest(&*one, &one_distro).unwrap();
    assert_eq!(fresh, round_state_digest(&*other, &other_distro).unwrap());
    assert_eq!(fresh, round_state_digest(&*one, &one_distro).unwrap());

    // A step fills the buffer: another state.
    step(&recipe, &*one, &mut one_distro, 1);
    let stepped = round_state_digest(&*one, &one_distro).unwrap();
    assert_ne!(stepped, fresh);

    // One weight moved: another state again.
    {
        let _no_grad = tch::no_grad_guard();
        let variable = one.variables().next().unwrap();
        let mut tensor = variable.logical_tensor();
        let _ = tensor.g_add_scalar_(1.0);
    }
    assert_ne!(round_state_digest(&*one, &one_distro).unwrap(), stepped);
}

/// A trainer commits each round to the state the round trains from. Its first
/// round trains from the checkpoint and a zeroed buffer, which anyone can
/// rebuild, and the digest it reports is that state's. Its next round trains
/// from another state, and a second trainer that went through the same rounds
/// reports the same digests.
#[test]
fn a_trainer_commits_to_the_state_each_round_trains_from() {
    let recipe = genesis_recipe();
    let checkpoint = load(&recipe);
    let zeroed = distro(&recipe, &*checkpoint);
    let first_state = round_state_digest(&*checkpoint, &zeroed).unwrap();

    let mut digests = vec![];
    for _ in 0..2 {
        let first = train_round(&recipe, round_trainer(&recipe, None), 1, 1, vec![]);
        assert_eq!(first.state_digest, Some(first_state));
        let sent = first.distro_results.clone().unwrap();
        let second = train_round(&recipe, first.trainer, 2, 2, vec![sent]);
        let second_state = second.state_digest.expect("the trainer commits");
        assert_ne!(second_state, first_state);
        digests.push(second_state);
    }
    assert_eq!(digests[0], digests[1]);
}

/// The files in `dir`, by name.
fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    files
}

/// A kept state is its encoding, whole, in a file named by its digest, and
/// the digest is the one the round commits to. Nothing else is left beside
/// it.
#[test]
fn a_kept_state_is_its_encoding_in_a_file_named_by_its_digest() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);
    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &node).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let kept = keep_round_state(dir.path(), &*model, &node).unwrap();
    assert_eq!(kept.digest, round_state_digest(&*model, &node).unwrap());
    let file = kept.file.unwrap();
    assert_eq!(file, round_state_file(dir.path(), &kept.digest));
    assert_eq!(std::fs::read(&file).unwrap(), bytes);
    assert_eq!(files_in(dir.path()), vec![file]);
}

/// A state that cannot be kept is still committed to, by the digest of the
/// state the round trains from, and the failure is reported, not hidden.
#[test]
fn a_state_that_cannot_be_kept_is_still_committed_to() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);

    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    let kept = keep_round_state(&missing, &*model, &node).unwrap();
    assert_eq!(kept.digest, round_state_digest(&*model, &node).unwrap());
    let err = kept.file.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(!missing.exists());
    assert!(files_in(dir.path()).is_empty());
}

/// The length of a state follows from the model's shapes and dtypes alone:
/// for a model trained in fp32, one in bf16, and one in bf16 whose buffer is
/// kept in fp32 (wienerlabs/leviathan-net#90).
#[test]
fn a_states_length_is_known_from_its_shapes_alone() {
    let recipe = genesis_recipe();
    for (kind, buffer_kind) in [
        (Kind::Float, None),
        (Kind::BFloat16, None),
        (Kind::BFloat16, Some(Kind::Float)),
    ] {
        let model = load_in(&recipe, kind);
        let mut node = distro(&recipe, &*model);
        if let Some(buffer_kind) = buffer_kind {
            node.set_error_feedback_kind(buffer_kind).unwrap();
        }
        step(&recipe, &*model, &mut node, 1);
        let mut bytes = vec![];
        write_round_state(&mut bytes, &*model, &node).unwrap();
        assert_eq!(
            round_state_len(&*model, &node).unwrap(),
            bytes.len() as u64,
            "{kind:?} with its buffer in {buffer_kind:?}"
        );
    }
}

/// A trainer keeping its states writes each round's into a file named by the
/// digest it reports, holding the state that digest names: the checkpoint
/// and a zeroed buffer first, then the next round's state.
#[test]
fn a_trainer_keeps_the_state_each_round_trains_from() {
    let recipe = genesis_recipe();
    let checkpoint = load(&recipe);
    let zeroed = distro(&recipe, &*checkpoint);
    let mut first_bytes = vec![];
    write_round_state(&mut first_bytes, &*checkpoint, &zeroed).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let trainer = round_trainer(&recipe, Some(dir.path().to_path_buf()));
    let first = train_round(&recipe, trainer, 1, 1, vec![]);
    let first_state = first.state_digest.expect("the trainer commits");
    let first_file = round_state_file(dir.path(), &first_state);
    assert_eq!(std::fs::read(&first_file).unwrap(), first_bytes);

    let sent = first.distro_results.clone().unwrap();
    let second = train_round(&recipe, first.trainer, 2, 2, vec![sent]);
    let second_state = second.state_digest.expect("the trainer commits");
    assert_ne!(second_state, first_state);
    let second_file = round_state_file(dir.path(), &second_state);
    let second_bytes = std::fs::read(&second_file).unwrap();
    assert_eq!(*blake3::hash(&second_bytes).as_bytes(), second_state);
    assert_eq!(second_bytes.len(), first_bytes.len());

    let mut expected = vec![first_file, second_file];
    expected.sort();
    assert_eq!(files_in(dir.path()), expected);
}

/// A trainer says how long the state it trains from is, as a verifier works
/// it out from its own copy of the model, and the state it keeps is that
/// long.
#[test]
fn a_trainer_says_how_long_its_state_is() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let expected = round_state_len(&*model, &distro(&recipe, &*model)).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let mut trainer = round_trainer(&recipe, Some(dir.path().to_path_buf()));
    assert_eq!(trainer.round_state_len().unwrap(), Some(expected));
    let first = train_round(&recipe, trainer, 1, 1, vec![]);
    let file = round_state_file(dir.path(), &first.state_digest.unwrap());
    assert_eq!(std::fs::metadata(file).unwrap().len(), expected);
}

/// What committing costs a node every round on the genesis rehearsal's model:
/// copying the state off the device and hashing it. Printed for the record.
#[test]
fn what_committing_to_a_round_costs() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);
    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &node).unwrap();
    let rounds = 20;
    let started = Instant::now();
    for _ in 0..rounds {
        round_state_digest(&*model, &node).unwrap();
    }
    let each = started.elapsed() / rounds;
    println!(
        "a round's state: {} bytes, committed to in {each:?} on the CPU",
        bytes.len()
    );
}
