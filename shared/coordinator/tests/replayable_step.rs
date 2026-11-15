//! Which step a verifier can replay from public data
//! (wienerlabs/leviathan-net#47).

use bytemuck::Zeroable;
use psyche_coordinator::Coordinator;

const START_STEP: u32 = 40;

/// The `epoch`-th epoch of a run, started at `START_STEP`, in its round of
/// height `height`, flagged as a cold start or not.
fn coordinator(epoch: u16, cold_start: bool, height: u32) -> Box<Coordinator> {
    let mut coordinator = Box::new(Coordinator::zeroed());
    coordinator.progress.epoch = epoch;
    coordinator.epoch_state.cold_start_epoch = cold_start.into();
    coordinator.epoch_state.start_step = START_STEP;
    coordinator.progress.step = START_STEP + height;
    let head = coordinator.epoch_state.rounds_head as usize;
    coordinator.epoch_state.rounds[head].height = height;
    coordinator
}

/// Every node trains the run's first step from the configured checkpoint with
/// a trainer it has just built, in whichever round of the first epoch the
/// replay is asked.
#[test]
fn the_runs_first_step_is_replayable() {
    for height in [0, 3] {
        assert_eq!(
            coordinator(0, false, height).replayable_step(),
            Some(START_STEP)
        );
        assert_eq!(
            coordinator(0, true, height).replayable_step(),
            Some(START_STEP)
        );
    }
}

/// A running node carries its model and its error feedback through cooldown,
/// warmup and pauses rather than reloading the checkpoint, so no step of a
/// later epoch can be replayed from it, cold start or not.
#[test]
fn no_step_of_a_later_epoch_is_replayable() {
    for cold_start in [false, true] {
        for height in [0, 3] {
            assert_eq!(coordinator(1, cold_start, height).replayable_step(), None);
            assert_eq!(coordinator(7, cold_start, height).replayable_step(), None);
        }
    }
}
